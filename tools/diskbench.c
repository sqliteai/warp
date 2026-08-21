/*
 * diskbench.c — measure the I/O path WASTE actually uses.
 *
 * The engine reads whole experts as ~12 MB records scattered across a huge
 * file, with the page cache bypassed (F_NOCACHE / O_DIRECT), so sequential
 * `dd` numbers are misleading. This measures:
 *   1. sequential write   (how fast the download/conversion can land)
 *   2. sequential read, cache-bypassed
 *   3. random record reads, cache-bypassed  <- the number that sets tok/s
 *   4. the same with N threads (the engine's async pool)
 *
 * Every row says whether the bypass was actually obtained, because on a
 * filesystem that refuses it these are RAM numbers wearing a disk's label.
 *
 * The last argument turns GB/s into tok/s for a model that reads that many
 * GB per token cold. It has no default: the column used to assume K3's 12.5
 * unconditionally, which is ~8x off on a 48B model (1.61 GB/token measured)
 * and says so nowhere. `waste bench` prints the real figure for a container
 * as "disk N GB total, M GB/token"; pass M here, or leave it out and read
 * the GB/s.
 *
 * Build: cc -O2 -o diskbench tools/diskbench.c -lpthread
 *   Windows cross:  x86_64-w64-mingw32-gcc-posix -O2 -o diskbench.exe \
 *                       tools/diskbench.c -lpthread
 *   The -posix driver is not optional there: mingw ships win32 and posix
 *   threads as separate compilers and only -posix has pthread.h, the same
 *   reason the Makefile names it for the engine.
 * Usage: ./diskbench /Volumes/WasteDisk/k3/.bench [file_gb] [rec_mb] [threads] [gb_per_token]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#ifndef _WIN32
#include <unistd.h>
#endif

/* The engine's six non-POSIX calls, rather than a second copy of them here.
 * This file reimplemented the I/O layer in raw POSIX and so did not compile
 * on Windows at all — posix_memalign, pread and fsync all implicit (#36,
 * gap 4) — while src/platform.h has had waste_pread and waste_aligned_alloc
 * since the port. The engine's own bypass was never affected: bank_open
 * goes through waste_open_stream and CreateFileA with
 * FILE_FLAG_NO_BUFFERING. Only this tool was left behind, which is the same
 * shape as issue #22, where the tool reported the page cache on Linux for
 * the whole window after the engine's bypass was fixed. */
#include "../src/platform.h"

static double now(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

/* The bypass, one mechanism under three names — the same contract bank_open
 * meets in model.c, and for the same reason: without it the kernel serves the
 * file back out of RAM and this tool reports the page cache. It did exactly
 * that on Linux until now, where O_DIRECT appeared in the comment above and
 * nowhere in the code, so a Gen3 x4 drive benched at 44 GB/s sequential
 * against a 3.9 GB/s link (issue #22 — the same bug LEARNED §14 fixed in the
 * engine, in the one place it was left behind).
 *
 * O_DIRECT wants offset, length and buffer aligned to the device's logical
 * block. All three come free here: buffers are posix_memalign'd to 4096, the
 * record size is masked to a 4096 multiple, and offsets are whole records. */
#define DIO_ALIGN 4096u

/* Cleared by any open that could not get the bypass, including the reader
 * threads'. Every one of them stores the same value and none reads it back,
 * so ordering does not matter — it is atomic to keep the race out of the
 * abstract machine, not because anything here needs to synchronise. */
static _Atomic int g_direct = 1;

static const char *g_path;
static size_t g_rec, g_file;
static int g_reps;

static int bench_sync(int fd) {
#ifdef _WIN32
    return FlushFileBuffers((HANDLE)_get_osfhandle(fd)) ? 0 : -1;
#else
    return fsync(fd);
#endif
}

/* The probe and its two helpers exist only for the platforms whose bypass
 * accepts the open and then refuses the transfer. macOS's F_NOCACHE has no
 * alignment contract, so it has nothing to probe and these are not compiled
 * there. */
#if defined(_WIN32) || (defined(__linux__) && defined(O_DIRECT))
/* One positional write, the counterpart of platform.h's waste_pread. Only
 * the probe needs it, so it lives here rather than in the engine's header. */
static int64_t bench_pwrite(int fd, const void *src, size_t n, int64_t off) {
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    OVERLAPPED ov = {0};
    ov.Offset = (DWORD)(off & 0xFFFFFFFF);
    ov.OffsetHigh = (DWORD)((uint64_t)off >> 32);
    DWORD got = 0;
    if (!WriteFile(h, src, (DWORD)n, &got, &ov)) return -1;
    return (int64_t)got;
#else
    return pwrite(fd, src, n, off);
#endif
}

static int bench_truncate0(int fd) {
#ifdef _WIN32
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    LARGE_INTEGER z = {0};
    if (!SetFilePointerEx(h, z, NULL, FILE_BEGIN)) return -1;
    return SetEndOfFile(h) ? 0 : -1;
#else
    return ftruncate(fd, 0);
#endif
}

/* O_DIRECT and FILE_FLAG_NO_BUFFERING both accept the open and then fail
 * every misaligned transfer, so eligibility is necessary and not sufficient:
 * tmpfs takes the flag and refuses the read, and so would a device wanting a
 * bigger block than we align to. The engine confirms with a transfer rather
 * than with the open; do the same, or a refusing filesystem turns into
 * "short read -1" and a table of zeroes with no cause given. */
static int dio_probe(int fd, int writing) {
    void *buf = waste_aligned_alloc(DIO_ALIGN, DIO_ALIGN);
    if (!buf) return 0;
    memset(buf, 0, DIO_ALIGN);
    const int64_t got = writing ? bench_pwrite(fd, buf, DIO_ALIGN, 0)
                                : waste_pread(fd, buf, DIO_ALIGN, 0);
    waste_aligned_free(buf);
    if (writing && got == (int64_t)DIO_ALIGN) bench_truncate0(fd);  /* undo it */
    return got == (int64_t)DIO_ALIGN;
}
#endif

/* The bypassed open, or -1 if this platform/filesystem will not give one.
 * Never falls back itself — that is open_path's job, in one place, so that
 * a platform nobody added a branch for cannot end up reporting a bypass it
 * does not have.
 *
 * That is precisely what happened on Windows (#36, gap 4): both assignments
 * clearing g_direct sat inside the __linux__ and __APPLE__ blocks, so
 * control reached the plain buffered open with the flag still 1 and
 * bypass_note() printed "cache bypassed" over page-cache numbers. The bug
 * was the shape of the code rather than a missing branch, so the shape is
 * what changed. */
static int open_bypass(int writing) {
#if defined(_WIN32)
    const DWORD access = writing ? GENERIC_WRITE : GENERIC_READ;
    const DWORD disp = writing ? CREATE_ALWAYS : OPEN_EXISTING;
    const HANDLE h = CreateFileA(g_path, access,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, disp,
                                 FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH,
                                 NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int fd = _open_osfhandle((intptr_t)h,
                                   (writing ? _O_WRONLY : _O_RDONLY) | _O_BINARY);
    if (fd < 0) { CloseHandle(h); return -1; }
    if (dio_probe(fd, writing)) return fd;
    close(fd);
    return -1;
#elif defined(__linux__) && defined(O_DIRECT)
    const int rw = writing ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
    const int fd = open(g_path, rw | O_DIRECT, 0644);
    if (fd < 0) return -1;
    if (dio_probe(fd, writing)) return fd;
    close(fd);
    return -1;
#elif defined(__APPLE__)
    /* F_NOCACHE has no alignment contract, so this is the whole bypass on
     * macOS and its failure is the difference between the number the row
     * claims and the number it prints. */
    const int rw = writing ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
    const int fd = open(g_path, rw, 0644);
    if (fd < 0) return -1;
    if (fcntl(fd, F_NOCACHE, 1) < 0) { close(fd); return -1; }
    fcntl(fd, F_RDAHEAD, 0);
    return fd;
#else
    (void)writing;
    return -1;                       /* no bypass known here; say so */
#endif
}

/* Open the working file with the page cache out of the way, and fall back
 * rather than fail when the filesystem will not have it — a bench that
 * quietly measures something else is worse than one that says it could not
 * (LEARNED §14). Falling back clears g_direct, and the rows say so.
 *
 * The write goes through the same door as the reads, and that is not
 * cosmetic: F_NOCACHE only stops *new* pages being cached, it does not evict
 * what is already resident, so a buffered write leaves the whole file in the
 * UBC and every read row below then measures RAM. Measured on an M5 Pro, 1 GB
 * file: 8.07 GB/s sequential read with the write bypassed, 26.04 GB/s with it
 * buffered. Linux would have survived the same mistake, since an O_DIRECT
 * read writes back and invalidates the range first — which is exactly why it
 * has to be a rule here and not a judgement call per platform. */
static int open_path(int writing) {
    const int fd = open_bypass(writing);
    if (fd >= 0) return fd;
    g_direct = 0;                    /* the only assignment, on every path */

#ifdef _WIN32
    const DWORD access = writing ? GENERIC_WRITE : GENERIC_READ;
    const DWORD disp = writing ? CREATE_ALWAYS : OPEN_EXISTING;
    const HANDLE h = CreateFileA(g_path, access,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, disp,
                                 FILE_FLAG_RANDOM_ACCESS, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const int pfd = _open_osfhandle((intptr_t)h,
                                    (writing ? _O_WRONLY : _O_RDONLY) | _O_BINARY);
    if (pfd < 0) { CloseHandle(h); return -1; }
    return pfd;
#else
    const int rw = writing ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;
    const int pfd = open(g_path, rw, 0644);
    if (pfd < 0) return pfd;
#ifdef __linux__
    /* No bypass to be had: at least stop the kernel reading ahead into pages
     * nothing will ask for, which is what the engine settles for too. */
    posix_fadvise(pfd, 0, 0, POSIX_FADV_RANDOM);
#endif
    return pfd;
#endif
}

static const char *bypass_note(void) {
    return g_direct ? "cache bypassed" : "PAGE CACHE, not the disk";
}

typedef struct { int id, nthreads; double bytes; } targ;

static void *rand_reader(void *p) {
    targ *a = (targ *)p;
    int fd = open_path(0);
    if (fd < 0) { perror("open"); return NULL; }
    void *buf = NULL;
    buf = waste_aligned_alloc(DIO_ALIGN, g_rec);
    if (!buf) { close(fd); return NULL; }
    size_t nrec = g_file / g_rec;
    unsigned seed = 12345u + a->id * 7919u;
    double got = 0;
    for (int i = 0; i < g_reps; i++) {
        seed = seed * 1103515245u + 12345u;
        /* Not off_t: under LLP64 — MSYS2 UCRT64, the documented Windows
         * build — off_t is a 32-bit long, while g_file and g_rec are size_t
         * and waste_pread takes an int64_t precisely so that an offset past
         * 2 GiB survives the trip to ReadFile's Offset/OffsetHigh pair.
         * Narrowing through off_t here threw that away at the last step: the
         * offsets that wrapped negative were refused by waste_pread's
         * `off < 0` guard, which returns -1 without touching errno and prints
         * as "short read -1: No error", and the ones that wrapped positive
         * did not fail at all — they quietly read the wrong place, so a run
         * kept its whole working set inside the first 2 GiB while reporting
         * the size it was asked for. The default file size is 8 GB, so this
         * was the default path, and a 2 GiB hot region flatters a consumer
         * SSD enough to make the number worth more than it is. */
        int64_t off = (int64_t)(seed % nrec) * (int64_t)g_rec;
        int64_t r = waste_pread(fd, buf, g_rec, off);
        if (r != (int64_t)g_rec) { fprintf(stderr, "short read %lld: %s\n", (long long)r, strerror(errno)); break; }
        got += r;
    }
    a->bytes = got;
    waste_aligned_free(buf); close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s PATH [file_gb] [rec_mb] [threads] [gb_per_token]\n"
                        "  gb_per_token: from `waste bench` on the container you are\n"
                        "  sizing for; omitted, the tok/s column is omitted with it\n", argv[0]);
        return 1;
    }
    g_path = argv[1];
    double file_gb = argc > 2 ? atof(argv[2]) : 8.0;
    double rec_mb  = argc > 3 ? atof(argv[3]) : 12.0;
    int maxthreads = argc > 4 ? atoi(argv[4]) : 8;
    /* No default, deliberately. This was K3's 12.5 hardcoded into the format
     * string, so every run printed a K3 answer whatever the container being
     * sized — off by ~8x on a 48B model, and silent about it. A column that
     * cannot be derived from what was measured does not get printed. */
    double gb_tok = argc > 5 ? atof(argv[5]) : 0.0;
    g_file = (size_t)(file_gb * (1u << 30));
    g_rec  = (size_t)(rec_mb * (1u << 20)) & ~4095UL;
    /* A record under one page rounds to zero and divides by it two screens
     * further down, as a crash rather than as a usage error. */
    if (!g_rec || g_file < g_rec) {
        fprintf(stderr, "record must be >= 4 KiB and file must hold at least one\n");
        return 1;
    }

    printf("file %.1f GB, record %.1f MB, path %s\n", file_gb, rec_mb, g_path);

    /* 1. sequential write */
    int fd = open_path(1);
    if (fd < 0) { perror("open write"); return 1; }
    void *buf = waste_aligned_alloc(DIO_ALIGN, g_rec);
    if (!buf) return 1;
    memset(buf, 0xA5, g_rec);
    double t0 = now(); size_t written = 0;
    while (written < g_file) {
        ssize_t w = write(fd, buf, g_rec);
        if (w <= 0) { perror("write"); break; }
        written += w;
    }
    bench_sync(fd); close(fd);
    double dt = now() - t0;
    printf("seq write   : %6.2f GB/s  (%s)\n", written / dt / (1u << 30), bypass_note());

    /* 2. sequential read, cache-bypassed */
    fd = open_path(0);
    if (fd < 0) { perror("open read"); return 1; }
    t0 = now(); size_t rd = 0; ssize_t r;
    while ((r = read(fd, buf, g_rec)) > 0) rd += r;
    if (r < 0) perror("read");   /* a failed read otherwise just ends the loop and shortens the row */
    dt = now() - t0; close(fd);
    printf("seq read    : %6.2f GB/s  (%s)\n", rd / dt / (1u << 30), bypass_note());

    /* 3+4. random record reads, 1..maxthreads */
    g_reps = 40;
    for (int nt = 1; nt <= maxthreads; nt *= 2) {
        pthread_t th[64]; targ ta[64];
        t0 = now();
        for (int i = 0; i < nt; i++) {
            ta[i].id = i; ta[i].nthreads = nt; ta[i].bytes = 0;
            pthread_create(&th[i], NULL, rand_reader, &ta[i]);
        }
        double tot = 0;
        for (int i = 0; i < nt; i++) { pthread_join(th[i], NULL); tot += ta[i].bytes; }
        dt = now() - t0;
        double gbs = tot / dt / (1u << 30);
        if (gb_tok > 0)
            printf("rand %2d thr : %6.2f GB/s  -> %.2f tok/s at %.4g GB/token cold  (%s)\n",
                   nt, gbs, gbs / gb_tok, gb_tok, bypass_note());
        else
            printf("rand %2d thr : %6.2f GB/s  (%s)\n", nt, gbs, bypass_note());
    }

    if (!g_direct) {
        fflush(stdout);   /* or the diagnostic overtakes the table it is about */
        fprintf(stderr,
                "\nthe page cache could not be bypassed on this filesystem, so the rows\n"
                "above are partly the kernel serving RAM back and are not the disk. The\n"
                "engine falls back the same way and reports it as direct_io=0; move the\n"
                "working file to the filesystem the container will live on.\n");
    }

    waste_aligned_free(buf);
    unlink(g_path);
    return 0;
}
