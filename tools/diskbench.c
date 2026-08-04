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
 * Build: cc -O2 -o diskbench tools/diskbench.c
 * Usage: ./diskbench /Volumes/WasteDisk/k3/.bench [file_gb] [rec_mb] [threads]
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

static double now(void) {
    struct timeval t; gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

/* Linux has no F_NOCACHE; the page cache is bypassed with O_DIRECT at open
 * time instead, which is what the engine itself does (see model.c). Without it
 * this benchmark reports page-cache throughput on Linux — e.g. 42 GB/s
 * sequential on a drive whose link tops out at 3.9 GB/s. Buffers here are
 * already posix_memalign'd to 4096 and the record size is a 4096 multiple, so
 * O_DIRECT's alignment requirements are met. */
#if defined(__linux__) && defined(O_DIRECT)
#define DIO_FLAG O_DIRECT
#else
#define DIO_FLAG 0
#endif

static void nocache(int fd) {
#ifdef __APPLE__
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
#endif
    (void)fd;
}

static const char *g_path;
static size_t g_rec, g_file;
static int g_reps;

typedef struct { int id, nthreads; double bytes; } targ;

static void *rand_reader(void *p) {
    targ *a = (targ *)p;
    int fd = open(g_path, O_RDONLY | DIO_FLAG);
    if (fd < 0) { perror("open"); return NULL; }
    nocache(fd);
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, g_rec)) { close(fd); return NULL; }
    size_t nrec = g_file / g_rec;
    unsigned seed = 12345u + a->id * 7919u;
    double got = 0;
    for (int i = 0; i < g_reps; i++) {
        seed = seed * 1103515245u + 12345u;
        off_t off = (off_t)(seed % nrec) * g_rec;
        ssize_t r = pread(fd, buf, g_rec, off);
        if (r != (ssize_t)g_rec) { fprintf(stderr, "short read %zd\n", r); break; }
        got += r;
    }
    a->bytes = got;
    free(buf); close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s PATH [file_gb] [rec_mb] [threads]\n", argv[0]); return 1; }
    g_path = argv[1];
    double file_gb = argc > 2 ? atof(argv[2]) : 8.0;
    double rec_mb  = argc > 3 ? atof(argv[3]) : 12.0;
    int maxthreads = argc > 4 ? atoi(argv[4]) : 8;
    g_file = (size_t)(file_gb * (1u << 30));
    g_rec  = (size_t)(rec_mb * (1u << 20)) & ~4095UL;

    printf("file %.1f GB, record %.1f MB, path %s\n", file_gb, rec_mb, g_path);

    /* 1. sequential write */
    int fd = open(g_path, O_WRONLY | O_CREAT | O_TRUNC | DIO_FLAG, 0644);
    if (fd < 0) { perror("open write"); return 1; }
    nocache(fd);
    void *buf;
    if (posix_memalign(&buf, 4096, g_rec)) return 1;
    memset(buf, 0xA5, g_rec);
    double t0 = now(); size_t written = 0;
    while (written < g_file) {
        ssize_t w = write(fd, buf, g_rec);
        if (w <= 0) { perror("write"); break; }
        written += w;
    }
    fsync(fd); close(fd);
    double dt = now() - t0;
    printf("seq write   : %6.2f GB/s\n", written / dt / (1u << 30));

    /* 2. sequential read, cache-bypassed */
    fd = open(g_path, O_RDONLY | DIO_FLAG); nocache(fd);
    t0 = now(); size_t rd = 0; ssize_t r;
    while ((r = read(fd, buf, g_rec)) > 0) rd += r;
    dt = now() - t0; close(fd);
    printf("seq read    : %6.2f GB/s  (cache bypassed)\n", rd / dt / (1u << 30));

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
        printf("rand %2d thr : %6.2f GB/s  -> %.2f tok/s at 12.5 GB/token cold\n",
               nt, gbs, gbs / 12.5);
    }

    free(buf);
    unlink(g_path);
    return 0;
}
