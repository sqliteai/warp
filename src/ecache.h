/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * ecache.h — bounded expert cache over the per-layer banks.
 *
 * This is the piece the whole K3 argument rests on: the model does not fit
 * in RAM, so experts are streamed from disk and a bounded cache decides how
 * often that costs a read. Gate 2 measured the hit rates in simulation;
 * this implements them for real.
 *
 * Policy is LFRU — frequency first, recency as tiebreak, victims chosen
 * from a small random sample rather than a full scan (the same
 * approximation the Gate 2 simulator used).
 * Gate 2 also showed why: at small cache fractions plain LRU collapses to
 * 5% where LFRU still gets 29%.
 *
 * Reads bypass the page cache (F_NOCACHE on macOS, O_DIRECT on Linux,
 * FILE_FLAG_NO_BUFFERING on Windows). That is deliberate:
 * with a 17 GB container on a 64 GB machine the kernel would cache
 * everything and the hit rate we measure would be a fiction. K3's ~900 GB
 * gets no such help, so the engine must not depend on it.
 */

#ifndef WASTE_ECACHE_H
#define WASTE_ECACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* `fetch(user, layer, expert, dst)` must fill rec_bytes and return 0.
 * With read-ahead on it is called from the reader threads as well as the
 * caller's, so an implementation must be safe to enter concurrently. */
typedef int (*waste_fetch_fn)(void *user, int layer, int expert, uint8_t *dst);

enum {
    EC_EMPTY = 0,        /* no record, and none on the way                  */
    EC_READY,            /* data is the record named by key                 */
    EC_INFLIGHT,         /* an I/O thread is filling data right now         */
    EC_FAILED            /* the read for key failed; get() reports and drops */
};

typedef struct {
    int32_t  key;        /* layer<<16 | expert, or -1 when empty            */
    uint8_t  state;      /* one of the EC_* above                           */
    uint8_t  fresh;      /* read was issued for this batch: not a hit yet   */
    uint32_t hits;       /* LFRU frequency term                             */
    uint64_t last;       /* LFRU recency term                               */
    uint64_t pin;        /* hint generation holding it; 0 = evictable       */
    uint8_t *data;
} waste_eslot;

/* The most experts one hint may name. top_k is bounded at 64 by cfg_sane,
 * so a decode hint always fits; a prefill chunk names more and is clipped,
 * which costs read-ahead on the tail and nothing else. */
#define WASTE_PF_MAX 64

struct waste_eio;        /* opaque: the reader threads and their queue      */

typedef struct {
    waste_eslot *slot;
    int32_t *hash;       /* open addressing: hash -> slot index, -1 empty   */
    int n_slots, hash_mask;
    size_t rec_bytes, budget_bytes;
    uint64_t clock, hits, misses, bytes_read, evictions;
    uint64_t prefetched; /* misses whose read was started before it blocked */
    uint64_t spec_issued;/* records the lookahead fetched on a guess         */
    unsigned rng;
    int policy;          /* 0 = LFRU, 1 = LRU                               */

    /* read-ahead; io == NULL means every fetch is synchronous, which is
     * what the engine did before 0.7.0 and still does when asked for zero
     * reader threads. */
    struct waste_eio *io;
    waste_fetch_fn fetch;
    void *fetch_user;
    int depth;                       /* reads kept in flight               */
    uint64_t pf_gen;                 /* current hint generation, never 0   */
    int pf_ids[WASTE_PF_MAX];
    int pf_layer, pf_n, pf_issued;

    /* Purgeable slots (WASTE_PURGEABLE=1, macOS). A slot is volatile while
     * idle, so under pressure the kernel drops it instead of compressing
     * and swapping it, and the engine re-reads what it dropped — which is
     * the pread it already knows how to do. LEARNED.md §16 is why: the
     * cliff there is not the cache missing, it is a cache hit turning into
     * a page fault. Off by default; it is a prototype. */
    int purgeable;
    int wired;                       /* WASTE_MLOCK: slots pinned in RAM   */
    int wire_failed;                 /* how many mlock calls were refused  */
    size_t slot_bytes;               /* what was handed to vm_allocate     */
    int last_used;                   /* held nonvolatile until the next get */
    /* The same claim last_used makes, for more than one record at a time.
     * A layer that hands each of its top-k experts to a different thread
     * holds all k at once, so "the one the caller is using" stops being a
     * single slot. Released together by waste_ecache_release. */
    int held[WASTE_PF_MAX];
    int n_held;
    uint64_t purged;                 /* slots the kernel reclaimed         */
} waste_ecache;

/* O_DIRECT requires the destination buffer to be aligned to the device's
 * logical block size, so record buffers come from here rather than malloc.
 * Expert records are whole 4 KiB pages by construction, which covers both
 * the 512- and 4096-byte cases; on macOS the alignment is merely harmless. */
/* 16 KiB, the Apple Silicon page: O_DIRECT wants 512 or 4096, and Metal's
 * newBufferWithBytesNoCopy wants a whole page — one alignment serves both,
 * which is what lets the GPU read trunk weights the CPU already has. */
#define WASTE_DIO_ALIGN 16384
void *waste_dio_alloc(size_t n);
void  waste_dio_free(void *p);

/* Wiring, shared by the cache and the trunk so one switch cannot mean two
 * things. WASTE_MLOCK: 0/unset off, `cache` (or 1, which is what §30 of
 * LEARNED.md measured), `trunk`, `all`. */
#define WASTE_WIRE_CACHE 1
#define WASTE_WIRE_TRUNK 2
int waste_mlock_mode(void);
int waste_wire(void *p, size_t n);          /* 1 on success */

/* budget_bytes 0 disables caching (every access reads). Returns 0 on ok. */
int  waste_ecache_init(waste_ecache *c, size_t budget_bytes, size_t rec_bytes,
                       int policy);
void waste_ecache_free(waste_ecache *c);
void waste_ecache_clear(waste_ecache *c);

/* Returns a pointer to the expert's record bytes, reading it through
 * `fetch` on a miss, or waiting for the read a hint already started.
 * NULL on failure. */
const uint8_t *waste_ecache_get(waste_ecache *c, int layer, int expert,
                                waste_fetch_fn fetch, void *user);

/* waste_ecache_get, except the record stays claimed after the next one is
 * asked for. A caller that hands k records to k threads needs every one of
 * them to outlive the loop that collected them; get() alone releases each
 * as the next arrives, and on macOS a released slot is volatile — the
 * kernel may drop it and the reader would see zeros rather than weights.
 *
 * Holds at most WASTE_PF_MAX; returns NULL if the set is full or the read
 * failed. Every hold must be matched by one waste_ecache_release, which
 * releases the whole set. Not for concurrent use: collect on one thread,
 * then read the pointers from many. */
const uint8_t *waste_ecache_hold(waste_ecache *c, int layer, int expert,
                                 waste_fetch_fn fetch, void *user);
void waste_ecache_release(waste_ecache *c);

/* ---- read-ahead ---------------------------------------------------------
 *
 * The disk is nearly saturated at queue depth 1 (10.73 of 12.89 GB/s on the
 * internal SSD at K3's 11.83 MB record), so this is not about bandwidth: it
 * is that a blocking pread stops the arithmetic. A MoE layer knows all of
 * its top-K expert ids before it reads the first one, so the reads can be
 * issued ahead and consumed as the matmuls finish. docs/EFFICIENCY.md has
 * the measurement.
 *
 * nthreads 0 leaves every fetch synchronous. Returns 0 on success; failing
 * to start the threads is not fatal, the cache simply stays synchronous. */
int  waste_ecache_io_start(waste_ecache *c, waste_fetch_fn fetch, void *user,
                           int nthreads, int depth);
void waste_ecache_io_stop(waste_ecache *c);

/* Name the records this layer is about to ask for, in the order it will ask.
 * Reads for the first `depth` of them that are not resident start now, and
 * each waste_ecache_get releases one more into the pipe. Calling it with a
 * synchronous cache is a no-op, so call sites need no conditional. */
/* Residency of a whole layer's experts, without touching any of them.
 *
 * `out[e]` becomes 1 when expert e of `layer` is in the cache and READY —
 * usable with no disk read. Nothing is claimed, no recency moves, and
 * nothing counts as a hit or a miss: this answers a question, it does not
 * make a request. INFLIGHT deliberately reads as 0, so "resident" means
 * arrived rather than asked for.
 *
 * One call per layer rather than one per expert because a single mutex
 * covers the whole cache, and a per-expert probe would take it E times per
 * layer per token — 83k times a token on K3.
 *
 * For a router that wants to prefer experts it already has (arXiv:2412.00099).
 */
void waste_ecache_resident_mask(waste_ecache *c, int layer, int n, uint8_t *out);

void waste_ecache_hint(waste_ecache *c, int layer, const int *ids, int n);

/* Speculative fill for a layer that has not routed yet. Unlike a hint these
 * are not pinned and are not counted as misses: they are a guess, and the
 * demand stream's hit rate has to keep meaning what it meant before. */
void waste_ecache_prefetch(waste_ecache *c, int layer, const int *ids, int n);

/* Persist / restore which experts this workload actually uses. Gate 5
 * measured a 0% hit rate until the cache exceeds one token's working set;
 * a warm start does not change that floor, but it does remove the cold
 * ramp at the beginning of every run. */
int waste_ecache_save_usage(const waste_ecache *c, const char *path,
                            uint64_t tokens);
int waste_ecache_warm(waste_ecache *c, const char *path,
                      waste_fetch_fn fetch, void *user);

static inline double waste_ecache_hit_rate(const waste_ecache *c)
{
    const uint64_t t = c->hits + c->misses;
    return t ? (double)c->hits / (double)t : 0.0;
}

#ifdef __cplusplus
}
#endif
#endif /* WASTE_ECACHE_H */
