/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* ecache.c — see ecache.h. */

#include "ecache.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "waste_format.h"

#define EC_SAMPLE 16      /* victims sampled per eviction (Redis-style)     */
#define EC_QCAP   256     /* outstanding read requests                      */
#define EC_MAXIO  8       /* reader threads                                 */

/* Rounding the length up as well as the address is not decoration: both
 * O_DIRECT and FILE_FLAG_NO_BUFFERING refuse a transfer whose length is
 * not a whole number of sectors, and the tail of a record buffer is the
 * one the caller does not think about. */
void *waste_dio_alloc(size_t n)
{
    const size_t pad = (n + WASTE_DIO_ALIGN - 1) / WASTE_DIO_ALIGN * WASTE_DIO_ALIGN;
    return waste_aligned_alloc(WASTE_DIO_ALIGN, pad);
}

void waste_dio_free(void *p) { waste_aligned_free(p); }

/* ---- purgeable slots ----------------------------------------------------
 *
 * Measured before it was built: a volatile<->nonvolatile round trip costs
 * 0.33 us, i.e. 0.5 ms over a K3 token's 1472 experts, and vm_allocate
 * returns 16 KiB-aligned pages so O_DIRECT is unaffected.
 *
 * The contract that matters: a purged object reads back as zeros, which
 * would be a silently wrong expert. So a slot is made nonvolatile *before*
 * anything reads or writes it, and the state that comes back says whether
 * the kernel took it in the meantime. Never read a volatile slot.
 */
#ifdef __APPLE__
#include <mach/mach.h>

static void *ec_purge_alloc(size_t n)
{
    vm_address_t a = 0;
    if (vm_allocate(mach_task_self(), &a, n,
                    VM_FLAGS_ANYWHERE | VM_FLAGS_PURGABLE) != KERN_SUCCESS)
        return NULL;
    return (void *)a;
}

static void ec_purge_free(void *p, size_t n)
{
    if (p) vm_deallocate(mach_task_self(), (vm_address_t)p, n);
}

/* Returns 1 if the kernel had reclaimed it, i.e. the record is gone. */
static int ec_nonvolatile(waste_ecache *c, int si)
{
    if (!c->purgeable) return 0;
    int st = VM_PURGABLE_NONVOLATILE;
    if (vm_purgable_control(mach_task_self(), (vm_address_t)c->slot[si].data,
                            VM_PURGABLE_SET_STATE, &st) != KERN_SUCCESS)
        return 0;
    return (st & VM_PURGABLE_EMPTY) != 0;
}

static void ec_volatile(waste_ecache *c, int si)
{
    if (!c->purgeable) return;
    int st = VM_PURGABLE_VOLATILE;
    vm_purgable_control(mach_task_self(), (vm_address_t)c->slot[si].data,
                        VM_PURGABLE_SET_STATE, &st);
}
#else
/* Linux's nearest equivalent is MADV_FREE plus a sentinel page to notice
 * the drop; it is not written, so the flag simply does nothing there. */
static void *ec_purge_alloc(size_t n) { (void)n; return NULL; }
static void  ec_purge_free(void *p, size_t n) { (void)p; (void)n; }
static int   ec_nonvolatile(waste_ecache *c, int si) { (void)c; (void)si; return 0; }
static void  ec_volatile(waste_ecache *c, int si) { (void)c; (void)si; }
#endif

/* The slot the last get() handed out stays nonvolatile while its caller
 * uses it, and is released here on the next one — moe_layer and moe_chunk
 * both finish with a record before asking for the next. */
/* Wiring the slots down. The opposite bargain to purgeable memory: that one
 * tells the kernel it may take a slot cheaply, this one tells it that it may
 * not take one at all. Whether that is an improvement depends entirely on
 * whether the memory exists, which is what §16's cliff is really about.
 *
 * macOS caps wired memory at vm.user_wire_limit, 7/8 of RAM on this machine
 * — the same fraction the budget resolver uses — so a cache that fits the
 * budget fits the limit. A failure is reported and not fatal: an engine that
 * refuses to open because it could not wire its cache would be worse than
 * one that runs with a pageable one. */
#ifndef _WIN32
#include <sys/mman.h>
int waste_wire(void *p, size_t n) { return mlock(p, n) == 0; }
#else
/* VirtualLock is bounded by the process *maximum working set*, not by
 * available memory, and the default maximum is far below any cache worth
 * wiring. So this wired nothing at all on Windows: 5928 of 5928 slots
 * refused with ERROR_WORKING_SET_QUOTA on a machine with 54 GB free and
 * nothing paging (#36, gap 5). mlock has no equivalent requirement, which
 * is why the port worked everywhere else and this went unnoticed.
 *
 * Raise the ceiling on demand rather than up front: waste_wire is handed one
 * slot at a time and never learns the total, so the first refusal is the
 * only place the size actually needed is known. SetProcessWorkingSetSize
 * takes both bounds, so the current minimum is read back and preserved —
 * passing a minimum above the maximum fails the call outright.
 *
 * If the raise is refused — it can need SE_INC_WORKING_SET_NAME, which a
 * service account may not hold — latch and stop asking. The caller's
 * accounting is unchanged, since a latched call still reports failure; what
 * goes away is thousands of syscalls that cannot succeed. Reporting the
 * failure honestly is the existing behaviour and stays: an engine that
 * refused to open because it could not wire its cache would be worse than
 * one running with a pageable cache.
 *
 * Unverified on a real Windows host: no machine here has one. It builds
 * under both toolchains and the failure path is the previous behaviour. */
static int wire_quota_exhausted;      /* raise refused; stop trying */

int waste_wire(void *p, size_t n)
{
    if (VirtualLock(p, n)) return 1;
    if (GetLastError() != ERROR_WORKING_SET_QUOTA || wire_quota_exhausted)
        return 0;

    SIZE_T lo = 0, hi = 0;
    if (!GetProcessWorkingSetSize(GetCurrentProcess(), &lo, &hi)) {
        wire_quota_exhausted = 1;
        return 0;
    }
    /* Room for this slot and the ones behind it, so the raise is not paid
     * once per slot. Overshooting costs nothing: the maximum is a ceiling
     * the process may reach, not a reservation. */
    const SIZE_T grow = n + (n < (64u << 20) ? (64u << 20) : n);
    if (hi > (SIZE_T)-1 - grow || !SetProcessWorkingSetSize(GetCurrentProcess(),
                                                            lo, hi + grow)) {
        wire_quota_exhausted = 1;
        return 0;
    }
    return VirtualLock(p, n) != 0;
}
#endif

int waste_mlock_mode(void)
{
    const char *e = getenv("WASTE_MLOCK");
    if (!e || !*e || *e == '0') return 0;
    if (!strcmp(e, "trunk")) return WASTE_WIRE_TRUNK;
    if (!strcmp(e, "cache")) return WASTE_WIRE_CACHE;
    if (!strcmp(e, "all") || !strcmp(e, "both"))
        return WASTE_WIRE_CACHE | WASTE_WIRE_TRUNK;
    /* Bare 1 means "wire what you can". It meant cache-only when §30 was
     * written, and §31 measured the trunk: wiring the cache alone is the
     * one setting that is worse than doing nothing, because it leaves the
     * hot part pageable. `cache` still names the old behaviour, for
     * reproducing that section. */
    return WASTE_WIRE_CACHE | WASTE_WIRE_TRUNK;
}

static void ec_release_last(waste_ecache *c)
{
    if (c->last_used < 0) return;
    /* Only if it is still a finished record: a slot that has since been
     * reclaimed for another expert may have a reader writing into it, and
     * a volatile object under an in-flight write is how a purged page
     * becomes a silently zeroed weight. */
    if (c->slot[c->last_used].state == EC_READY) ec_volatile(c, c->last_used);
    c->last_used = -1;
}

static int32_t ec_key(int layer, int expert) { return (layer << 16) | expert; }

static uint32_t ec_hash(int32_t k)
{
    uint32_t x = (uint32_t)k;
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/* ---- reader threads -----------------------------------------------------
 *
 * One mutex covers the whole cache: slot metadata, the hash, the counters,
 * the hint cursor and this queue. It is held for pointer arithmetic only —
 * never across a fetch, which is the entire point — so contention between
 * the compute thread and the readers is not measurable next to an 11.83 MB
 * pread.
 *
 * A worker owns slot->data exclusively while the slot is EC_INFLIGHT: the
 * victim sampler will not choose an in-flight slot and get() will not read
 * one until it has been marked ready, so the payload itself needs no lock.
 */

typedef struct { int slot, layer, expert; } eio_job;

struct waste_eio {
    pthread_t th[EC_MAXIO];
    int nthreads;
    pthread_mutex_t mu;
    pthread_cond_t  work;   /* a job was queued, or stop was set            */
    pthread_cond_t  done;   /* some slot left EC_INFLIGHT                   */
    eio_job q[EC_QCAP];
    int qhead, qn, stop;
    waste_ecache *c;
};

static void ec_lock(waste_ecache *c)   { if (c->io) pthread_mutex_lock(&c->io->mu); }
static void ec_unlock(waste_ecache *c) { if (c->io) pthread_mutex_unlock(&c->io->mu); }

static void *eio_worker(void *p)
{
    struct waste_eio *io = (struct waste_eio *)p;
    waste_ecache *c = io->c;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (!io->stop && io->qn == 0) pthread_cond_wait(&io->work, &io->mu);
        if (io->stop && io->qn == 0) { pthread_mutex_unlock(&io->mu); return NULL; }
        const eio_job j = io->q[io->qhead];
        io->qhead = (io->qhead + 1) % EC_QCAP;
        io->qn--;
        pthread_mutex_unlock(&io->mu);

        const int rc = c->fetch(c->fetch_user, j.layer, j.expert,
                                c->slot[j.slot].data);

        pthread_mutex_lock(&io->mu);
        c->slot[j.slot].state = rc == 0 ? (uint8_t)EC_READY : (uint8_t)EC_FAILED;
        pthread_cond_broadcast(&io->done);
        pthread_mutex_unlock(&io->mu);
    }
}

/* caller holds the lock */
static int eio_push(struct waste_eio *io, int slot, int layer, int expert)
{
    if (io->qn == EC_QCAP) return -1;
    io->q[(io->qhead + io->qn) % EC_QCAP] = (eio_job){ slot, layer, expert };
    io->qn++;
    pthread_cond_signal(&io->work);
    return 0;
}

int waste_ecache_io_start(waste_ecache *c, waste_fetch_fn fetch, void *user,
                          int nthreads, int depth)
{
    if (c->io || nthreads <= 0 || c->n_slots <= 0 || !fetch) return 0;
    if (nthreads > EC_MAXIO) nthreads = EC_MAXIO;
    /* Every pinned slot is one the victim sampler may not take, so the
     * pipeline can never be allowed to reach the size of the cache. */
    if (depth < 1) depth = 1;
    if (depth > c->n_slots / 4) depth = c->n_slots / 4;
    if (depth < 1) return 0;

    struct waste_eio *io = (struct waste_eio *)calloc(1, sizeof *io);
    if (!io) return -1;
    io->c = c;
    if (pthread_mutex_init(&io->mu, NULL)) { free(io); return -1; }
    pthread_cond_init(&io->work, NULL);
    pthread_cond_init(&io->done, NULL);

    c->fetch = fetch;
    c->fetch_user = user;
    c->depth = depth;
    c->io = io;                       /* readers dereference c->io->mu     */

    for (int i = 0; i < nthreads; i++) {
        if (pthread_create(&io->th[i], NULL, eio_worker, io)) break;
        io->nthreads++;
    }
    if (io->nthreads == 0) {          /* nothing started: stay synchronous */
        c->io = NULL;
        pthread_cond_destroy(&io->work);
        pthread_cond_destroy(&io->done);
        pthread_mutex_destroy(&io->mu);
        free(io);
        return -1;
    }
    return 0;
}

void waste_ecache_io_stop(waste_ecache *c)
{
    struct waste_eio *io = c->io;
    if (!io) return;
    pthread_mutex_lock(&io->mu);
    io->stop = 1;
    pthread_cond_broadcast(&io->work);
    pthread_mutex_unlock(&io->mu);
    for (int i = 0; i < io->nthreads; i++) pthread_join(io->th[i], NULL);
    /* Drop the pointer before the mutex dies: ec_lock tests c->io. */
    c->io = NULL;
    pthread_cond_destroy(&io->work);
    pthread_cond_destroy(&io->done);
    pthread_mutex_destroy(&io->mu);
    free(io);
}

/* ---- cache -------------------------------------------------------------- */

int waste_ecache_init(waste_ecache *c, size_t budget_bytes, size_t rec_bytes,
                      int policy)
{
    memset(c, 0, sizeof *c);
    c->rec_bytes = rec_bytes;
    c->budget_bytes = budget_bytes;
    c->policy = policy;
    c->rng = 0x9e3779b9u;
    /* Generations start at 1 so that a slot's zeroed pin means "never
     * hinted" rather than "pinned by the current batch". */
    c->pf_gen = 1;
    c->last_used = -1;
    c->n_held = 0;
    {   const char *e = getenv("WASTE_PURGEABLE");
        c->purgeable = e && *e != '0';
        c->wired = (waste_mlock_mode() & WASTE_WIRE_CACHE) != 0;
        /* One says the kernel may take a slot for free, the other that it
         * may never take one. Asking for both is a contradiction, not a
         * configuration. */
        if (c->wired) c->purgeable = 0; }
    if (!rec_bytes || budget_bytes < rec_bytes) return 0;   /* no cache */

    c->n_slots = (int)(budget_bytes / rec_bytes);
    int hs = 1;
    while (hs < c->n_slots * 2) hs <<= 1;
    c->hash_mask = hs - 1;

    c->slot = (waste_eslot *)calloc((size_t)c->n_slots, sizeof *c->slot);
    c->hash = (int32_t *)malloc((size_t)hs * sizeof *c->hash);
    if (!c->slot || !c->hash) { waste_ecache_free(c); return -1; }
    memset(c->hash, 0xff, (size_t)hs * sizeof *c->hash);   /* all -1 */

    c->slot_bytes = (rec_bytes + WASTE_DIO_ALIGN - 1) / WASTE_DIO_ALIGN *
                    WASTE_DIO_ALIGN;
    for (int i = 0; i < c->n_slots; i++) {
        c->slot[i].key = -1;
        c->slot[i].state = EC_EMPTY;
        c->slot[i].data = c->purgeable
                              ? (uint8_t *)ec_purge_alloc(c->slot_bytes)
                              : (uint8_t *)waste_dio_alloc(rec_bytes);
        /* A purgeable allocation that fails is not a reason to refuse the
         * model: fall back to ordinary memory for the whole cache, since a
         * mix would have to remember which slot came from where. */
        if (!c->slot[i].data && c->purgeable) {
            for (int j = 0; j < i; j++) ec_purge_free(c->slot[j].data, c->slot_bytes);
            c->purgeable = 0;
            i = -1;
            continue;
        }
        if (!c->slot[i].data) { waste_ecache_free(c); return -1; }
        if (c->wired && !waste_wire(c->slot[i].data, rec_bytes)) c->wire_failed++;
    }
    if (c->wire_failed)
        fprintf(stderr, "waste: could not wire %d of %d cache slots; "
                        "those stay pageable\n", c->wire_failed, c->n_slots);
    return 0;
}

void waste_ecache_free(waste_ecache *c)
{
    waste_ecache_io_stop(c);
    if (c->slot) {
        for (int i = 0; i < c->n_slots; i++) {
            if (c->purgeable) ec_purge_free(c->slot[i].data, c->slot_bytes);
            else waste_dio_free(c->slot[i].data);
        }
        free(c->slot);
    }
    free(c->hash);
    c->slot = NULL; c->hash = NULL; c->n_slots = 0;
}

static int ec_lookup(waste_ecache *c, int32_t key)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    for (int probe = 0; probe <= c->hash_mask; probe++) {
        const int32_t si = c->hash[h];
        if (si < 0) return -1;
        if (c->slot[si].key == key) return si;
        h = (h + 1) & (uint32_t)c->hash_mask;
    }
    return -1;
}

static void ec_insert(waste_ecache *c, int32_t key, int slot)
{
    uint32_t h = ec_hash(key) & (uint32_t)c->hash_mask;
    while (c->hash[h] >= 0) h = (h + 1) & (uint32_t)c->hash_mask;
    c->hash[h] = slot;
}

/* Rebuilding is simpler and safe: open addressing with deletions needs
 * tombstones, and evictions are rare relative to lookups. */
static void ec_rehash(waste_ecache *c)
{
    memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].key >= 0) ec_insert(c, c->slot[i].key, i);
}

/* A slot is untouchable while a reader owns its buffer, and while the hint
 * that reserved it is still being consumed — otherwise a deep pipeline
 * evicts the record it just read, one layer before it is used. */
static int ec_is_held(const waste_ecache *c, int i)
{
    for (int k = 0; k < c->n_held; k++)
        if (c->held[k] == i) return 1;
    return 0;
}

static int ec_pinned(const waste_ecache *c, int i)
{
    /* last_used is the record the caller is holding a pointer to right now.
     * get() releases one more read into the pipe before it returns, so
     * without this the reader threads could be handed the very slot whose
     * bytes the caller is about to multiply. The hint path pins it anyway;
     * the synchronous fallback inside get() does not, and that is the hole
     * this closes. */
    return c->slot[i].state == EC_INFLIGHT || c->slot[i].pin == c->pf_gen ||
           i == c->last_used || ec_is_held(c, i);
}

static int ec_victim(waste_ecache *c)
{
    /* free slot first */
    for (int i = 0; i < c->n_slots; i++)
        if (c->slot[i].state == EC_EMPTY) return i;

    int best = -1;
    uint32_t best_h = 0;
    uint64_t best_l = 0;
    for (int s = 0; s < EC_SAMPLE; s++) {
        c->rng = c->rng * 1664525u + 1013904223u;
        const int i = (int)(c->rng % (uint32_t)c->n_slots);
        if (ec_pinned(c, i)) continue;
        const waste_eslot *sl = &c->slot[i];
        int better;
        if (c->policy == 1)                       /* LRU */
            better = (best < 0) || sl->last < best_l;
        else                                      /* LFRU */
            better = (best < 0) || sl->hits < best_h ||
                     (sl->hits == best_h && sl->last < best_l);
        if (better) { best = i; best_h = sl->hits; best_l = sl->last; }
    }
    /* A sample that lands entirely on pinned slots is not an error, it is
     * the small-cache case: fall back to a scan rather than to no cache. */
    if (best < 0)
        for (int i = 0; i < c->n_slots; i++)
            if (!ec_pinned(c, i)) { best = i; break; }
    return best;
}

/* Claim `vi` for `key`, counting the read that is about to happen. Caller
 * holds the lock.
 *
 * `fresh` marks a read the current hint issued, and it is also what decides
 * the pin: a synchronous claim must NOT take one. Pinning it looked
 * harmless and was not — with read-ahead off the generation never advances,
 * so every slot a synchronous read claimed stayed pinned for the life of
 * the process, the victim sampler ran out of candidates, and the forward
 * pass quietly continued with the experts it had. EC_INFLIGHT alone already
 * protects a slot whose buffer is being written. */
/* `fresh` marks a record this layer's hint pulled in, and it is also the pin;
 * `spec` marks one the lookahead guessed at.
 *
 * The accounting differs on purpose. hits/misses describe the *demand*
 * stream — of the experts a token actually asked for, how many were already
 * there — and every number in LEARNED.md means that. A speculative read is
 * not a demand access, so it must not become a miss, or a prefetcher that
 * guessed wrong would look like a cache that performed badly. It is real
 * I/O, so its bytes are counted; if the guess lands, the token that asks for
 * it finds it resident and scores an ordinary hit. */
static void ec_claim_spec(waste_ecache *c, int vi, int32_t key)
{
    ec_nonvolatile(c, vi);
    const int had = c->slot[vi].key >= 0;
    c->slot[vi].key = key;
    c->slot[vi].state = EC_INFLIGHT;
    c->slot[vi].fresh = 0;              /* a demand hit on it is a real hit */
    c->slot[vi].pin = 0;                /* belongs to no hint generation    */
    c->slot[vi].hits = 1;
    c->slot[vi].last = c->clock;
    if (had) { c->evictions++; ec_rehash(c); }
    else ec_insert(c, key, vi);
    c->spec_issued++;
    c->bytes_read += c->rec_bytes;
}

static void ec_claim(waste_ecache *c, int vi, int32_t key, int fresh)
{
    /* Whoever fills this slot is about to write into it, so it has to stop
     * being volatile first; whether the kernel had already taken the old
     * record is of no interest, it is being overwritten either way. */
    ec_nonvolatile(c, vi);
    const int had = c->slot[vi].key >= 0;
    c->slot[vi].key = key;
    c->slot[vi].state = EC_INFLIGHT;
    c->slot[vi].fresh = (uint8_t)fresh;
    c->slot[vi].pin = fresh ? c->pf_gen : 0;
    c->slot[vi].hits = 1;
    c->slot[vi].last = c->clock;
    if (had) { c->evictions++; ec_rehash(c); }
    else ec_insert(c, key, vi);
    c->misses++;
    c->bytes_read += c->rec_bytes;
}

static void ec_drop(waste_ecache *c, int vi)
{
    c->slot[vi].key = -1;
    c->slot[vi].state = EC_EMPTY;
    c->slot[vi].fresh = 0;
    c->slot[vi].pin = 0;
    ec_rehash(c);
}

/* Release one more read into the pipe. Caller holds the lock. */
static void ec_issue_next(waste_ecache *c)
{
    if (!c->io) return;               /* pf_n is 0 without it, but say so  */
    while (c->pf_issued < c->pf_n) {
        const int eid = c->pf_ids[c->pf_issued++];
        const int32_t key = ec_key(c->pf_layer, eid);
        const int si = ec_lookup(c, key);
        if (si >= 0) {
            /* Already here or already coming. It costs no I/O, so it does
             * not consume pipeline depth either — keep looking. */
            c->slot[si].pin = c->pf_gen;
            continue;
        }
        const int vi = ec_victim(c);
        if (vi < 0) { c->pf_issued--; return; }   /* retry on the next get */
        ec_claim(c, vi, key, 1);
        c->prefetched++;
        if (eio_push(c->io, vi, c->pf_layer, eid) != 0) {
            /* queue full: undo, and let get() fetch it synchronously */
            c->prefetched--;
            c->misses--;
            c->bytes_read -= c->rec_bytes;
            ec_drop(c, vi);
            c->pf_issued--;
        }
        return;
    }
}

void waste_ecache_prefetch(waste_ecache *c, int layer, const int *ids, int n)
{
    if (!c->io || c->n_slots <= 0 || n <= 0) return;
    ec_lock(c);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0) continue;
        const int32_t key = ec_key(layer, ids[i]);
        if (ec_lookup(c, key) >= 0) continue;   /* already here, or coming */
        const int vi = ec_victim(c);
        if (vi < 0) break;                      /* everything pinned; skip */
        ec_claim_spec(c, vi, key);
        if (eio_push(c->io, vi, layer, ids[i]) != 0) {
            c->spec_issued--;
            c->bytes_read -= c->rec_bytes;
            ec_drop(c, vi);
            break;                              /* queue full: stop asking */
        }
    }
    ec_unlock(c);
}

void waste_ecache_hint(waste_ecache *c, int layer, const int *ids, int n)
{
    if (!c->io || c->n_slots <= 0 || n <= 0) return;

    ec_lock(c);
    /* Bumping the generation is what unpins the previous layer's records:
     * they are finished with by the time the next hint is placed, and no
     * explicit release can be forgotten on an error path. */
    c->pf_gen++;
    c->pf_layer = layer;
    c->pf_n = n > WASTE_PF_MAX ? WASTE_PF_MAX : n;
    /* Never pin more than a quarter of the cache. */
    if (c->pf_n > c->n_slots / 4) c->pf_n = c->n_slots / 4;
    for (int i = 0; i < c->pf_n; i++) c->pf_ids[i] = ids[i];
    c->pf_issued = 0;
    for (int d = 0; d < c->depth; d++) ec_issue_next(c);
    ec_unlock(c);
}

void waste_ecache_resident_mask(waste_ecache *c, int layer, int n, uint8_t *out)
{
    if (!out || n <= 0) return;
    memset(out, 0, (size_t)n);
    if (!c || c->n_slots <= 0) return;

    ec_lock(c);
    for (int e = 0; e < n; e++) {
        const int si = ec_lookup(c, ec_key(layer, e));
        /* READY only. An INFLIGHT slot has been asked for and has not
         * arrived, so a router told it was resident would pick it and then
         * wait on the same read it was trying to avoid. */
        if (si >= 0 && c->slot[si].state == EC_READY) out[e] = 1u;
    }
    ec_unlock(c);
}

const uint8_t *waste_ecache_get(waste_ecache *c, int layer, int expert,
                                waste_fetch_fn fetch, void *user)
{
    const int32_t key = ec_key(layer, expert);

    ec_lock(c);
    c->clock++;
    ec_release_last(c);

    if (c->n_slots > 0) {
        int si = ec_lookup(c, key);
        if (si >= 0) {
            while (c->io && c->slot[si].state == EC_INFLIGHT)
                pthread_cond_wait(&c->io->done, &c->io->mu);
            if (c->slot[si].state != EC_READY) {   /* failed, or lost its reader */
                ec_drop(c, si);
                ec_issue_next(c);
                ec_unlock(c);
                return NULL;
            }
            /* Claim it back from the kernel before reading a byte of it. A
             * purged object reads as zeros, so this test is the difference
             * between a miss and a silently wrong expert. */
            if (ec_nonvolatile(c, si)) {
                c->purged++;
                ec_drop(c, si);
                si = -1;                       /* fall through and re-read */
            }
            if (si >= 0) {
                /* A record this hint brought in was already counted as a
                 * miss when its read was issued; counting it again here
                 * would turn every prefetch into a fictitious hit. */
                if (c->slot[si].fresh) c->slot[si].fresh = 0;
                else { c->hits++; c->slot[si].hits++; }
                c->slot[si].last = c->clock;
                uint8_t *d = c->slot[si].data;
                c->last_used = si;
                ec_issue_next(c);
                ec_unlock(c);
                return d;
            }
        }
    }

    if (c->n_slots == 0) {                 /* caller falls back to its own buf */
        c->misses++;
        c->bytes_read += c->rec_bytes;
        ec_unlock(c);
        return NULL;
    }

    /* Not resident and not hinted: read it here, but still through a slot,
     * so a reader thread cannot pick the same one meanwhile. */
    const int vi = ec_victim(c);
    if (vi < 0) { ec_unlock(c); return NULL; }
    ec_claim(c, vi, key, 0);
    uint8_t *dst = c->slot[vi].data;
    ec_unlock(c);

    const int rc = fetch(user, layer, expert, dst);

    ec_lock(c);
    if (rc != 0) {
        ec_drop(c, vi);
        ec_unlock(c);
        return NULL;
    }
    c->slot[vi].state = EC_READY;
    c->last_used = vi;
    if (c->io) pthread_cond_broadcast(&c->io->done);
    ec_issue_next(c);
    ec_unlock(c);
    return dst;
}

const uint8_t *waste_ecache_hold(waste_ecache *c, int layer, int expert,
                                 waste_fetch_fn fetch, void *user)
{
    if (c->n_slots <= 0 || c->n_held >= WASTE_PF_MAX) return NULL;
    const uint8_t *r = waste_ecache_get(c, layer, expert, fetch, user);
    if (!r) return NULL;
    /* get() parked it in last_used, which the *next* get would release.
     * Move it into the held set instead, so the claim outlives the loop
     * that is collecting these. */
    ec_lock(c);
    if (c->last_used >= 0) {
        if (!ec_is_held(c, c->last_used)) c->held[c->n_held++] = c->last_used;
        c->last_used = -1;
    }
    ec_unlock(c);
    return r;
}

void waste_ecache_release(waste_ecache *c)
{
    ec_lock(c);
    for (int k = 0; k < c->n_held; k++) {
        const int si = c->held[k];
        /* Same test as ec_release_last: a slot reclaimed for another
         * expert may have a reader writing into it, and a volatile object
         * under an in-flight write is how a purged page becomes a silently
         * zeroed weight. */
        if (c->slot[si].state == EC_READY) ec_volatile(c, si);
    }
    c->n_held = 0;
    ec_unlock(c);
}

/* Back to a freshly-opened cache: no records, no counters. A sweep needs it
 * between arms — leaving the cache warm would hand the second configuration
 * the first one's work and measure the order instead of the setting. */
void waste_ecache_clear(waste_ecache *c)
{
    ec_lock(c);
    for (int i = 0; i < c->n_slots; i++) {
        if (c->slot[i].state == EC_READY) ec_volatile(c, i);
        c->slot[i].key = -1;
        c->slot[i].state = EC_EMPTY;
        c->slot[i].fresh = 0;
        c->slot[i].pin = 0;
        c->slot[i].hits = 0;
        c->slot[i].last = 0;
    }
    if (c->hash) memset(c->hash, 0xff, ((size_t)c->hash_mask + 1) * sizeof *c->hash);
    c->clock = c->hits = c->misses = c->bytes_read = 0;
    c->evictions = c->prefetched = c->spec_issued = 0;
    c->pf_n = c->pf_issued = 0;
    c->last_used = -1;
    c->n_held = 0;
    ec_unlock(c);
}

/* ---- learned hotlist ---------------------------------------------------- */

int waste_ecache_save_usage(const waste_ecache *c, const char *path,
                            uint64_t tokens)
{
    if (!c->slot || !path) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    waste_usage_hdr h = { WASTE_MAGIC_USAGE, 1, tokens };
    int rc = fwrite(&h, sizeof h, 1, f) == 1 ? 0 : -1;
    for (int i = 0; i < c->n_slots && !rc; i++) {
        if (c->slot[i].key < 0 || c->slot[i].state != EC_READY) continue;
        waste_usage_ent e;
        e.layer = (uint16_t)(c->slot[i].key >> 16);
        e.expert_id = (uint16_t)(c->slot[i].key & 0xFFFF);
        e.hits = c->slot[i].hits;
        e.last_seen = (uint32_t)c->slot[i].last;
        e.next_layer_top = 0;
        if (fwrite(&e, sizeof e, 1, f) != 1) rc = -1;
    }
    fclose(f);
    if (rc) remove(path);
    return rc;
}

/* Descending by hit count, ties by (layer, expert) so a given usage file
 * warms the same way every run whatever qsort does with equal keys. */
static int hits_desc(const void *a, const void *b)
{
    const waste_usage_ent *x = (const waste_usage_ent *)a;
    const waste_usage_ent *y = (const waste_usage_ent *)b;
    if (x->hits != y->hits) return x->hits > y->hits ? -1 : 1;
    if (x->layer != y->layer) return x->layer < y->layer ? -1 : 1;
    if (x->expert_id != y->expert_id) return x->expert_id < y->expert_id ? -1 : 1;
    return 0;
}

/* Preload the hottest recorded experts, best first, until the cache is
 * full. Returns how many were loaded, or -1. */
int waste_ecache_warm(waste_ecache *c, const char *path,
                      waste_fetch_fn fetch, void *user)
{
    if (!c->slot || !path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    waste_usage_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || h.magic != WASTE_MAGIC_USAGE) {
        fclose(f);
        return -1;
    }

    long start = ftell(f);
    fseek(f, 0, SEEK_END);
    const long n = (ftell(f) - start) / (long)sizeof(waste_usage_ent);
    fseek(f, start, SEEK_SET);
    if (n <= 0) { fclose(f); return 0; }

    waste_usage_ent *ent = (waste_usage_ent *)malloc((size_t)n * sizeof *ent);
    if (!ent || fread(ent, sizeof *ent, (size_t)n, f) != (size_t)n) {
        free(ent); fclose(f); return -1;
    }
    fclose(f);

    /* Hottest first. This was a partial selection over the first n_slots,
     * which is O(n_slots * n) — with K3's 46 GB cache that is 46000 by
     * 46000, seconds of a cold open spent sorting before a single expert
     * is read. The whole array sorted properly is n log n. */
    qsort(ent, (size_t)n, sizeof *ent, hits_desc);
    const int want = (int)n < c->n_slots ? (int)n : c->n_slots;

    int loaded = 0;
    for (int i = 0; i < want; i++) {
        const int32_t key = ((int32_t)ent[i].layer << 16) | ent[i].expert_id;
        if (ec_lookup(c, key) >= 0) continue;
        const int vi = ec_victim(c);
        if (vi < 0) break;
        if (fetch(user, ent[i].layer, ent[i].expert_id, c->slot[vi].data) != 0) continue;
        const int had = c->slot[vi].key >= 0;
        c->slot[vi].key = key;
        c->slot[vi].state = EC_READY;
        c->slot[vi].hits = ent[i].hits;
        c->slot[vi].last = ++c->clock;
        if (had) ec_rehash(c); else ec_insert(c, key, vi);
        ec_volatile(c, vi);        /* warmed and idle: the kernel may have it */
        loaded++;
    }
    free(ent);
    return loaded;
}
