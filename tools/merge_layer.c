/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * merge_layer.c — collapse one expert bank into a single averaged expert.
 *
 * The k3-mini experiment: replace a layer's E routed experts with one
 * expert whose weights are a convex combination of theirs,
 *
 *     Wbar_kind = sum_e alpha_e * W_e,kind        sum_e alpha_e = 1
 *
 * so the MoE layer becomes an ordinary FFN and the container stops
 * streaming. The alphas come from outside (tools/merge_experts.py) because
 * choosing them is the interesting part and measuring them is a different
 * job from decoding a terabyte.
 *
 * Why C and not numpy: K3 is 82,432 records, 1.02 TB, 2.7e12 weights to
 * decode. The decode itself is three table lookups and an FMA per eight
 * weights, so the whole pass is disk-bound if and only if the arithmetic
 * keeps up, which numpy's per-call overhead on 11 MB matrices does not.
 *
 * Reads a key=value job file rather than a pile of positional arguments:
 * a merge needs three matrix shapes, a codebook base and a stage count,
 * and a command line carrying eleven integers is a command line nobody
 * can read back six months later.
 *
 * Not in src/: this converts models, and by the rule in CLAUDE.md that
 * makes it a tool. It links libwaste.a only for waste_vq_encode and
 * waste_crc32 — the two pieces that must agree bit-for-bit with what the
 * engine and tools/convert.py already do.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#include "../src/waste_format.h"
#include "../src/crc32.h"

#define MAXK 3                   /* gate, up, down                          */
#define IDX_BLOCK 64             /* rows per index block; matches convert.py */

void waste_vq_encode(const float *X, int n, const float *books, int stages,
                     int entries, int dim, uint8_t *out, int nthreads);

/* fp16 -> float, subnormals included. Copied rather than included from
 * src/simd.h because that header pulls the whole kernel dispatch in; the
 * value has to match waste_f16 exactly and the two are checked against
 * each other by the round-trip in tools/merge_experts.py --selftest. */
static float f16(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f, mn = h & 0x3ff;
    float f;
    uint32_t bits;
    if (e == 0) {
        f = (float)mn * 5.9604644775390625e-08f;      /* 2^-24 */
        return (h >> 15) ? -f : f;
    }
    bits = sign | ((e + 112u) << 23) | (mn << 13);
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t to_f16(float f)
{
    uint32_t b;
    uint32_t sign, exp;
    int32_t e;
    memcpy(&b, &f, 4);
    sign = (b >> 31) << 15;
    e = (int32_t)((b >> 23) & 0xff) - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7bff);     /* clamp, no inf */
    if (e <= 0) return (uint16_t)sign;                 /* flush tiny to zero */
    exp = (uint32_t)e << 10;
    return (uint16_t)(sign | exp | ((b >> 13) & 0x3ff));
}

/* ---- job file ---------------------------------------------------------- */

typedef struct {
    char bank[1024], codebooks[1024], alpha[1024], gains[1024], out[1024];
    int n_experts, cb_base, stages, entries, vec_dim, layer, threads, check;
    int clusters;                /* records written; 1 = collapse to one   */
    int m[MAXK], n[MAXK];
} job;

static int job_read(job *j, const char *path)
{
    FILE *f = fopen(path, "r");
    char line[2048];
    if (!f) return -1;
    memset(j, 0, sizeof *j);
    j->vec_dim = 8;
    j->entries = 256;
    j->stages = 3;
    j->clusters = 1;
    while (fgets(line, sizeof line, f)) {
        char *eq = strchr(line, '=');
        char *k = line, *v;
        size_t L;
        if (!eq || line[0] == '#') continue;
        *eq = 0;
        v = eq + 1;
        L = strlen(v);
        while (L && (v[L - 1] == '\n' || v[L - 1] == '\r' || v[L - 1] == ' ')) v[--L] = 0;
        if (!strcmp(k, "bank")) snprintf(j->bank, sizeof j->bank, "%s", v);
        else if (!strcmp(k, "codebooks")) snprintf(j->codebooks, sizeof j->codebooks, "%s", v);
        else if (!strcmp(k, "alpha")) snprintf(j->alpha, sizeof j->alpha, "%s", v);
        else if (!strcmp(k, "gains")) snprintf(j->gains, sizeof j->gains, "%s", v);
        else if (!strcmp(k, "clusters")) j->clusters = atoi(v);
        else if (!strcmp(k, "out")) snprintf(j->out, sizeof j->out, "%s", v);
        else if (!strcmp(k, "n_experts")) j->n_experts = atoi(v);
        else if (!strcmp(k, "cb_base")) j->cb_base = atoi(v);
        else if (!strcmp(k, "stages")) j->stages = atoi(v);
        else if (!strcmp(k, "entries")) j->entries = atoi(v);
        else if (!strcmp(k, "vec_dim")) j->vec_dim = atoi(v);
        else if (!strcmp(k, "layer")) j->layer = atoi(v);
        else if (!strcmp(k, "threads")) j->threads = atoi(v);
        else if (!strcmp(k, "check")) j->check = atoi(v);
        else if (!strcmp(k, "m0")) j->m[0] = atoi(v);
        else if (!strcmp(k, "n0")) j->n[0] = atoi(v);
        else if (!strcmp(k, "m1")) j->m[1] = atoi(v);
        else if (!strcmp(k, "n1")) j->n[1] = atoi(v);
        else if (!strcmp(k, "m2")) j->m[2] = atoi(v);
        else if (!strcmp(k, "n2")) j->n[2] = atoi(v);
    }
    fclose(f);
    if (!j->bank[0] || !j->out[0] || j->n_experts < 1) return -1;
    if (j->clusters < 1 || j->clusters > 256 || j->clusters > j->n_experts) return -1;
    for (int k = 0; k < MAXK; k++)
        if (j->m[k] < 1 || j->n[k] < 1 || j->n[k] % j->vec_dim) return -1;
    return 0;
}

/* ---- decode + accumulate ----------------------------------------------- */

typedef struct {
    const uint8_t *idx;          /* blocked indices for this matrix          */
    const uint16_t *scale;       /* one fp16 per output row                  */
    float *acc;                  /* [M][N] running sum                       */
    const float *books;          /* [stages][entries][vec_dim]               */
    float alpha;
    int M, N, nvr, stages, entries, vec_dim;
    int r0, r1;
} dec_arg;

static void *dec_run(void *p)
{
    dec_arg *a = (dec_arg *)p;
    const int nvr = a->nvr, st = a->stages, vd = a->vec_dim;
    const size_t per = (size_t)a->entries * vd;
    for (int r = a->r0; r < a->r1; r++) {
        const int blk = r / IDX_BLOCK, rib = r % IDX_BLOCK;
        const float s = a->alpha * f16(a->scale[r]);
        float *dst = a->acc + (size_t)r * a->N;
        for (int v = 0; v < nvr; v++) {
            const uint8_t *ix = a->idx +
                (((size_t)blk * nvr + v) * IDX_BLOCK + rib) * st;
            float acc8[64];
            for (int d = 0; d < vd; d++) acc8[d] = 0.0f;
            for (int t = 0; t < st; t++) {
                const float *C = a->books + (size_t)t * per + (size_t)ix[t] * vd;
                for (int d = 0; d < vd; d++) acc8[d] += C[d];
            }
            for (int d = 0; d < vd; d++) dst[v * vd + d] += s * acc8[d];
        }
    }
    return NULL;
}

static void run_rows(dec_arg *base, int M, int nthreads)
{
    pthread_t th[64];
    dec_arg arg[64];
    int n = nthreads < 1 ? 1 : (nthreads > 64 ? 64 : nthreads);
    int per = (M + n - 1) / n, used = 0;
    for (int i = 0; i < n; i++) {
        int r0 = i * per, r1 = r0 + per;
        if (r0 >= M) break;
        if (r1 > M) r1 = M;
        arg[used] = *base;
        arg[used].r0 = r0;
        arg[used].r1 = r1;
        used++;
    }
    for (int i = 1; i < used; i++) pthread_create(&th[i], NULL, dec_run, &arg[i]);
    if (used) dec_run(&arg[0]);
    for (int i = 1; i < used; i++) pthread_join(th[i], NULL);
}

/* ---- main -------------------------------------------------------------- */

static int ncpu(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
}

int main(int argc, char **argv)
{
    job j;
    if (argc < 2) { fprintf(stderr, "usage: %s job.txt\n", argv[0]); return 2; }
    if (job_read(&j, argv[1]) != 0) { fprintf(stderr, "bad job file\n"); return 2; }
    if (j.threads <= 0) j.threads = ncpu();

    /* codebooks: [kind][stage] at absolute id cb_base + kind*stages + stage */
    const size_t cbrec = 16 + (size_t)j.entries * j.vec_dim * 2;
    const size_t per = (size_t)j.entries * j.vec_dim;
    float *books = (float *)malloc((size_t)MAXK * j.stages * per * sizeof(float));
    {
        FILE *cf = fopen(j.codebooks, "rb");
        uint16_t *raw = (uint16_t *)malloc(per * 2);
        if (!cf || !books || !raw) { fprintf(stderr, "codebooks: open\n"); return 1; }
        for (int k = 0; k < MAXK; k++)
            for (int s = 0; s < j.stages; s++) {
                const long id = j.cb_base + k * j.stages + s;
                waste_codebook_hdr h;
                if (fseek(cf, (long)(id * (long)cbrec), SEEK_SET) ||
                    fread(&h, sizeof h, 1, cf) != 1 ||
                    h.magic != WASTE_MAGIC_CODEBOOK ||
                    (int)h.n_entries != j.entries || h.vec_dim != j.vec_dim) {
                    fprintf(stderr, "codebook %ld: not where the manifest says\n", id);
                    return 1;
                }
                if (fread(raw, 2, per, cf) != per) { fprintf(stderr, "codebook read\n"); return 1; }
                for (size_t i = 0; i < per; i++)
                    books[((size_t)k * j.stages + s) * per + i] = f16(raw[i]);
            }
        free(raw);
        fclose(cf);
    }

    /* ---- alphas and per-cluster gains ---------------------------------
     * alpha is [clusters][n_experts]; each row is a convex combination and
     * must sum to 1, so every cluster is a proper mean of its members and
     * the SiLU downstream sees inputs of the right magnitude. The cluster's
     * *share* of the layer's routed weight is a separate number, applied to
     * the down projection only — scaling gate/up would move the
     * nonlinearity's operating point instead of the output. */
    const int C = j.clusters, E = j.n_experts;
    float *alpha = (float *)calloc((size_t)C * E, sizeof(float));
    float *gain = (float *)malloc((size_t)C * sizeof(float));
    if (!alpha || !gain) return 1;
    for (int c = 0; c < C; c++) gain[c] = 1.0f;

    /* No default. A missing alpha= used to mean "uniform", and that silently
     * produced three byte-identical containers from three different merge
     * policies — the driver had built the alpha files and forgotten to name
     * them in the job. A merge with no stated weights is a job description
     * that is incomplete, not one with an obvious reading. */
    if (!j.alpha[0]) {
        fprintf(stderr, "job: alpha=PATH is required (%d x %d float32)\n", C, E);
        return 2;
    }
    {
        FILE *af = fopen(j.alpha, "rb");
        if (!af || fread(alpha, sizeof(float), (size_t)C * E, af) != (size_t)C * E) {
            fprintf(stderr, "alpha: want %d x %d floats from %s\n", C, E, j.alpha);
            return 1;
        }
        fclose(af);
        for (int c = 0; c < C; c++) {
            double sum = 0;
            for (int e = 0; e < E; e++) sum += alpha[(size_t)c * E + e];
            if (sum < 0.999 || sum > 1.001) {
                fprintf(stderr, "alpha: cluster %d sums to %.6f, not 1\n", c, sum);
                return 1;
            }
        }
    }
    if (j.gains[0]) {
        FILE *gf = fopen(j.gains, "rb");
        if (!gf || fread(gain, sizeof(float), (size_t)C, gf) != (size_t)C) {
            fprintf(stderr, "gains: want %d floats from %s\n", C, j.gains);
            return 1;
        }
        fclose(gf);
    }

    /* accumulators, one set per cluster */
    size_t elems[MAXK];
    float **acc = (float **)calloc((size_t)C * MAXK, sizeof(float *));
    if (!acc) return 1;
    for (int k = 0; k < MAXK; k++) elems[k] = (size_t)j.m[k] * j.n[k];
    for (int c = 0; c < C; c++)
        for (int k = 0; k < MAXK; k++) {
            acc[(size_t)c * MAXK + k] = (float *)calloc(elems[k], sizeof(float));
            if (!acc[(size_t)c * MAXK + k]) {
                fprintf(stderr, "acc c%d k%d: out of memory\n", c, k);
                return 1;
            }
        }

    /* ---- one pass over the bank ---------------------------------------
     * Every record is read once and accumulated into whichever clusters
     * claim it. Reading the bank once per cluster would be C times 1.02 TB
     * on K3, which is the difference between five minutes and an afternoon. */
    FILE *bf = fopen(j.bank, "rb");
    if (!bf) { fprintf(stderr, "%s: cannot open\n", j.bank); return 1; }
    fseek(bf, 0, SEEK_END);
    const long long fsz = ftello(bf);
    const long long rec_bytes = fsz / E;
    if (rec_bytes <= 0 || rec_bytes % WASTE_ALIGN || rec_bytes * E != fsz) {
        fprintf(stderr, "%s: %lld bytes is not %d whole records\n", j.bank, fsz, E);
        return 1;
    }
    uint8_t *rec = (uint8_t *)malloc((size_t)rec_bytes);
    if (!rec) return 1;

    int used = 0;
    for (int e = 0; e < E; e++) {
        int any = 0;
        for (int c = 0; c < C && !any; c++) any = alpha[(size_t)c * E + e] != 0.0f;
        if (!any) continue;                  /* in no cluster: never read it */
        if (fseeko(bf, (off_t)((long long)e * rec_bytes), SEEK_SET) ||
            fread(rec, 1, (size_t)rec_bytes, bf) != (size_t)rec_bytes) {
            fprintf(stderr, "L%d e%d: short read\n", j.layer, e);
            return 1;
        }
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        if (h->magic != WASTE_MAGIC_EXPERT || h->layer != j.layer || h->expert_id != e) {
            fprintf(stderr, "L%d e%d: record header disagrees with the index\n", j.layer, e);
            return 1;
        }
        const uint32_t off[MAXK] = { h->gate_off, h->up_off, h->down_off };
        const uint16_t *sc = (const uint16_t *)(rec + h->chan_corr_off);
        for (int c = 0; c < C; c++) {
            const float a_ce = alpha[(size_t)c * E + e];
            size_t sc_at = 0;
            if (a_ce == 0.0f) continue;
            for (int k = 0; k < MAXK; k++) {
                dec_arg a;
                memset(&a, 0, sizeof a);
                a.idx = rec + off[k];
                a.scale = sc + sc_at;
                a.acc = acc[(size_t)c * MAXK + k];
                a.books = books + (size_t)k * j.stages * per;
                a.alpha = a_ce;
                a.M = j.m[k];
                a.N = j.n[k];
                a.nvr = j.n[k] / j.vec_dim;
                a.stages = j.stages;
                a.entries = j.entries;
                a.vec_dim = j.vec_dim;
                run_rows(&a, j.m[k], j.threads);
                sc_at += (size_t)j.m[k];
            }
        }
        used++;
    }
    fclose(bf);
    free(rec);

    /* ---- re-encode and write, cluster by cluster ------------------------
     * Reusing the layer's existing codebooks keeps codebooks.bin
     * byte-identical and cb_base valid, which matters because the merged
     * records have to be loadable by an unmodified engine. The codebooks
     * were fitted on per-row-absmax-normalized single experts; a merged
     * matrix is renormalized the same way, so it lands in the same domain.
     * What it is NOT is refitted, and `check=1` is what reports that cost
     * separately from the merge's own. */
    const int check = j.check;
    FILE *of = fopen(j.out, "wb");
    size_t rec_out = 0;
    if (!of) { fprintf(stderr, "%s: cannot create\n", j.out); return 1; }
    for (int c = 0; c < C; c++) {
        uint8_t *payload[MAXK];
        size_t paylen[MAXK];
        uint16_t *scales[MAXK];
        /* The cluster's share of the routed weight, folded into down only.
         * A uniform positive scale on a row is absorbed exactly by that
         * row's fp16 scale, so this costs no accuracy. */
        if (gain[c] != 1.0f) {
            float *d = acc[(size_t)c * MAXK + 2];
            for (size_t i = 0; i < elems[2]; i++) d[i] *= gain[c];
        }
        for (int k = 0; k < MAXK; k++) {
            const int M = j.m[k], N = j.n[k], vd = j.vec_dim, st = j.stages;
            const int nvr = N / vd, nb = (M + IDX_BLOCK - 1) / IDX_BLOCK;
            float *W = acc[(size_t)c * MAXK + k];
            float *X = (float *)malloc(elems[k] * sizeof(float));
            uint8_t *flat = (uint8_t *)malloc((size_t)M * nvr * st);
            scales[k] = (uint16_t *)malloc((size_t)M * 2);
            paylen[k] = (size_t)nb * nvr * IDX_BLOCK * st;
            payload[k] = (uint8_t *)calloc(paylen[k], 1);
            if (!X || !flat || !scales[k] || !payload[k]) { fprintf(stderr, "encode: oom\n"); return 1; }
            for (int r = 0; r < M; r++) {
                const float *w = W + (size_t)r * N;
                float mx = 1e-8f;
                for (int i = 0; i < N; i++) { const float v = fabsf(w[i]); if (v > mx) mx = v; }
                scales[k][r] = to_f16(mx);
                /* Normalize by the value the record will actually carry, not
                 * by the float one: f16 rounding of the scale otherwise shows
                 * up as a systematic gain error on every row. */
                {
                    const float inv = 1.0f / f16(scales[k][r]);
                    float *x = X + (size_t)r * N;
                    for (int i = 0; i < N; i++) x[i] = w[i] * inv;
                }
            }
            waste_vq_encode(X, M * nvr, books + (size_t)k * st * per, st,
                            j.entries, vd, flat, j.threads);
            /* flat is [row-major vector][stage]; the record wants
             * [block][vec][row_in_block][stage]. */
            for (int r = 0; r < M; r++) {
                const int blk = r / IDX_BLOCK, rib = r % IDX_BLOCK;
                for (int v = 0; v < nvr; v++)
                    for (int t = 0; t < st; t++)
                        payload[k][(((size_t)blk * nvr + v) * IDX_BLOCK + rib) * st + t] =
                            flat[((size_t)r * nvr + v) * st + t];
            }
            free(X);
            free(flat);

            /* Decode what was just encoded and diff it against the
             * accumulator. This separates the two losses the merge stacks:
             * the averaging itself, which is the experiment, and re-quantizing
             * the average, which is overhead. Without it a bad merge has two
             * suspects and no way to tell them apart. */
            if (check && c == 0) {
                float *back = (float *)calloc(elems[k], sizeof(float));
                double num = 0, den = 0;
                dec_arg a;
                if (!back) { fprintf(stderr, "check: oom\n"); return 1; }
                memset(&a, 0, sizeof a);
                a.idx = payload[k];
                a.scale = scales[k];
                a.acc = back;
                a.books = books + (size_t)k * st * per;
                a.alpha = 1.0f;
                a.M = M; a.N = N; a.nvr = nvr;
                a.stages = st; a.entries = j.entries; a.vec_dim = vd;
                run_rows(&a, M, j.threads);
                for (size_t i = 0; i < elems[k]; i++) {
                    const double d = (double)back[i] - (double)W[i];
                    num += d * d;
                    den += (double)W[i] * (double)W[i];
                }
                fprintf(stderr, "L%d c0 %s: requant rel L2 %.4f\n", j.layer,
                        k == 0 ? "gate" : k == 1 ? "up" : "down",
                        den > 0 ? sqrt(num / den) : 0.0);
                free(back);
            }
            free(W);
            acc[(size_t)c * MAXK + k] = NULL;
        }

        {
            const size_t hdr = sizeof(waste_expert_hdr);
            size_t body = 0, off = hdr;
            uint32_t o[MAXK];
            for (int k = 0; k < MAXK; k++) { o[k] = (uint32_t)off; off += paylen[k]; body += paylen[k]; }
            const uint32_t corr_off = (uint32_t)off;
            for (int k = 0; k < MAXK; k++) { off += (size_t)j.m[k] * 2; body += (size_t)j.m[k] * 2; }
            const size_t total = hdr + body;
            const size_t blocks = (total + WASTE_ALIGN - 1) / WASTE_ALIGN;
            uint8_t *buf = (uint8_t *)calloc(blocks * WASTE_ALIGN, 1);
            waste_expert_hdr *h = (waste_expert_hdr *)buf;
            if (!buf) return 1;
            /* A bank is indexed by multiplying the expert id by a fixed
             * record size, so every record in it has to be the same size.
             * The shapes are identical across clusters, so this can only
             * fail if something above changed per cluster — say it rather
             * than write a bank whose index arithmetic is wrong. */
            if (rec_out && blocks * WASTE_ALIGN != rec_out) {
                fprintf(stderr, "L%d c%d: record size %zu != %zu\n",
                        j.layer, c, blocks * WASTE_ALIGN, rec_out);
                return 1;
            }
            rec_out = blocks * WASTE_ALIGN;
            for (int k = 0; k < MAXK; k++) memcpy(buf + o[k], payload[k], paylen[k]);
            {
                size_t at = corr_off;
                for (int k = 0; k < MAXK; k++) {
                    memcpy(buf + at, scales[k], (size_t)j.m[k] * 2);
                    at += (size_t)j.m[k] * 2;
                }
            }
            h->magic = WASTE_MAGIC_EXPERT;
            h->layer = (uint16_t)j.layer;
            h->expert_id = (uint16_t)c;
            h->fmt = WQ_VQ3R;
            h->flags = 0;
            h->codebook_id = (uint16_t)j.cb_base;
            h->lowrank_id = 0;
            h->reserved0 = 0;
            h->rec_4k_blocks = (uint32_t)blocks;
            h->gate_off = o[0];
            h->up_off = o[1];
            h->down_off = o[2];
            h->chan_corr_off = corr_off;
            h->crc32 = waste_crc32(buf + hdr, body);
            h->reserved1[0] = h->reserved1[1] = 0;
            if (fwrite(buf, 1, rec_out, of) != rec_out) {
                fprintf(stderr, "%s: write failed\n", j.out);
                return 1;
            }
            free(buf);
        }
        for (int k = 0; k < MAXK; k++) { free(payload[k]); free(scales[k]); }
    }
    fclose(of);
    printf("L%d: %d/%d experts -> %d record(s) of %zu bytes\n",
           j.layer, used, E, C, rec_out);
    return 0;
}
