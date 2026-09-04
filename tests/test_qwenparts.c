/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_qwenparts.c — isolated Qwen ops vs tools/qwenparts_ref.py.
 *
 *   ./test_qwenparts out.bin
 *   uv run --with torch --no-project python tools/qwenparts_ref.py out.bin
 *
 * Layout (little-endian; f32 unless noted):
 *   [ple]   i32 T, i32 ngram, i32 heads, i32 eos, i32 vocab, i32 seed,
 *           i32 pos, i32 ids[T], i64 multipliers[ngram],
 *           i64 sizes[heads], i32 local_rows[heads],
 *           i32 eos_T, i32 eos_ids[eos_T], i32 eos_shift,
 *           i32 eos_shifted[eos_T],
 *           i32 q8_cols, i32 q8_group, i8 q[pad], f16 scales[ng], f32 deq[cols]
 *   [hc]    i32 hc, i32 hid, i32 rank, f32 eps, hyper[hc*hid], nw[hc*hid],
 *           down[rank*hc*hid], up[hc*hid*rank], inject[hc*hc*hid],
 *           mixed[hid], inj_w[hc], combined[hc*hid]
 *   [gdn]   i32 T, i32 Hk, i32 Hv, i32 Dk, i32 Dv,
 *           q[T*Hk*Dk], k[T*Hk*Dk], v[T*Hv*Dv], a[T*Hv], A_log[Hv], dt[Hv],
 *           o_step[Hv*Dv] (last token decode), o_fwd[T*Hv*Dv], o_chunk[T*Hv*Dv]
 *   [qsa]   i32 T, i32 Hq, i32 Hkv, i32 D, i32 Dk, i32 compress, i32 topk,
 *           i32 query_pos, q_idx[Hq*Dk], raw_k[T*Dk], k_ln[Dk],
 *           cos[T*rot], sin[T*rot], sel_n, i32 sel[sel_n],
 *           q[Hq*D], k[T*Hkv*D], v[T*Hkv*D], attn[Hq*D]
 *   [moe]   i32 E, i32 K, logits[E], i32 idx[K], w[K], shared_gate,
 *           shared[H], routed[H], out[H]  (H = 8 in the dump)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"
#include "../src/qwen_gdn.h"
#include "../src/qwen_hc.h"
#include "../src/qwen_moe.h"
#include "../src/qwen_ple.h"
#include "../src/qwen_qsa.h"

static uint64_t rng = 0x243F6A8885A308D3ULL;
static float frand(void)
{
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((double)((rng >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0
                   * 2.0 - 1.0);
}

static void wr(FILE *f, const float *v, int n) { fwrite(v, sizeof(float), (size_t)n, f); }
static void wi(FILE *f, int v) { fwrite(&v, sizeof(int), 1, f); }
static void wf(FILE *f, float v) { fwrite(&v, sizeof(float), 1, f); }
static void w64(FILE *f, int64_t v) { fwrite(&v, sizeof(int64_t), 1, f); }

static uint16_t f32_to_f16(float x)
{
    union { float f; uint32_t u; } a;
    a.f = x;
    uint32_t u = a.u;
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t man = u & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man = (man | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | (man >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

int main(int argc, char **argv)
{
    const char *out = argc > 1 ? argv[1] : "qwenparts.bin";
    FILE *f = fopen(out, "wb");
    if (!f) { perror("open"); return 1; }

    /* ---- 1. PLE hashing, EOS reset, Q8G row ---------------------------- */
    {
        const int T = 8, ngram = 3, heads = 16, eos = 2, vocab = 256, seed = 0;
        const int pos = 7;
        int ids[8] = { 11, 13, 17, 19, 23, 29, 31, 37 };
        int64_t mult[3], sizes[16];
        int local[16];
        waste_qwen_ple_multipliers(mult, ngram, 0, seed, vocab);
        for (int h = 0; h < heads; h++)
            sizes[h] = waste_qwen_ple_nth_prime_after(64 - 1, h + 1);
        waste_qwen_ple_row_ids(ids, T, pos, eos, ngram, 8, mult, sizes, local);

        wi(f, T); wi(f, ngram); wi(f, heads); wi(f, eos); wi(f, vocab); wi(f, seed);
        wi(f, pos);
        for (int i = 0; i < T; i++) wi(f, ids[i]);
        for (int i = 0; i < ngram; i++) w64(f, mult[i]);
        for (int h = 0; h < heads; h++) w64(f, sizes[h]);
        for (int h = 0; h < heads; h++) wi(f, local[h]);

        const int eos_T = 6, eos_shift = 2;
        int eos_ids[6] = { 5, 7, 2, 9, 11, 13 };
        int shifted[6];
        waste_qwen_ple_shift_eos(eos_ids, eos_T, eos_shift, eos, shifted);
        wi(f, eos_T);
        for (int i = 0; i < eos_T; i++) wi(f, eos_ids[i]);
        wi(f, eos_shift);
        for (int i = 0; i < eos_T; i++) wi(f, shifted[i]);

        const int cols = 160, group = 128;
        const int ng = (cols + group - 1) / group;
        const int pad = ng * group;
        int8_t *q = calloc((size_t)pad, 1);
        uint16_t *sc = calloc((size_t)ng, 2);
        float *deq = calloc((size_t)cols, 4);
        for (int i = 0; i < cols; i++) q[i] = (int8_t)((i * 13) % 241 - 120);
        sc[0] = f32_to_f16(0.01f);
        sc[1] = f32_to_f16(0.02f);
        waste_tensor t;
        memset(&t, 0, sizeof t);
        t.q = q; t.qs = sc; t.group = group; t.bits = 8; t.rowbytes = (size_t)pad;
        t.shape[0] = 1; t.shape[1] = cols; t.ndim = 2; t.n = (size_t)cols;
        waste_deq_row(&t, 0, cols, deq);
        wi(f, cols); wi(f, group);
        fwrite(q, 1, (size_t)pad, f);
        fwrite(sc, 2, (size_t)ng, f);
        wr(f, deq, cols);
        free(q); free(sc); free(deq);
    }

    /* ---- 2. HyperConnection Mix / Combine ------------------------------ */
    {
        const int hc = 4, hid = 8, rank = 4;
        const float eps = 1e-6f;
        const int H = hc * hid;
        float *hyper = malloc((size_t)H * 4), *nw = malloc((size_t)H * 4);
        float *down = malloc((size_t)rank * H * 4), *up = malloc((size_t)H * rank * 4);
        float *inject = malloc((size_t)hc * H * 4);
        float *mixed = malloc((size_t)hid * 4), *inj_w = malloc((size_t)hc * 4);
        float *comb = malloc((size_t)H * 4), *block = malloc((size_t)hid * 4);
        float *scratch = malloc((size_t)(2 * H + rank + hc + 16) * 4);
        for (int i = 0; i < H; i++) { hyper[i] = frand(); nw[i] = frand() * 0.1f; }
        for (int i = 0; i < rank * H; i++) down[i] = frand() * 0.2f;
        for (int i = 0; i < H * rank; i++) up[i] = frand() * 0.2f;
        for (int i = 0; i < hc * H; i++) inject[i] = frand() * 0.2f;
        for (int i = 0; i < hid; i++) block[i] = frand();
        waste_qwen_hc_gates(hyper, nw, down, up, inject, hc, hid, rank, eps,
                            mixed, inj_w, scratch);
        waste_qwen_hc_combine(hyper, block, inj_w, hc, hid, comb);
        wi(f, hc); wi(f, hid); wi(f, rank); wf(f, eps);
        wr(f, hyper, H); wr(f, nw, H); wr(f, down, rank * H); wr(f, up, H * rank);
        wr(f, inject, hc * H); wr(f, block, hid);
        wr(f, mixed, hid); wr(f, inj_w, hc); wr(f, comb, H);
        free(hyper); free(nw); free(down); free(up); free(inject);
        free(mixed); free(inj_w); free(comb); free(block); free(scratch);
    }

    /* ---- 3. GDN decode + chunked prefill ------------------------------- */
    {
        const int T = 5, Hk = 2, Hv = 6, Dk = 8, Dv = 8;
        float *q = malloc((size_t)T * Hk * Dk * 4);
        float *k = malloc((size_t)T * Hk * Dk * 4);
        float *v = malloc((size_t)T * Hv * Dv * 4);
        float *a = malloc((size_t)T * Hv * 4);
        float *A = malloc((size_t)Hv * 4), *dt = malloc((size_t)Hv * 4);
        float *g = malloc((size_t)T * Hv * 4), *beta = malloc((size_t)T * Hv * 4);
        float *S1 = calloc((size_t)Hv * Dk * Dv, 4);
        float *S2 = calloc((size_t)Hv * Dk * Dv, 4);
        float *S3 = calloc((size_t)Hv * Dk * Dv, 4);
        float *o_step = malloc((size_t)Hv * Dv * 4);
        float *o_fwd = malloc((size_t)T * Hv * Dv * 4);
        float *o_chunk = malloc((size_t)T * Hv * Dv * 4);
        float *scratch = malloc((size_t)(Hv * Dk * Dv * 8 + 4096) * 4);
        for (int i = 0; i < T * Hk * Dk; i++) { q[i] = frand(); k[i] = frand(); }
        for (int i = 0; i < T * Hv * Dv; i++) v[i] = frand();
        for (int i = 0; i < T * Hv; i++) a[i] = frand();
        for (int i = 0; i < Hv; i++) { A[i] = frand() * 2.0f; dt[i] = frand(); }
        for (int t = 0; t < T; t++) {
            waste_qwen_gdn_decay(a + t * Hv, A, dt, Hv, g + t * Hv);
            for (int h = 0; h < Hv; h++)
                beta[t * Hv + h] = 1.0f / (1.0f + expf(-a[t * Hv + h] * 0.5f));
        }
        /* last-token decode from zeros, then full forward and chunk */
        waste_qwen_gdn_step(Hk, Hv, Dk, Dv,
                            q + (T - 1) * Hk * Dk, k + (T - 1) * Hk * Dk,
                            v + (T - 1) * Hv * Dv, g + (T - 1) * Hv,
                            beta + (T - 1) * Hv, S1, o_step, scratch);
        waste_qwen_gdn_forward(T, Hk, Hv, Dk, Dv, q, k, v, g, beta, S2, o_fwd, scratch);
        waste_qwen_gdn_chunk(T, Hk, Hv, Dk, Dv, 64, q, k, v, g, beta, S3, o_chunk, scratch);
        wi(f, T); wi(f, Hk); wi(f, Hv); wi(f, Dk); wi(f, Dv);
        wr(f, q, T * Hk * Dk); wr(f, k, T * Hk * Dk); wr(f, v, T * Hv * Dv);
        wr(f, a, T * Hv); wr(f, A, Hv); wr(f, dt, Hv);
        wr(f, g, T * Hv); wr(f, beta, T * Hv);
        wr(f, o_step, Hv * Dv); wr(f, o_fwd, T * Hv * Dv); wr(f, o_chunk, T * Hv * Dv);
        free(q); free(k); free(v); free(a); free(A); free(dt); free(g); free(beta);
        free(S1); free(S2); free(S3); free(o_step); free(o_fwd); free(o_chunk);
        free(scratch);
    }

    /* ---- 4. QSA indexer + attention ------------------------------------ */
    {
        const int T = 10, Hq = 4, Hkv = 2, D = 8, Dk = 8, compress = 4, topk = 2;
        const int query_pos = 9, rot = 4;
        float *q_idx = malloc((size_t)Hq * Dk * 4);
        float *raw_k = malloc((size_t)T * Dk * 4);
        float *k_ln = malloc((size_t)Dk * 4);
        float *cos = malloc((size_t)T * rot * 4), *sinv = malloc((size_t)T * rot * 4);
        float *q = malloc((size_t)Hq * D * 4);
        float *k = malloc((size_t)T * Hkv * D * 4);
        float *v = malloc((size_t)T * Hkv * D * 4);
        float *attn = malloc((size_t)Hq * D * 4);
        float *scratch = malloc((size_t)(T * Hq + Hq * D + 256) * 4);
        int sel[32];
        int n_complete = (query_pos + 1) / compress;
        float *work = malloc((size_t)(n_complete * Dk + n_complete + Dk) * 4);
        int *taken = malloc((size_t)(n_complete > 0 ? n_complete : 1) * sizeof(int));
        for (int i = 0; i < Hq * Dk; i++) q_idx[i] = frand();
        for (int i = 0; i < T * Dk; i++) raw_k[i] = frand();
        for (int i = 0; i < Dk; i++) k_ln[i] = frand() * 0.1f;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < rot; r++) {
                float ang = (float)t * 0.1f * (float)(r + 1);
                cos[t * rot + r] = cosf(ang);
                sinv[t * rot + r] = sinf(ang);
            }
        for (int i = 0; i < Hq * D; i++) q[i] = frand();
        for (int i = 0; i < T * Hkv * D; i++) { k[i] = frand(); v[i] = frand(); }
        int nsel = waste_qwen_qsa_select(q_idx, Hq, Dk, raw_k, T, query_pos,
                                         cos, sinv, rot, k_ln, 1e-6f,
                                         compress, topk, sel, work, taken);
        waste_qwen_qsa_attn(q, Hq, D, k, v, Hkv, T, sel, nsel,
                            1.0f / sqrtf((float)D), attn, scratch);
        wi(f, T); wi(f, Hq); wi(f, Hkv); wi(f, D); wi(f, Dk);
        wi(f, compress); wi(f, topk); wi(f, query_pos); wi(f, rot);
        wr(f, q_idx, Hq * Dk); wr(f, raw_k, T * Dk); wr(f, k_ln, Dk);
        wr(f, cos, T * rot); wr(f, sinv, T * rot);
        wi(f, nsel);
        for (int i = 0; i < nsel; i++) wi(f, sel[i]);
        wr(f, q, Hq * D); wr(f, k, T * Hkv * D); wr(f, v, T * Hkv * D);
        wr(f, attn, Hq * D);
        free(q_idx); free(raw_k); free(k_ln); free(cos); free(sinv);
        free(q); free(k); free(v); free(attn); free(scratch);
        free(work); free(taken);
    }

    /* ---- 5. Softmax top-k MoE, original router order ------------------- */
    {
        const int E = 16, K = 4, H = 8;
        float logits[16], w[4], shared[8], routed[8], out[8], prob[16];
        int idx[4];
        uint8_t used[16];
        float gate_in = 0.4f;
        for (int e = 0; e < E; e++) logits[e] = frand() * 3.0f;
        for (int i = 0; i < H; i++) { shared[i] = frand(); routed[i] = 0.0f; }
        waste_qwen_moe_route(logits, E, K, 1, idx, w, prob, used);
        /* fake expert outputs in router order: expert j contributes w[j]*e_j
         * where e_j is a rank-1 pattern of the expert id, so order matters. */
        for (int j = 0; j < K; j++)
            for (int i = 0; i < H; i++)
                routed[i] += w[j] * ((float)(idx[j] + 1) * 0.1f + (float)i * 0.01f);
        const float sg = 1.0f / (1.0f + expf(-gate_in));
        for (int i = 0; i < H; i++) out[i] = routed[i] + sg * shared[i];
        wi(f, E); wi(f, K); wi(f, H);
        wr(f, logits, E);
        for (int j = 0; j < K; j++) wi(f, idx[j]);
        wr(f, w, K);
        wf(f, gate_in);
        wr(f, shared, H); wr(f, routed, H); wr(f, out, H);
    }

    /* ---- official geometry, bounded allocations ----------------------- */
    {
        const int hc = 4, hid = 8, rank = 320;
        const float eps = 1e-6f;
        const int H = hc * hid;
        float *hyper = malloc((size_t)H * 4), *nw = malloc((size_t)H * 4);
        float *down = malloc((size_t)rank * H * 4), *up = malloc((size_t)H * rank * 4);
        float *inject = malloc((size_t)hc * H * 4);
        float *mixed = malloc((size_t)hid * 4), *inj_w = malloc((size_t)hc * 4);
        float *comb = malloc((size_t)H * 4), *block = malloc((size_t)hid * 4);
        float *scratch = malloc((size_t)(2 * H + rank + hc + 16) * 4);
        for (int i = 0; i < H; i++) { hyper[i] = frand(); nw[i] = frand() * 0.1f; }
        for (int i = 0; i < rank * H; i++) down[i] = frand() * 0.05f;
        for (int i = 0; i < H * rank; i++) up[i] = frand() * 0.05f;
        for (int i = 0; i < hc * H; i++) inject[i] = frand() * 0.05f;
        for (int i = 0; i < hid; i++) block[i] = frand();
        waste_qwen_hc_gates(hyper, nw, down, up, inject, hc, hid, rank, eps,
                            mixed, inj_w, scratch);
        waste_qwen_hc_combine(hyper, block, inj_w, hc, hid, comb);
        wi(f, hc); wi(f, hid); wi(f, rank); wf(f, eps);
        wr(f, hyper, H); wr(f, nw, H); wr(f, down, rank * H); wr(f, up, H * rank);
        wr(f, inject, hc * H); wr(f, block, hid);
        wr(f, mixed, hid); wr(f, inj_w, hc); wr(f, comb, H);
        free(hyper); free(nw); free(down); free(up); free(inject);
        free(mixed); free(inj_w); free(comb); free(block); free(scratch);
    }
    {
        const int T = 4, Hk = 16, Hv = 48, Dk = 128, Dv = 128;
        float *q = malloc((size_t)T * Hk * Dk * 4);
        float *k = malloc((size_t)T * Hk * Dk * 4);
        float *v = malloc((size_t)T * Hv * Dv * 4);
        float *a = malloc((size_t)T * Hv * 4);
        float *A = malloc((size_t)Hv * 4), *dt = malloc((size_t)Hv * 4);
        float *g = malloc((size_t)T * Hv * 4), *beta = malloc((size_t)T * Hv * 4);
        float *S2 = calloc((size_t)Hv * Dk * Dv, 4);
        float *S3 = calloc((size_t)Hv * Dk * Dv, 4);
        float *o_fwd = malloc((size_t)T * Hv * Dv * 4);
        float *o_chunk = malloc((size_t)T * Hv * Dv * 4);
        float *scratch = malloc((size_t)(Dv + 16) * 4);
        for (int i = 0; i < T * Hk * Dk; i++) { q[i] = frand(); k[i] = frand(); }
        for (int i = 0; i < T * Hv * Dv; i++) v[i] = frand();
        for (int i = 0; i < T * Hv; i++) a[i] = frand();
        for (int i = 0; i < Hv; i++) { A[i] = frand() * 2.0f; dt[i] = frand(); }
        for (int t = 0; t < T; t++) {
            waste_qwen_gdn_decay(a + t * Hv, A, dt, Hv, g + t * Hv);
            for (int h = 0; h < Hv; h++)
                beta[t * Hv + h] = 1.0f / (1.0f + expf(-a[t * Hv + h] * 0.5f));
        }
        waste_qwen_gdn_forward(T, Hk, Hv, Dk, Dv, q, k, v, g, beta, S2, o_fwd, scratch);
        waste_qwen_gdn_chunk(T, Hk, Hv, Dk, Dv, 64, q, k, v, g, beta, S3, o_chunk, scratch);
        wi(f, T); wi(f, Hk); wi(f, Hv); wi(f, Dk); wi(f, Dv);
        wr(f, o_fwd, T * Hv * Dv); wr(f, o_chunk, T * Hv * Dv);
        free(q); free(k); free(v); free(a); free(A); free(dt); free(g); free(beta);
        free(S2); free(S3); free(o_fwd); free(o_chunk); free(scratch);
    }
    {
        const int T = 2051, Hq = 2, Dk = 8, compress = 4, topk = 512, rot = 4;
        const int query_pos = T - 1;
        float *q_idx = malloc((size_t)Hq * Dk * 4);
        float *raw_k = malloc((size_t)T * Dk * 4);
        float *k_ln = malloc((size_t)Dk * 4);
        float *cos = malloc((size_t)T * rot * 4), *sinv = malloc((size_t)T * rot * 4);
        int *sel = malloc((size_t)(topk * compress + compress) * sizeof(int));
        int n_complete = (query_pos + 1) / compress;
        float *work = malloc((size_t)(n_complete * Dk + n_complete + Dk) * 4);
        int *taken = malloc((size_t)n_complete * sizeof(int));
        for (int i = 0; i < Hq * Dk; i++) q_idx[i] = frand();
        for (int i = 0; i < T * Dk; i++) raw_k[i] = frand();
        for (int i = 0; i < Dk; i++) k_ln[i] = frand() * 0.1f;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < rot; r++) {
                float ang = (float)t * 0.1f * (float)(r + 1);
                cos[t * rot + r] = cosf(ang);
                sinv[t * rot + r] = sinf(ang);
            }
        int nsel = waste_qwen_qsa_select(q_idx, Hq, Dk, raw_k, T, query_pos,
                                         cos, sinv, rot, k_ln, 1e-6f,
                                         compress, topk, sel, work, taken);
        float pooled[8];
        waste_qwen_qsa_pool_block(raw_k, 4, Dk, k_ln, 1e-6f, pooled);
        wi(f, T); wi(f, Hq); wi(f, Dk); wi(f, compress); wi(f, topk);
        wi(f, nsel); wr(f, pooled, Dk); wr(f, k_ln, Dk); wr(f, raw_k, 4 * Dk);
        free(q_idx); free(raw_k); free(k_ln); free(cos); free(sinv);
        free(sel); free(work); free(taken);
    }
    {
        const int E = 16, K = 10;
        float logits[16], w[10], prob[16];
        int idx[10];
        uint8_t used[16];
        for (int e = 0; e < E; e++) logits[e] = frand() * 3.0f;
        waste_qwen_moe_route(logits, E, K, 1, idx, w, prob, used);
        wi(f, E); wi(f, K);
        wr(f, logits, E);
        for (int j = 0; j < K; j++) wi(f, idx[j]);
        wr(f, w, K);
    }

    fclose(f);
    printf("wrote %s\n", out);
    return 0;
}
