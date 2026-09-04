/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_qsa.h — Qwen Sparse Attention. Not MLA.
 *
 * Indexer: FP32 mean of four keys, MRoPE, ReLU-sum scores, top-512 blocks,
 * plus the 0–3 tail tokens, then attention over the original K/V.
 * Production decode calls waste_qwen_qsa_select; there is one pooling path.
 */

#ifndef WASTE_QWEN_QSA_H
#define WASTE_QWEN_QSA_H

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved MRoPE: freqs [3][half] -> out [half]. section is 3 ints. */
void waste_qwen_mrope_interleave(const float *freqs_t, const float *freqs_h,
                                 const float *freqs_w, const int *section,
                                 int half, float *out);

/* Apply RoPE to the first rotary_dim components (cos/sin length).
 * rotary_dim > 256 is refused (no write). */
int waste_qwen_rope_apply(float *x, int dim, const float *cos, const float *sin,
                          int rotary_dim);

/* Mean of `compress` consecutive raw keys, then RMSNorm (1+w).
 * raw_k is [compress][Dk]. */
void waste_qwen_qsa_pool_block(const float *raw_k, int compress, int Dk,
                               const float *k_ln_w, float eps, float *out);

/* Select tokens for one query. raw_k is [T][Dk] (one indexer KV head).
 * Writes up to budget+compress-1 ids into sel, returns the count.
 * q_heads is [Hq][Dk] already layernormed and RoPE'd.
 * work >= n_complete*Dk + n_complete + Dk floats; taken >= n_complete ints.
 * work/taken may be NULL when n_complete == 0. */
int waste_qwen_qsa_select(const float *q_heads, int Hq, int Dk,
                          const float *raw_k, int T, int query_pos,
                          const float *full_cos, const float *full_sin,
                          int rotary_dim, const float *k_ln_w, float eps,
                          int compress, int block_topk,
                          int *sel, float *work, int *taken);

/* Causal softmax attention over selected tokens. q [Hq][D], k/v [T][Hkv][D],
 * n_rep = Hq/Hkv, scaling = 1/sqrt(D). sel[n_sel] are token indices. */
void waste_qwen_qsa_attn(const float *q, int Hq, int D,
                         const float *k, const float *v, int Hkv, int T,
                         const int *sel, int n_sel, float scale,
                         float *out, float *scratch);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_QSA_H */
