/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_gdn.h — Qwen Gated DeltaNet. Not KDA.
 *
 * Official decode uses torch_recurrent_gated_delta_rule; official prefill
 * uses torch_chunk_gated_delta_rule. The portable CPU path is the sequential
 * recurrence. waste_qwen_gdn_chunk is that recurrence (chunk_size ignored).
 * Persistent state is two buffers:
 *   1. recurrent S  [Hv][Dk][Dv]
 *   2. QKV short-conv ring  [(2*Hk+Hv)*D][K-1]
 * QK-repeat (16 QK heads onto 48 V heads) is an identity, not a buffer.
 * g is a per-head scalar, not a per-K-dim diagonal.
 */

#ifndef WASTE_QWEN_GDN_H
#define WASTE_QWEN_GDN_H

#ifdef __cplusplus
extern "C" {
#endif

/* g = -exp(A_log) * softplus(a + dt_bias), length Hv. */
void waste_qwen_gdn_decay(const float *a, const float *A_log, const float *dt,
                          int Hv, float *g);

/* One decode step. q,k are [Hk][Dk] (repeated onto Hv inside), v [Hv][Dv],
 * g_log and beta [Hv], S [Hv][Dk][Dv], o [Hv][Dv].
 * scratch >= Dv. */
void waste_qwen_gdn_step(int Hk, int Hv, int Dk, int Dv,
                         const float *q, const float *k, const float *v,
                         const float *g_log, const float *beta,
                         float *S, float *o, float *scratch);

/* T decode steps, time-major inputs [T][H][*]. */
void waste_qwen_gdn_forward(int T, int Hk, int Hv, int Dk, int Dv,
                            const float *q, const float *k, const float *v,
                            const float *g_log, const float *beta,
                            float *S, float *o, float *scratch);

/* Sequential recurrence over T (portable prefill). chunk_size is ignored.
 * Equal to waste_qwen_gdn_forward; proven against official chunk/recurrent
 * kernels at 16 QK / 48 V / dim 128. */
void waste_qwen_gdn_chunk(int T, int Hk, int Hv, int Dk, int Dv, int chunk,
                          const float *q, const float *k, const float *v,
                          const float *g_log, const float *beta,
                          float *S, float *o, float *scratch);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_GDN_H */
