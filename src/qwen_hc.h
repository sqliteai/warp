/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_hc.h — Qwen Gated Residual (HyperConnection Mix / Combine).
 *
 * Distinct from Kimi AttnRes. Mix reads four residual branches through a
 * rank-320 bottleneck; Combine writes a per-branch scalar gate times the
 * block output back onto the original streams.
 */

#ifndef WASTE_QWEN_HC_H
#define WASTE_QWEN_HC_H

#ifdef __cplusplus
extern "C" {
#endif

/* Grouped RMSNorm: (1 + weight) * x * rsqrt(mean(x^2)+eps) per group.
 * Official Qwen4ExpTextRMSNorm. */
void waste_qwen_rmsnorm(float *o, const float *x, const float *w,
                        int n, int group, float eps);

/* Mix: hyper [hc*hid] -> mixed [hid]. scratch >= hc*hid + rank + hc*hid. */
void waste_qwen_hc_mix(const float *hyper, const float *norm_w,
                       const float *down, const float *up,
                       int hc, int hid, int rank, float eps,
                       float *mixed, float *scratch);

/* Combine gates: same Mix, plus injection_weights [hc] =
 * 2 * sigmoid(inject(normed) / hc). scratch as Mix plus hc. */
void waste_qwen_hc_gates(const float *hyper, const float *norm_w,
                         const float *down, const float *up,
                         const float *inject, int hc, int hid, int rank,
                         float eps, float *mixed, float *inj_w, float *scratch);

/* out = hyper + inj_w[b] * block[hid] for each branch b. */
void waste_qwen_hc_combine(const float *hyper, const float *block,
                           const float *inj_w, int hc, int hid, float *out);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_HC_H */
