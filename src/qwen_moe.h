/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_moe.h — Qwen4ExpTextTopKRouter.
 *
 * Softmax over every expert, then top-k, then an optional renormalize —
 * not the sigmoid-plus-bias router the Kimi and GLM families use, and not
 * interchangeable with it: softmax couples the experts, so the same
 * logits give a different selection under the two.
 */

#ifndef WASTE_QWEN_MOE_H
#define WASTE_QWEN_MOE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Top-k over `E` softmax probabilities.
 *
 * `idx` and `w` are length K and come back in *selection* order — highest
 * probability first — and not sorted by expert id. That order is part of
 * the contract: the MoE reduction accumulates in it, and floating-point
 * addition is not associative, so re-ordering the experts changes the last
 * bits of every token. Ties go to the lower expert id, which is what makes
 * the order reproducible at all.
 *
 * `prob` is E floats of scratch and `used` E bytes; both are written.
 * `idx`/`w` are always written (zeros when the arguments are unusable), so
 * a caller that ignores the return value routes to expert 0 with weight 0
 * rather than reading uninitialized memory. Returns 0, or -1.
 */
int waste_qwen_moe_route(const float *logits, int E, int K, int renorm,
                         int *idx, float *w, float *prob, uint8_t *used);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_MOE_H */
