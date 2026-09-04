/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* qwen_moe.c — see qwen_moe.h. */

#include "qwen_moe.h"

#include <math.h>

int waste_qwen_moe_route(const float *logits, int E, int K, int renorm,
                         int *idx, float *w, float *prob, uint8_t *used)
{
    if (idx && w && K > 0)
        for (int j = 0; j < K; j++) { idx[j] = 0; w[j] = 0.0f; }
    if (!logits || !idx || !w || !prob || !used || E < 1 || K < 1 || K > E)
        return -1;
    float m = logits[0];
    for (int e = 1; e < E; e++) if (logits[e] > m) m = logits[e];
    float z = 0.0f;
    for (int e = 0; e < E; e++) { prob[e] = expf(logits[e] - m); z += prob[e]; }
    if (z < 1e-20f) z = 1e-20f;
    for (int e = 0; e < E; e++) { prob[e] /= z; used[e] = 0; }
    /* K passes over E rather than a sort: K is 10 and E is 512, and this
     * way the tie rule (lowest id wins, because `>` is strict) is the same
     * on every platform. */
    for (int j = 0; j < K; j++) {
        int best = -1;
        float bv = -1.0f;
        for (int e = 0; e < E; e++) {
            if (used[e]) continue;
            if (prob[e] > bv) { bv = prob[e]; best = e; }
        }
        idx[j] = best >= 0 ? best : 0;
        w[j] = best >= 0 ? prob[best] : 0.0f;
        if (best >= 0) used[best] = 1;
    }
    if (renorm && K > 1) {
        float s = 0.0f;
        for (int j = 0; j < K; j++) s += w[j];
        if (s > 1e-20f)
            for (int j = 0; j < K; j++) w[j] /= s;
    }
    return 0;
}
