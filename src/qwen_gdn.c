/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* qwen_gdn.c — see qwen_gdn.h. Official gated-delta recurrence, not KDA. */

#include "qwen_gdn.h"

#include <math.h>
#include <string.h>

static float l2_rnorm(const float *x, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return 1.0f / sqrtf(s + 1e-6f);
}

void waste_qwen_gdn_decay(const float *a, const float *A_log, const float *dt,
                          int Hv, float *g)
{
    for (int h = 0; h < Hv; h++) {
        const float z = a[h] + dt[h];
        const float sp = logf(1.0f + expf(-fabsf(z))) + (z > 0.0f ? z : 0.0f);
        g[h] = -expf(A_log[h]) * sp;
    }
}

static void gdn_one(int Hv, int Dk, int Dv, int group,
                    const float *q, const float *k, const float *v,
                    const float *g_log, const float *beta,
                    float *S, float *o, float *u)
{
    const float qscale = 1.0f / sqrtf((float)Dk);
    for (int h = 0; h < Hv; h++) {
        const int src = h / group;
        const float *qh = q + (size_t)src * Dk;
        const float *kh = k + (size_t)src * Dk;
        const float *vh = v + (size_t)h * Dv;
        float *Sh = S + (size_t)h * Dk * Dv;
        float *oh = o + (size_t)h * Dv;
        const float qn = l2_rnorm(qh, Dk) * qscale;
        const float kn = l2_rnorm(kh, Dk);
        const float decay = expf(g_log[h]);
        const float b = beta[h];

        memset(u, 0, (size_t)Dv * sizeof(float));
        for (int kk = 0; kk < Dk; kk++) {
            float *row = Sh + (size_t)kk * Dv;
            const float kv = kh[kk] * kn;
            for (int i = 0; i < Dv; i++) { row[i] *= decay; u[i] += row[i] * kv; }
        }
        for (int i = 0; i < Dv; i++) u[i] = b * (vh[i] - u[i]);
        memset(oh, 0, (size_t)Dv * sizeof(float));
        for (int kk = 0; kk < Dk; kk++) {
            float *row = Sh + (size_t)kk * Dv;
            const float kv = kh[kk] * kn;
            const float qv = qh[kk] * qn;
            for (int i = 0; i < Dv; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
        }
    }
}

void waste_qwen_gdn_step(int Hk, int Hv, int Dk, int Dv,
                         const float *q, const float *k, const float *v,
                         const float *g_log, const float *beta,
                         float *S, float *o, float *scratch)
{
    const int group = Hk > 0 ? Hv / Hk : 1;
    gdn_one(Hv, Dk, Dv, group > 0 ? group : 1, q, k, v, g_log, beta, S, o, scratch);
}

void waste_qwen_gdn_forward(int T, int Hk, int Hv, int Dk, int Dv,
                            const float *q, const float *k, const float *v,
                            const float *g_log, const float *beta,
                            float *S, float *o, float *scratch)
{
    for (int t = 0; t < T; t++) {
        waste_qwen_gdn_step(Hk, Hv, Dk, Dv,
                            q + (size_t)t * Hk * Dk, k + (size_t)t * Hk * Dk,
                            v + (size_t)t * Hv * Dv, g_log + (size_t)t * Hv,
                            beta + (size_t)t * Hv,
                            S, o + (size_t)t * Hv * Dv, scratch);
    }
}

/* Portable prefill: sequential recurrence. Official GPU prefill calls
 * torch_chunk_gated_delta_rule; the two match at fp32 for official geometry. */
void waste_qwen_gdn_chunk(int T, int Hk, int Hv, int Dk, int Dv, int chunk,
                          const float *q, const float *k, const float *v,
                          const float *g_log, const float *beta,
                          float *S, float *o, float *scratch)
{
    (void)chunk;
    waste_qwen_gdn_forward(T, Hk, Hv, Dk, Dv, q, k, v, g_log, beta, S, o, scratch);
}
