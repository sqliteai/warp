/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * kda.c — Kimi Delta Attention kernels (see kda.h for the recurrence).
 *
 * The decode step touches the recurrent state exactly twice:
 *   pass A: decay rows by exp(g) and accumulate u = S'^T k
 *   pass B: rank-1 update S += k d^T and accumulate o = S^T q
 * Both walk S row-major, so the state streams linearly through cache.
 */

#include "kda.h"
#include "waste_backend.h"

#include <math.h>
#include <string.h>

static float l2_rnorm(const float *x, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    return 1.0f / sqrtf(s + 1e-12f);
}

void waste_kda_step(int H, int K, int V,
                    const float *q, const float *k, const float *v,
                    const float *g_log, const float *beta,
                    float *S, float *o, float *u)
{
    const float qscale = 1.0f / sqrtf((float)K);

    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * K;
        const float *kh = k + (size_t)h * K;
        const float *vh = v + (size_t)h * V;
        const float *gh = g_log + (size_t)h * K;
        float *Sh = S + (size_t)h * K * V;
        float *oh = o + (size_t)h * V;
        const float b = beta[h];

        /* q, k are L2-normalized per head; q additionally scaled by K^-0.5 */
        const float qn = l2_rnorm(qh, K) * qscale;
        const float kn = l2_rnorm(kh, K);

        memset(u, 0, (size_t)V * sizeof(float));

        /* pass A: S <- Diag(exp(g)) S, and u = S^T k (post-decay) */
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float d = expf(gh[kk]);
            const float kv = kh[kk] * kn;
            for (int i = 0; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
        }

        /* delta: d = beta * (v - u), reused as the rank-1 right factor */
        for (int i = 0; i < V; i++) u[i] = b * (vh[i] - u[i]);

        /* pass B: S += k d^T, and o = S^T q (post-update) */
        memset(oh, 0, (size_t)V * sizeof(float));
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float kv = kh[kk] * kn;
            const float qv = qh[kk] * qn;
            for (int i = 0; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
        }
    }
}

void waste_kda_forward(int T, int H, int K, int V,
                       const float *q, const float *k, const float *v,
                       const float *g_log, const float *beta,
                       float *S, float *o, float *scratch)
{
    for (int t = 0; t < T; t++) {
        waste_kda_step(H, K, V,
                       q + (size_t)t * H * K, k + (size_t)t * H * K,
                       v + (size_t)t * H * V, g_log + (size_t)t * H * K,
                       beta + (size_t)t * H,
                       S, o + (size_t)t * H * V, scratch);
    }
}

void waste_short_conv_step(int C, int KS, const float *w, const float *bias,
                           float *ring, const float *x, float *y)
{
    /* ring holds the previous KS-1 samples per channel, oldest first */
    const int R = KS - 1;
    for (int c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * KS;
        float *rc = ring + (size_t)c * R;
        float acc = bias ? bias[c] : 0.0f;
        for (int j = 0; j < R; j++) acc += rc[j] * wc[j];
        acc += x[c] * wc[R];
        for (int j = 0; j + 1 < R; j++) rc[j] = rc[j + 1];
        if (R > 0) rc[R - 1] = x[c];
        y[c] = acc / (1.0f + expf(-acc));       /* SiLU */
    }
}

void waste_rmsnorm_gated(int C, const float *x, const float *gate,
                         const float *weight, float eps, float *y)
{
    float s = 0.0f;
    for (int i = 0; i < C; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)C + eps);
    for (int i = 0; i < C; i++) {
        const float g = 1.0f / (1.0f + expf(-gate[i]));
        y[i] = x[i] * r * weight[i] * g;
    }
}

/* Universal baseline registration: fills every slot. Other backends
 * overwrite only what they implement (sqlite-vector discipline). */
void waste_mvq_rows_f32(int b, int e, void *p);
void waste_lutb_range(int lo, int hi, void *p);
void waste_vq_rows_p6(int b, int e, void *p);

const char *waste_kda_register_cpu(waste_kernels *t)
{
    t->kda_step = waste_kda_step;
    /* The portable range kernels live in model.c, next to the code that
     * builds their arguments; an ISA backend overwrites these. */
    t->mvq_rows_f32 = waste_mvq_rows_f32;
    t->lutb_range = waste_lutb_range;
    t->vq_rows_p6 = waste_vq_rows_p6;
    t->short_conv_step = waste_short_conv_step;
    t->rmsnorm_gated = waste_rmsnorm_gated;
    return "CPU";
}
