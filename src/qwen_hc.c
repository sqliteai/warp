/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* qwen_hc.c — see qwen_hc.h. */

#include "qwen_hc.h"

#include <math.h>
#include <string.h>

static float silu(float x) { return x / (1.0f + expf(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

static void matvec(float *y, const float *W, const float *x, int out, int in)
{
    for (int i = 0; i < out; i++) {
        float s = 0.0f;
        const float *w = W + (size_t)i * in;
        for (int j = 0; j < in; j++) s += w[j] * x[j];
        y[i] = s;
    }
}

void waste_qwen_rmsnorm(float *o, const float *x, const float *w,
                        int n, int group, float eps)
{
    if (group <= 0 || group > n) group = n;
    const int ng = n / group;
    for (int g = 0; g < ng; g++) {
        const float *xg = x + (size_t)g * group;
        float *og = o + (size_t)g * group;
        const float *wg = w + (size_t)g * group;
        float s = 0.0f;
        for (int i = 0; i < group; i++) s += xg[i] * xg[i];
        const float r = 1.0f / sqrtf(s / (float)group + eps);
        for (int i = 0; i < group; i++)
            og[i] = xg[i] * r * (1.0f + wg[i]);
    }
}

void waste_qwen_hc_mix(const float *hyper, const float *norm_w,
                       const float *down, const float *up,
                       int hc, int hid, int rank, float eps,
                       float *mixed, float *scratch)
{
    const int H = hc * hid;
    float *normed = scratch;
    float *lo = scratch + H;
    float *gate = scratch + H + rank;
    waste_qwen_rmsnorm(normed, hyper, norm_w, H, hid, eps);
    matvec(lo, down, normed, rank, H);
    for (int i = 0; i < rank; i++) lo[i] = silu(lo[i] / (float)hc);
    matvec(gate, up, lo, H, rank);
    for (int i = 0; i < H; i++) gate[i] = sigmoid(gate[i]);
    for (int d = 0; d < hid; d++) {
        float s = 0.0f;
        for (int b = 0; b < hc; b++) {
            const int i = b * hid + d;
            s += gate[i] * normed[i];
        }
        mixed[d] = s / (float)hc;
    }
}

void waste_qwen_hc_gates(const float *hyper, const float *norm_w,
                         const float *down, const float *up,
                         const float *inject, int hc, int hid, int rank,
                         float eps, float *mixed, float *inj_w, float *scratch)
{
    const int H = hc * hid;
    waste_qwen_hc_mix(hyper, norm_w, down, up, hc, hid, rank, eps, mixed, scratch);
    float *normed = scratch;
    float *tmp = scratch + H + rank + H;
    waste_qwen_rmsnorm(normed, hyper, norm_w, H, hid, eps);
    matvec(tmp, inject, normed, hc, H);
    for (int b = 0; b < hc; b++)
        inj_w[b] = 2.0f * sigmoid(tmp[b] / (float)hc);
}

void waste_qwen_hc_combine(const float *hyper, const float *block,
                           const float *inj_w, int hc, int hid, float *out)
{
    for (int b = 0; b < hc; b++)
        for (int d = 0; d < hid; d++)
            out[b * hid + d] = hyper[b * hid + d] + inj_w[b] * block[d];
}
