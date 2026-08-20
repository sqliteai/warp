/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * kda_neon.c — ARM NEON specialization of the KDA kernels.
 *
 * Compiled only on ARM; registers over the CPU baseline at init. Must
 * produce the same results as kda.c (tools/kda_ref.py checks both against
 * the reference with WASTE_BACKEND=cpu / auto).
 */

#if defined(__ARM_NEON) || defined(__aarch64__)

#include "kda.h"
#include "waste_backend.h"
#include "simd.h"

#include <arm_neon.h>
#include <math.h>
#include <string.h>

static float l2_rnorm_neon(const float *x, int n)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += x[i] * x[i];
    return 1.0f / sqrtf(s + 1e-12f);
}

static void kda_step_neon(int H, int K, int V,
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

        const float qn = l2_rnorm_neon(qh, K) * qscale;
        const float kn = l2_rnorm_neon(kh, K);

        memset(u, 0, (size_t)V * sizeof(float));

        /* pass A: decay rows, accumulate u = S'^T k */
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float32x4_t vd = vdupq_n_f32(expf(gh[kk]));
            const float32x4_t vk = vdupq_n_f32(kh[kk] * kn);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vmulq_f32(vld1q_f32(row + i), vd);
                vst1q_f32(row + i, r);
                vst1q_f32(u + i, vfmaq_f32(vld1q_f32(u + i), r, vk));
            }
            const float d = expf(gh[kk]), kv = kh[kk] * kn;
            for (; i < V; i++) { row[i] *= d; u[i] += row[i] * kv; }
        }

        /* delta */
        {
            const float32x4_t vb = vdupq_n_f32(b);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t d = vsubq_f32(vld1q_f32(vh + i), vld1q_f32(u + i));
                vst1q_f32(u + i, vmulq_f32(vb, d));
            }
            for (; i < V; i++) u[i] = b * (vh[i] - u[i]);
        }

        /* pass B: rank-1 update, accumulate o = S^T q */
        memset(oh, 0, (size_t)V * sizeof(float));
        for (int kk = 0; kk < K; kk++) {
            float *row = Sh + (size_t)kk * V;
            const float32x4_t vk = vdupq_n_f32(kh[kk] * kn);
            const float32x4_t vq = vdupq_n_f32(qh[kk] * qn);
            int i = 0;
            for (; i + 4 <= V; i += 4) {
                float32x4_t r = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(u + i), vk);
                vst1q_f32(row + i, r);
                vst1q_f32(oh + i, vfmaq_f32(vld1q_f32(oh + i), r, vq));
            }
            const float kv = kh[kk] * kn, qv = qh[kk] * qn;
            for (; i < V; i++) { row[i] += u[i] * kv; oh[i] += row[i] * qv; }
        }
    }
}

static void vq_rows_p6_neon(int b, int e, void *p)
{
    vqp_arg *a = (vqp_arg *)p;
    const int nv = a->nv;
    const int en = 64;              /* validated at load; the tbl4 width   */
    float acc[VQ_TILE];

    for (int r0 = b; r0 < e; r0 += VQ_TILE) {
        const int nr = (r0 + VQ_TILE <= e) ? VQ_TILE : e - r0;
        for (int r = 0; r < nr; r++) acc[r] = 0.0f;

        for (int v0 = 0; v0 < nv; v0 += WASTE_VQ_LUT_BLK) {
            int v1 = v0 + WASTE_VQ_LUT_BLK;
            if (v1 > nv) v1 = nv;
            /* int16 is enough: 4 stages x 32 positions x 127 = 16256. */
            int16_t sum[VQ_TILE];
            memset(sum, 0, sizeof sum);

            if (nr == VQ_TILE) {
                int16x8_t s[8];
                for (int i = 0; i < 8; i++) s[i] = vdupq_n_s16(0);
                for (int v = v0; v < v1; v++) {
                    const int8_t *T = a->lut8 + (size_t)v * 4 * en;
                    int8x16x4_t T0, T1, T2, T3;
                    for (int k = 0; k < 4; k++) {
                        T0.val[k] = vld1q_s8(T +   0 + k * 16);
                        T1.val[k] = vld1q_s8(T +  64 + k * 16);
                        T2.val[k] = vld1q_s8(T + 128 + k * 16);
                        T3.val[k] = vld1q_s8(T + 192 + k * 16);
                    }
                    const uint8_t *ix = a->idx +
                        ((size_t)(r0 / VQ_TILE) * nv + v) * VQ_TILE * 3;
                    for (int g = 0; g < 4; g++) {
                        const uint8x16x3_t I = vld3q_u8(ix + g * 48);
                        const uint8x16_t j0 =
                            vandq_u8(I.val[0], vdupq_n_u8(0x3f));
                        const uint8x16_t j1 = vandq_u8(
                            vorrq_u8(vshrq_n_u8(I.val[0], 6),
                                     vshlq_n_u8(I.val[1], 2)), vdupq_n_u8(0x3f));
                        const uint8x16_t j2 = vandq_u8(
                            vorrq_u8(vshrq_n_u8(I.val[1], 4),
                                     vshlq_n_u8(I.val[2], 4)), vdupq_n_u8(0x3f));
                        const uint8x16_t j3 = vshrq_n_u8(I.val[2], 2);
                        /* Each lookup is widened on its own rather than
                         * summed in int8 first: two int8 tables can add to
                         * 254 and the table is worth a bit more than the
                         * two instructions that would save. */
                        const int8x16_t r0v = vqtbl4q_s8(T0, j0);
                        const int8x16_t r1v = vqtbl4q_s8(T1, j1);
                        const int8x16_t r2v = vqtbl4q_s8(T2, j2);
                        const int8x16_t r3v = vqtbl4q_s8(T3, j3);
                        int16x8_t lo = s[g * 2], hi = s[g * 2 + 1];
                        lo = vaddw_s8(lo, vget_low_s8(r0v));
                        hi = vaddw_s8(hi, vget_high_s8(r0v));
                        lo = vaddw_s8(lo, vget_low_s8(r1v));
                        hi = vaddw_s8(hi, vget_high_s8(r1v));
                        lo = vaddw_s8(lo, vget_low_s8(r2v));
                        hi = vaddw_s8(hi, vget_high_s8(r2v));
                        lo = vaddw_s8(lo, vget_low_s8(r3v));
                        hi = vaddw_s8(hi, vget_high_s8(r3v));
                        s[g * 2] = lo; s[g * 2 + 1] = hi;
                    }
                }
                for (int i = 0; i < 8; i++) vst1q_s16(sum + i * 8, s[i]);
            } else {
                for (int v = v0; v < v1; v++) {
                    const int8_t *T = a->lut8 + (size_t)v * 4 * en;
                    const uint8_t *ix = a->idx +
                        ((size_t)(r0 / VQ_TILE) * nv + v) * VQ_TILE * 3;
                    for (int r = 0; r < nr; r++) {
                        const unsigned b0 = ix[r * 3], b1 = ix[r * 3 + 1],
                                       b2 = ix[r * 3 + 2];
                        sum[r] = (int16_t)(sum[r] +
                            T[P6_J0(b0, b1, b2)] +
                            T[en + P6_J1(b0, b1, b2)] +
                            T[2 * en + P6_J2(b0, b1, b2)] +
                            T[3 * en + P6_J3(b0, b1, b2)]);
                    }
                }
            }
            const float ls = a->lscale[v0 / WASTE_VQ_LUT_BLK];
            for (int r = 0; r < nr; r++) acc[r] += ls * (float)sum[r];
        }
        for (int r = 0; r < nr; r++)
            a->y[r0 + r] = acc[r] * waste_f16(a->scale[r0 + r]);
    }
}

const char *waste_kda_register_neon(waste_kernels *t)
{
    t->kda_step = kda_step_neon;
    t->vq_rows_p6 = vq_rows_p6_neon;
    /* short_conv_step and rmsnorm_gated stay on the CPU baseline until
     * they show up in a profile — partial override is the whole point. */
    /* The name reports what this build *uses*, not what the CPU offers.
     * waste_cpu_features() detects dotprod and i8mm, and neither drives a
     * kernel yet — no SDOT or SMMLA is emitted anywhere in the engine — so
     * naming them here only made `waste version` overstate the binary.
     * Add the suffix back in the same commit that adds the kernel. */
    return "NEON";
}

#endif /* ARM */
