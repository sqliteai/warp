/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * simd_avx512.c — AVX-512 versions of the two range kernels (see simd.h).
 *
 * Same contract as simd_avx2.c: built with -mavx512f -mavx512bw, so nothing
 * here runs until waste_cpu_features() has said the machine supports them.
 * Sixteen lanes instead of eight, which mainly shows in the LUT build,
 * where the codebook has 256 entries and divides evenly.
 */

#include "simd.h"
#include "waste_backend.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <stdlib.h>

static void mvq_rows_f32_avx512(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;
    int8_t *unp = NULL;
    if (a->bits != 8) {
        unp = (int8_t *)malloc((size_t)g);
        if (!unp) return;
    }
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const int8_t *w;
            if (a->bits == 3) {
                for (int i = 0; i < g; i++)
                    unp[i] = (int8_t)waste_q3_at((const uint8_t *)row, (long)k * g + i);
                w = unp;
            } else if (a->bits == 4) {
                const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
                int i = 0;
                for (; i + 64 <= g; i += 64) {
                    const __m256i by = _mm256_loadu_si256((const __m256i *)(p4 + i / 2));
                    const __m256i lo = _mm256_and_si256(by, _mm256_set1_epi8(0x0F));
                    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(by, 4),
                                                        _mm256_set1_epi8(0x0F));
                    /* unpack works per 128-bit lane, so fix the lane order up */
                    const __m256i e0 = _mm256_unpacklo_epi8(lo, hi);
                    const __m256i e1 = _mm256_unpackhi_epi8(lo, hi);
                    const __m256i bias = _mm256_set1_epi8(8);
                    const __m256i q0 = _mm256_sub_epi8(
                        _mm256_permute2x128_si256(e0, e1, 0x20), bias);
                    const __m256i q1 = _mm256_sub_epi8(
                        _mm256_permute2x128_si256(e0, e1, 0x31), bias);
                    _mm256_storeu_si256((__m256i *)(unp + i), q0);
                    _mm256_storeu_si256((__m256i *)(unp + i + 32), q1);
                }
                for (; i < g; i++) {
                    const uint8_t byte = p4[i >> 1];
                    unp[i] = (int8_t)((i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8);
                }
                w = unp;
            } else {
                w = row + (size_t)k * g;
            }
            const float *xx = x + (size_t)k * g;
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
            __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
            int i = 0;
            for (; i + 32 <= lim; i += 32) {
                const __m512i w0 = _mm512_cvtepi8_epi32(
                    _mm_loadu_si128((const __m128i *)(w + i)));
                const __m512i w1 = _mm512_cvtepi8_epi32(
                    _mm_loadu_si128((const __m128i *)(w + i + 16)));
                s0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(w0),
                                     _mm512_loadu_ps(xx + i), s0);
                s1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(w1),
                                     _mm512_loadu_ps(xx + i + 16), s1);
            }
            float part = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
            for (; i < lim; i++) part += (float)w[i] * xx[i];
            acc += waste_f16(ws[k]) * part;
        }
        a->y[o] = acc;
    }
    free(unp);
}

static void lutb_range_avx512(int lo, int hi, void *p)
{
    const lutb_arg *a = (const lutb_arg *)p;
    const int en = a->entries, vd = a->vec_dim, st = a->stages;
    for (int v = lo; v < hi; v++) {
        const float *xv = a->x + (size_t)v * vd;
        for (int s = 0; s < st; s++) {
            const float *CT = a->booksT + (size_t)(a->cb_base + s) * en * vd;
            float *dst = a->lut + ((size_t)v * st + s) * en;
            int c = 0;
            for (; c + 64 <= en; c += 64) {
                __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(),
                       a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
                for (int d = 0; d < vd; d++) {
                    const __m512 xd = _mm512_set1_ps(xv[d]);
                    const float *cr = CT + (size_t)d * en + c;
                    a0 = _mm512_fmadd_ps(xd, _mm512_loadu_ps(cr), a0);
                    a1 = _mm512_fmadd_ps(xd, _mm512_loadu_ps(cr + 16), a1);
                    a2 = _mm512_fmadd_ps(xd, _mm512_loadu_ps(cr + 32), a2);
                    a3 = _mm512_fmadd_ps(xd, _mm512_loadu_ps(cr + 48), a3);
                }
                _mm512_storeu_ps(dst + c, a0);
                _mm512_storeu_ps(dst + c + 16, a1);
                _mm512_storeu_ps(dst + c + 32, a2);
                _mm512_storeu_ps(dst + c + 48, a3);
            }
            for (; c < en; c++) {
                float t = 0;
                for (int d = 0; d < vd; d++) t += xv[d] * CT[(size_t)d * en + c];
                dst[c] = t;
            }
        }
    }
}

#if defined(__AVX512VBMI__)
static void vq_rows_p6_avx512(int b, int e, void *p)
{
    vqp_arg *a = (vqp_arg *)p;
    const int nv = a->nv;
    const int en = 64;              /* validated at load; the tbl4 width   */
    float acc[VQ_TILE];

    /* Unpack mask: maps 48 bytes of packed 3-byte indices into 16 dwords (24-bit values) */
    static const uint8_t unpack_mask_arr[64] = {
        0, 1, 2, 63,   3, 4, 5, 63,   6, 7, 8, 63,   9, 10, 11, 63,
        12, 13, 14, 63, 15, 16, 17, 63, 18, 19, 20, 63, 21, 22, 23, 63,
        24, 25, 26, 63, 27, 28, 29, 63, 30, 31, 32, 63, 33, 34, 35, 63,
        36, 37, 38, 63, 39, 40, 41, 63, 42, 43, 44, 63, 45, 46, 47, 63
    };
    const __m512i unpack_mask = _mm512_loadu_si512((const __m512i *)unpack_mask_arr);
    const __m512i mask6 = _mm512_set1_epi32(0x3f);

    for (int r0 = b; r0 < e; r0 += VQ_TILE) {
        const int nr = (r0 + VQ_TILE <= e) ? VQ_TILE : e - r0;
        for (int r = 0; r < nr; r++) acc[r] = 0.0f;

        for (int v0 = 0; v0 < nv; v0 += WASTE_VQ_LUT_BLK) {
            int v1 = v0 + WASTE_VQ_LUT_BLK;
            if (v1 > nv) v1 = nv;
            int16_t sum[VQ_TILE];
            memset(sum, 0, sizeof sum);

            if (nr == VQ_TILE) {
                __m512i s[4];
                for (int g = 0; g < 4; g++) s[g] = _mm512_setzero_si512();

                for (int v = v0; v < v1; v++) {
                    const int8_t *T = a->lut8 + (size_t)v * 4 * en;
                    const __m512i T0 = _mm512_loadu_si512((const __m512i *)(T + 0));
                    const __m512i T1 = _mm512_loadu_si512((const __m512i *)(T + 64));
                    const __m512i T2 = _mm512_loadu_si512((const __m512i *)(T + 128));
                    const __m512i T3 = _mm512_loadu_si512((const __m512i *)(T + 192));

                    const uint8_t *ix = a->idx +
                        ((size_t)(r0 / VQ_TILE) * nv + v) * VQ_TILE * 3;

                    for (int g = 0; g < 4; g++) {
                        const __m512i raw = _mm512_maskz_loadu_epi8(0x0000FFFFFFFFFFFFULL, ix + g * 48);
                        const __m512i val = _mm512_permutexvar_epi8(unpack_mask, raw);

                        const __m512i j0 = _mm512_and_si512(val, mask6);
                        const __m512i j1 = _mm512_and_si512(_mm512_srli_epi32(val, 6), mask6);
                        const __m512i j2 = _mm512_and_si512(_mm512_srli_epi32(val, 12), mask6);
                        const __m512i j3 = _mm512_and_si512(_mm512_srli_epi32(val, 18), mask6);

                        const __m512i r0v = _mm512_permutexvar_epi8(j0, T0);
                        const __m512i r1v = _mm512_permutexvar_epi8(j1, T1);
                        const __m512i r2v = _mm512_permutexvar_epi8(j2, T2);
                        const __m512i r3v = _mm512_permutexvar_epi8(j3, T3);

                        const __m512i r0_s = _mm512_srai_epi32(_mm512_slli_epi32(r0v, 24), 24);
                        const __m512i r1_s = _mm512_srai_epi32(_mm512_slli_epi32(r1v, 24), 24);
                        const __m512i r2_s = _mm512_srai_epi32(_mm512_slli_epi32(r2v, 24), 24);
                        const __m512i r3_s = _mm512_srai_epi32(_mm512_slli_epi32(r3v, 24), 24);

                        const __m512i stage_sum = _mm512_add_epi32(
                            _mm512_add_epi32(r0_s, r1_s),
                            _mm512_add_epi32(r2_s, r3_s));
                        s[g] = _mm512_add_epi32(s[g], stage_sum);
                    }
                }
                for (int g = 0; g < 4; g++) {
                    int32_t tmp[16];
                    _mm512_storeu_si512((__m512i *)tmp, s[g]);
                    for (int i = 0; i < 16; i++) sum[g * 16 + i] = (int16_t)tmp[i];
                }
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
#endif

const char *waste_register_avx512(waste_kernels *t)
{
    t->mvq_rows_f32 = mvq_rows_f32_avx512;
    t->lutb_range = lutb_range_avx512;
#if defined(__AVX512VBMI__)
    if (waste_cpu_features() & WASTE_CPU_AVX512VBMI)
        t->vq_rows_p6 = vq_rows_p6_avx512;
#endif
    return "AVX-512";
}

#endif /* x86 */
