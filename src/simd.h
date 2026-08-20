/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * simd.h — the two range kernels that carry the engine's arithmetic, in a
 * form other translation units can implement.
 *
 * Everything hot is either a gather (the VQ apply, which no SIMD helps —
 * see LEARNED.md §12) or one of these two:
 *
 *   mvq_rows_f32   quantized weights x f32 activations, one row at a time.
 *                  Every trunk projection goes through it.
 *   lutb_range     the VQ codebook table, dst[c] = sum_d x[d]*C[d][c].
 *
 * They take a (begin, end, arg) range because that is what the thread pool
 * hands out, so dispatch costs one indirect call per range rather than one
 * per row. The argument structs live here for the same reason the kernels
 * do: an ISA-specific file has to see them.
 *
 * model.c registers the portable versions (with inline NEON where the
 * compiler targets ARM). simd_avx2.c and simd_avx512.c register theirs
 * when waste_cpu_features() says the machine can run them.
 */

#ifndef WASTE_SIMD_H
#define WASTE_SIMD_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *y; const int8_t *W; const uint16_t *ws;
    const int8_t *xq; const float *xs; int in, ng, group, bits;
    size_t rowbytes;
} mvq_arg;

typedef struct {
    float *lut; const float *booksT; const float *x;
    int cb_base, stages, entries, vec_dim;
} lutb_arg;

typedef struct {
    float *y; const uint8_t *idx; const uint16_t *scale;
    const int8_t *lut8; const float *lscale;
    int nv;
} vqp_arg;

#ifndef VQ_TILE
#define VQ_TILE 64
#endif

#ifndef WASTE_VQ_LUT_BLK
#define WASTE_VQ_LUT_BLK 32
#endif

/* Unpack of the converter's little-endian 4x6 packing. Kept as one macro so
 * the scalar path, NEON, and AVX-512 paths cannot drift apart. */
#define P6_J0(b0, b1, b2) ((b0) & 0x3f)
#define P6_J1(b0, b1, b2) ((((b0) >> 6) | ((b1) << 2)) & 0x3f)
#define P6_J2(b0, b1, b2) ((((b1) >> 4) | ((b2) << 4)) & 0x3f)
#define P6_J3(b0, b1, b2) (((b2) >> 2) & 0x3f)

/* fp16 scale -> float, and one signed 3-bit code out of a packed row. Both
 * are needed by every implementation, and both are small enough that a
 * shared inline beats a call. */
static inline float waste_f16(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f, mn = h & 0x3ff;
    if (e == 0) {
        /* Subnormal, value mantissa * 2^-24 — not zero. Flushing these
         * killed a whole group of 128 weights wherever a per-group scale
         * fell below 6.1e-05, which is exactly what happens in rows whose
         * weights are uniformly small. Found in K3's vision tower, where
         * one row of fc0 came out 27% wrong; the same bug was silently
         * degrading any quantized tensor with a tiny group. */
        float f = (float)mn * 5.9604644775390625e-08f;   /* 2^-24 */
        return sign ? -f : f;
    }
    const uint32_t bits = sign | ((e + 112u) << 23) | (mn << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline int waste_q3_at(const uint8_t *p, long i)
{
    const long off = i * 3;
    const int v = (p[off >> 3] >> (off & 7)) | (p[(off >> 3) + 1] << (8 - (off & 7)));
    return (v & 7) - 4;
}

#ifdef __cplusplus
}
#endif
#endif /* WASTE_SIMD_H */
