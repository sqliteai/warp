/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * test_hip_parity.c — Automated multi-step numerical drift verification test
 * comparing the active backend against the CPU reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "../src/waste_backend.h"
#include "../src/simd.h"

static float host_f16(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t e = (h >> 10) & 0x1f, mn = h & 0x3ff;
    if (e == 0) {
        float f = (float)mn * 5.9604644775390625e-08f;
        return sign ? -f : f;
    }
    const uint32_t bits = sign | ((e + 112u) << 23) | (mn << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t h = (uint16_t)(((x >> 16) & 0x8000) |
                 ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
                 ((x >> 13) & 0x03ff));
    return h;
}

/* Reference CPU MatVec */
static void cpu_mvq8(float *y, const int8_t *W, const uint16_t *ws, const float *x,
                     int M, int K, int group, size_t rowbytes)
{
    int ng = (K + group - 1) / group;
    for (int r = 0; r < M; r++) {
        const int8_t *w_row = W + (size_t)r * rowbytes;
        const uint16_t *sc_row = ws + (size_t)r * ng;
        float acc = 0.0f;
        for (int k = 0; k < ng; k++) {
            float s = host_f16(sc_row[k]);
            float part = 0.0f;
            for (int i = 0; i < group; i++) {
                part += (float)w_row[k * group + i] * x[k * group + i];
            }
            acc += s * part;
        }
        y[r] = acc;
    }
}

int main(void)
{
    waste_backend_init(WASTE_BE_AUTO);
    printf("Testing Backend Parity [%s]...\n", waste_backend_name());

    int M = 4096;
    int K = 4096;
    int group = 128;
    int ng = K / group;
    size_t rowbytes = (size_t)K;

    int8_t *W = (int8_t *)malloc((size_t)M * rowbytes);
    uint16_t *ws = (uint16_t *)malloc((size_t)M * ng * sizeof(uint16_t));
    float *x = (float *)malloc((size_t)K * sizeof(float));
    float *y_cpu = (float *)malloc((size_t)M * sizeof(float));
    float *y_act = (float *)malloc((size_t)M * sizeof(float));

    if (!W || !ws || !x || !y_cpu || !y_act) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (size_t i = 0; i < (size_t)M * rowbytes; i++) W[i] = (int8_t)((i % 255) - 127);
    for (size_t i = 0; i < (size_t)M * ng; i++) ws[i] = float_to_half(0.0125f * ((i % 10) + 1));
    for (int i = 0; i < K; i++) x[i] = ((float)(i % 100) - 50.0f) * 0.01f;

    /* Multi-step recurrent simulation (50 steps) */
    float max_accum_drift = 0.0f;
    for (int step = 0; step < 50; step++) {
        cpu_mvq8(y_cpu, W, ws, x, M, K, group, rowbytes);

        mvq_arg arg = { y_act, W, ws, NULL, x, K, ng, group, 8, rowbytes };
        waste_k.mvq_rows_f32(0, M, &arg);

        for (int i = 0; i < M; i++) {
            float abs_d = fabsf(y_act[i] - y_cpu[i]);
            float rel_d = abs_d / (fabsf(y_cpu[i]) + 1e-9f);
            if (rel_d > max_accum_drift) max_accum_drift = rel_d;
        }

        /* Perturb input x state for next step */
        for (int i = 0; i < K; i++) {
            x[i] = (x[i] * 0.9f) + (y_act[i % M] * 0.0001f);
        }
    }

    printf("  [50-Step Recurrent Parity Check] Max Relative Drift: %.2e\n", max_accum_drift);

    waste_backend_release_host_buffers();
    free(W);
    free(ws);
    free(x);
    free(y_cpu);
    free(y_act);

    if (max_accum_drift < 1e-3f) {
        printf("  => PASS: Multi-step numerical parity within FP noise bounds.\n");
        return 0;
    } else {
        printf("  => FAIL: Numerical drift exceeded tolerance.\n");
        return 1;
    }
}
