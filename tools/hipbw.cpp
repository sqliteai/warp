/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * hipbw.cpp — Standalone PCIe transfer, VRAM streaming, and Q8G/Q4G MatVec
 * mathematical parity verification harness against CPU reference.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

struct HipMatVecParams {
    int in;
    int ng;
    int group;
    int bits;
    uint32_t rowbytes;
    uint32_t m;
};

__device__ __forceinline__ float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}

__global__ void mvq8_hip_kernel(float * __restrict__ y,
                                const int8_t * __restrict__ W,
                                const uint16_t * __restrict__ ws,
                                const float * __restrict__ x,
                                HipMatVecParams p)
{
    const uint32_t warp_id = threadIdx.x / warpSize;
    const uint32_t lane_id = threadIdx.x % warpSize;
    const uint32_t warps_per_block = blockDim.x / warpSize;
    const uint32_t row = blockIdx.x * warps_per_block + warp_id;

    if (row >= p.m) return;

    const char4 *w_row = (const char4 *)(W + (size_t)row * p.rowbytes);
    const __half *sc_row = (const __half *)(ws + (size_t)row * p.ng);
    const float4 *x_vec = (const float4 *)x;

    const uint32_t nvec = p.rowbytes >> 2;
    const uint32_t gshift = (p.group == 128) ? 7u : (uint32_t)(__log2f((float)p.group) + 0.5f);

    float acc = 0.0f;
    for (uint32_t j = lane_id; j < nvec; j += warpSize) {
        const char4 wv = w_row[j];
        const float4 xv = x_vec[j];
        const float scale = __half2float(sc_row[(j << 2) >> gshift]);

        const float dot4 = (float)wv.x * xv.x +
                           (float)wv.y * xv.y +
                           (float)wv.z * xv.z +
                           (float)wv.w * xv.w;
        acc += scale * dot4;
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) {
        y[row] = acc;
    }
}

__global__ void mvq4_hip_kernel(float * __restrict__ y,
                                const uint8_t * __restrict__ W,
                                const uint16_t * __restrict__ ws,
                                const float * __restrict__ x,
                                HipMatVecParams p)
{
    const uint32_t warp_id = threadIdx.x / warpSize;
    const uint32_t lane_id = threadIdx.x % warpSize;
    const uint32_t warps_per_block = blockDim.x / warpSize;
    const uint32_t row = blockIdx.x * warps_per_block + warp_id;

    if (row >= p.m) return;

    const uint4 *w_row = (const uint4 *)(W + (size_t)row * p.rowbytes);
    const __half *sc_row = (const __half *)(ws + (size_t)row * p.ng);

    const uint32_t nvec = p.rowbytes >> 4;
    const uint32_t gshift = (p.group == 128) ? 7u : (uint32_t)(__log2f((float)p.group) + 0.5f);

    float acc = 0.0f;
    for (uint32_t j = lane_id; j < nvec; j += warpSize) {
        const uint4 packed = w_row[j];
        const uint32_t base = j << 5;
        float part = 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            uint32_t word = (b == 0) ? packed.x : ((b == 1) ? packed.y : ((b == 2) ? packed.z : packed.w));
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                uint32_t byte_val = (word >> (8 * k)) & 0xffu;
                uint32_t i = base + b * 8 + k * 2;
                part += (float)((int)(byte_val & 0x0fu) - 8) * x[i]
                      + (float)((int)(byte_val >> 4)   - 8) * x[i + 1];
            }
        }
        acc += __half2float(sc_row[base >> gshift]) * part;
    }

    acc = warp_reduce_sum(acc);
    if (lane_id == 0) {
        y[row] = acc;
    }
}

static inline float host_f16(uint16_t h)
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

static void cpu_mvq4(float *y, const uint8_t *W, const uint16_t *ws, const float *x,
                     int M, int K, int group, size_t rowbytes)
{
    int ng = (K + group - 1) / group;
    for (int r = 0; r < M; r++) {
        const uint8_t *w_row = W + (size_t)r * rowbytes;
        const uint16_t *sc_row = ws + (size_t)r * ng;
        float acc = 0.0f;
        for (int k = 0; k < ng; k++) {
            float s = host_f16(sc_row[k]);
            float part = 0.0f;
            for (int i = 0; i < group; i += 2) {
                uint8_t byte = w_row[k * (group / 2) + i / 2];
                int v0 = (int)(byte & 0x0f) - 8;
                int v1 = (int)(byte >> 4) - 8;
                part += (float)v0 * x[k * group + i] + (float)v1 * x[k * group + i + 1];
            }
            acc += s * part;
        }
        y[r] = acc;
    }
}

static void test_kernel_parity(int warpSize_val)
{
    printf("\n=== Mathematical Parity & Precision Validation ===\n");
    int M = 4096;
    int K = 4096;
    int group = 128;
    int ng = K / group;

    /* 1. Test Q8G */
    size_t rowbytes8 = (size_t)K;
    int8_t *h_w8 = (int8_t *)malloc((size_t)M * rowbytes8);
    uint16_t *h_ws = (uint16_t *)malloc((size_t)M * ng * sizeof(uint16_t));
    float *h_x = (float *)malloc((size_t)K * sizeof(float));
    float *h_y_cpu = (float *)malloc((size_t)M * sizeof(float));
    float *h_y_gpu = (float *)malloc((size_t)M * sizeof(float));

    for (size_t i = 0; i < (size_t)M * rowbytes8; i++) h_w8[i] = (int8_t)((i % 255) - 127);
    for (size_t i = 0; i < (size_t)M * ng; i++) h_ws[i] = float_to_half(0.0125f * ((i % 10) + 1));
    for (int i = 0; i < K; i++) h_x[i] = ((float)(i % 100) - 50.0f) * 0.01f;

    cpu_mvq8(h_y_cpu, h_w8, h_ws, h_x, M, K, group, rowbytes8);

    void *d_w8, *d_ws, *d_x, *d_y;
    CHECK_HIP(hipMalloc(&d_w8, (size_t)M * rowbytes8));
    CHECK_HIP(hipMalloc(&d_ws, (size_t)M * ng * sizeof(uint16_t)));
    CHECK_HIP(hipMalloc(&d_x, (size_t)K * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_y, (size_t)M * sizeof(float)));

    CHECK_HIP(hipMemcpy(d_w8, h_w8, (size_t)M * rowbytes8, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_ws, h_ws, (size_t)M * ng * sizeof(uint16_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_x, h_x, (size_t)K * sizeof(float), hipMemcpyHostToDevice));

    HipMatVecParams p8 = { K, ng, group, 8, (uint32_t)rowbytes8, (uint32_t)M };
    int block_dim = 256;
    int warps_per_block = block_dim / warpSize_val;
    int grid_dim = (M + warps_per_block - 1) / warps_per_block;

    mvq8_hip_kernel<<<grid_dim, block_dim>>>((float *)d_y, (const int8_t *)d_w8, (const uint16_t *)d_ws, (const float *)d_x, p8);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_y_gpu, d_y, (size_t)M * sizeof(float), hipMemcpyDeviceToHost));

    float max_rel_err8 = 0.0f;
    float max_abs_err8 = 0.0f;
    for (int i = 0; i < M; i++) {
        float abs_d = fabsf(h_y_gpu[i] - h_y_cpu[i]);
        float rel_d = abs_d / (fabsf(h_y_cpu[i]) + 1e-9f);
        if (abs_d > max_abs_err8) max_abs_err8 = abs_d;
        if (rel_d > max_rel_err8) max_rel_err8 = rel_d;
    }
    printf("  [mvq8 Q8G] Max Abs Error: %.2e | Max Rel Error: %.2e | Status: %s\n",
           max_abs_err8, max_rel_err8, (max_rel_err8 < 1e-3f) ? "PASS (FP ACCUMULATION NOISE)" : "FAIL");

    /* 2. Test Q4G */
    size_t rowbytes4 = (size_t)K / 2;
    uint8_t *h_w4 = (uint8_t *)malloc((size_t)M * rowbytes4);
    for (size_t i = 0; i < (size_t)M * rowbytes4; i++) h_w4[i] = (uint8_t)(i & 0xff);

    cpu_mvq4(h_y_cpu, h_w4, h_ws, h_x, M, K, group, rowbytes4);

    void *d_w4;
    CHECK_HIP(hipMalloc(&d_w4, (size_t)M * rowbytes4));
    CHECK_HIP(hipMemcpy(d_w4, h_w4, (size_t)M * rowbytes4, hipMemcpyHostToDevice));

    HipMatVecParams p4 = { K, ng, group, 4, (uint32_t)rowbytes4, (uint32_t)M };
    mvq4_hip_kernel<<<grid_dim, block_dim>>>((float *)d_y, (const uint8_t *)d_w4, (const uint16_t *)d_ws, (const float *)d_x, p4);
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_y_gpu, d_y, (size_t)M * sizeof(float), hipMemcpyDeviceToHost));

    float max_rel_err4 = 0.0f;
    float max_abs_err4 = 0.0f;
    for (int i = 0; i < M; i++) {
        float abs_d = fabsf(h_y_gpu[i] - h_y_cpu[i]);
        float rel_d = abs_d / (fabsf(h_y_cpu[i]) + 1e-9f);
        if (abs_d > max_abs_err4) max_abs_err4 = abs_d;
        if (rel_d > max_rel_err4) max_rel_err4 = rel_d;
    }
    printf("  [mvq4 Q4G] Max Abs Error: %.2e | Max Rel Error: %.2e | Status: %s\n",
           max_abs_err4, max_rel_err4, (max_rel_err4 < 1e-3f) ? "PASS (FP ACCUMULATION NOISE)" : "FAIL");

    CHECK_HIP(hipFree(d_w8));
    CHECK_HIP(hipFree(d_w4));
    CHECK_HIP(hipFree(d_ws));
    CHECK_HIP(hipFree(d_x));
    CHECK_HIP(hipFree(d_y));
    free(h_w8);
    free(h_w4);
    free(h_ws);
    free(h_x);
    free(h_y_cpu);
    free(h_y_gpu);
}

int main(int argc, char **argv)
{
    int device_id = 0;
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, device_id));
    printf("WARP Synthetic GPU Micro-Harness (tools/hipbw)\n");
    printf("Target Device: %s (Arch: %s, VRAM: %.2f GB, CUs: %d, WaveSize: %d)\n",
           prop.name, prop.gcnArchName, (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.multiProcessorCount, prop.warpSize);

    test_kernel_parity(prop.warpSize);
    return 0;
}
