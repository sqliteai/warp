/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * hip.hip — AMD ROCm/HIP backend for WARP.
 *
 * Implements GPU-accelerated quantized matrix-vector multiplication (mvq8/mvq4)
 * with dynamic wavefront reductions and unified dispatch table integration.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "waste_backend.h"
#include "waste_hip.h"
#include "simd.h"

static char g_hip_device_name[128] = "HIP (ROCm)";
static int g_hip_initialized = 0;
static int g_hip_device_id = 0;
static int g_wavefront_size = 32;

/* Device parameter struct matching WARP metadata */
struct HipMatVecParams {
    int in;
    int ng;
    int group;
    int bits;
    uint32_t rowbytes;
    uint32_t m;
};

/* In-register warp shuffle reduction for dynamic wavefront sizes (32 or 64) */
__device__ __forceinline__ float warp_reduce_sum(float val)
{
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}

/*
 * mvq8_hip_kernel: Quantized 8-bit MatVec (Q8G)
 *
 * Each warp/wavefront computes one output row.
 * Multiple rows are packed into each threadblock.
 */
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

    const uint32_t nvec = p.rowbytes >> 2; /* 4 int8 weights per char4 */
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

/*
 * mvq4_hip_kernel: Quantized 4-bit MatVec (Q4G)
 *
 * Unpacks 4-bit signed nibbles ((byte & 0x0f) - 8) and ((byte >> 4) - 8).
 */
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

    const uint32_t nvec = p.rowbytes >> 4; /* 16 bytes (32 nibbles) per uint4 */
    const uint32_t gshift = (p.group == 128) ? 7u : (uint32_t)(__log2f((float)p.group) + 0.5f);

    float acc = 0.0f;
    for (uint32_t j = lane_id; j < nvec; j += warpSize) {
        const uint4 packed = w_row[j];
        const uint32_t base = j << 5; /* 32 weights per uint4 */
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

/* Static temporary buffers for synchronous CPU-GPU bridge */
static void *d_x_buf = NULL;
static size_t d_x_cap = 0;
static void *d_y_buf = NULL;
static size_t d_y_cap = 0;
static void *d_w_buf = NULL;
static size_t d_w_cap = 0;
static void *d_ws_buf = NULL;
static size_t d_ws_cap = 0;

static void ensure_scratch(size_t x_bytes, size_t y_bytes, size_t w_bytes, size_t ws_bytes)
{
    if (x_bytes > d_x_cap) {
        if (d_x_buf) (void)hipFree(d_x_buf);
        (void)hipMalloc(&d_x_buf, x_bytes);
        d_x_cap = x_bytes;
    }
    if (y_bytes > d_y_cap) {
        if (d_y_buf) (void)hipFree(d_y_buf);
        (void)hipMalloc(&d_y_buf, y_bytes);
        d_y_cap = y_bytes;
    }
    if (w_bytes > d_w_cap) {
        if (d_w_buf) (void)hipFree(d_w_buf);
        (void)hipMalloc(&d_w_buf, w_bytes);
        d_w_cap = w_bytes;
    }
    if (ws_bytes > d_ws_cap) {
        if (d_ws_buf) (void)hipFree(d_ws_buf);
        (void)hipMalloc(&d_ws_buf, ws_bytes);
        d_ws_cap = ws_bytes;
    }
}

/*
 * Dispatch entry point registered in waste_k.mvq_rows_f32
 */
static void hip_mvq_rows_f32(int b, int e, void *arg)
{
    mvq_arg *a = (mvq_arg *)arg;
    uint32_t m = (uint32_t)(e - b);
    if (m == 0) return;

    HipMatVecParams p;
    p.in = a->in;
    p.ng = a->ng;
    p.group = a->group;
    p.bits = a->bits;
    p.rowbytes = (uint32_t)a->rowbytes;
    p.m = m;

    size_t x_bytes = (size_t)a->in * sizeof(float);
    size_t y_bytes = (size_t)m * sizeof(float);
    size_t w_bytes = (size_t)m * a->rowbytes;
    size_t ws_bytes = (size_t)m * a->ng * sizeof(uint16_t);

    ensure_scratch(x_bytes, y_bytes, w_bytes, ws_bytes);

    /* Transfer inputs */
    (void)hipMemcpyAsync(d_x_buf, a->xq ? (const void *)a->xq : (const void *)a->xs, x_bytes, hipMemcpyHostToDevice, 0);
    (void)hipMemcpyAsync(d_w_buf, a->W + (size_t)b * a->rowbytes, w_bytes, hipMemcpyHostToDevice, 0);
    (void)hipMemcpyAsync(d_ws_buf, a->ws + (size_t)b * a->ng, ws_bytes, hipMemcpyHostToDevice, 0);

    int block_dim = 256;
    int warps_per_block = block_dim / g_wavefront_size;
    int grid_dim = (m + warps_per_block - 1) / warps_per_block;

    if (a->bits == 8) {
        hipLaunchKernelGGL(mvq8_hip_kernel, dim3(grid_dim), dim3(block_dim), 0, 0,
                           (float *)d_y_buf, (const int8_t *)d_w_buf, (const uint16_t *)d_ws_buf,
                           (const float *)d_x_buf, p);
    } else if (a->bits == 4) {
        hipLaunchKernelGGL(mvq4_hip_kernel, dim3(grid_dim), dim3(block_dim), 0, 0,
                           (float *)d_y_buf, (const uint8_t *)d_w_buf, (const uint16_t *)d_ws_buf,
                           (const float *)d_x_buf, p);
    }

    /* Retrieve results */
    (void)hipMemcpy(a->y + b, d_y_buf, y_bytes, hipMemcpyDeviceToHost);
}

const char *waste_register_hip(waste_kernels *t)
{
    if (g_hip_initialized) {
        return g_hip_device_name;
    }

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count <= 0) {
        return NULL;
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, g_hip_device_id);
    if (err != hipSuccess) {
        return NULL;
    }

    g_wavefront_size = prop.warpSize;
    snprintf(g_hip_device_name, sizeof(g_hip_device_name), "HIP (%s - %s)",
             prop.name, prop.gcnArchName);

    /* Register GPU kernels and configure dispatch thresholds */
    t->mvq_rows_f32 = hip_mvq_rows_f32;
    t->on_device = 1;
    t->device_min_bytes = 1024 * 1024; /* Dispatches >= 1 MB go to GPU */

    g_hip_initialized = 1;
    return g_hip_device_name;
}
