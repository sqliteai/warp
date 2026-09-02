/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * hipbw.cpp — Standalone PCIe transfer and GPU memory bandwidth micro-harness
 * calibrated to WARP's verified tensor dimensions (Day 2 Registry).
 *
 * Measures:
 *   1. PCIe Host-to-Device (H2D) and Device-to-Host (D2H) transfer bandwidth.
 *   2. Empty-kernel GPU dispatch overhead / launch latency.
 *   3. Raw VRAM read/write streaming bandwidth on synthetic buffers.
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define CHECK_HIP(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* Empty kernel to isolate launch floor / dispatch overhead */
__global__ void empty_kernel(void)
{
}

/* Synthetic streaming read kernel (measures VRAM read bandwidth) */
__global__ void stream_read_kernel(const float4 * __restrict__ in, float * __restrict__ out, size_t nvec)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (size_t i = idx; i < nvec; i += gridDim.x * blockDim.x) {
        float4 v = in[i];
        sum += v.x + v.y + v.z + v.w;
    }
    if (idx == 0) *out = sum;
}

/* Benchmarks PCIe transfers and raw streaming on a given buffer size */
static void benchmark_shape(const char *name, int M, int K, int bytes_per_elem, int iterations)
{
    size_t total_bytes = (size_t)M * K * bytes_per_elem;
    double size_mb = (double)total_bytes / (1024.0 * 1024.0);

    printf("\n=== Target: %s [%d x %d] (%.2f MB) ===\n", name, M, K, size_mb);

    /* Allocate host memory */
    uint8_t *h_src = (uint8_t *)malloc(total_bytes);
    uint8_t *h_dst = (uint8_t *)malloc(total_bytes);
    if (!h_src || !h_dst) {
        fprintf(stderr, "Host memory allocation failed for %.2f MB\n", size_mb);
        return;
    }
    memset(h_src, 0x5a, total_bytes);

    /* Allocate GPU memory */
    void *d_buf = NULL;
    CHECK_HIP(hipMalloc(&d_buf, total_bytes));

    float *d_out = NULL;
    CHECK_HIP(hipMalloc(&d_out, sizeof(float)));

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    /* 1. Host-to-Device (PCIe H2D) */
    CHECK_HIP(hipEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        CHECK_HIP(hipMemcpyAsync(d_buf, h_src, total_bytes, hipMemcpyHostToDevice, 0));
    }
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms_h2d = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms_h2d, start, stop));
    double avg_ms_h2d = ms_h2d / iterations;
    double gb_sec_h2d = (size_mb / 1024.0) / (avg_ms_h2d / 1000.0);
    printf("  [PCIe H2D Transfer]   Avg: %7.3f ms  |  Throughput: %6.2f GB/s\n", avg_ms_h2d, gb_sec_h2d);

    /* 2. Device-to-Host (PCIe D2H) */
    CHECK_HIP(hipEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        CHECK_HIP(hipMemcpyAsync(h_dst, d_buf, total_bytes, hipMemcpyDeviceToHost, 0));
    }
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms_d2h = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms_d2h, start, stop));
    double avg_ms_d2h = ms_d2h / iterations;
    double gb_sec_d2h = (size_mb / 1024.0) / (avg_ms_d2h / 1000.0);
    printf("  [PCIe D2H Transfer]   Avg: %7.3f ms  |  Throughput: %6.2f GB/s\n", avg_ms_d2h, gb_sec_d2h);

    /* 3. VRAM Streaming Read Bandwidth */
    size_t nvec = total_bytes / sizeof(float4);
    int block_size = 256;
    int grid_size = 1024;
    CHECK_HIP(hipEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        stream_read_kernel<<<grid_size, block_size, 0, 0>>>((const float4 *)d_buf, d_out, nvec);
    }
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms_vram = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms_vram, start, stop));
    double avg_ms_vram = ms_vram / iterations;
    double gb_sec_vram = (size_mb / 1024.0) / (avg_ms_vram / 1000.0);
    printf("  [VRAM Read Streaming] Avg: %7.3f ms  |  Throughput: %6.2f GB/s\n", avg_ms_vram, gb_sec_vram);

    /* Cleanup */
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipFree(d_buf));
    CHECK_HIP(hipFree(d_out));
    free(h_src);
    free(h_dst);
}

static void benchmark_launch_latency(int iterations)
{
    printf("\n=== Kernel Launch Latency & Dispatch Floor ===\n");
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    /* Warm up */
    for (int i = 0; i < 50; i++) empty_kernel<<<1, 1>>>();
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        empty_kernel<<<1, 1, 0, 0>>>();
    }
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));

    float ms_total = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms_total, start, stop));
    double us_per_launch = (ms_total * 1000.0) / iterations;
    printf("  [Empty Kernel Launch Floor] %6.2f us per dispatch (%d dispatches in %.2f ms)\n",
           us_per_launch, iterations, ms_total);

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}

int main(int argc, char **argv)
{
    int device_id = 0;
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, device_id));
    printf("WARP Synthetic GPU Micro-Harness (tools/hipbw)\n");
    printf("Target Device: %s (Arch: %s, VRAM: %.2f GB, CUs: %d)\n",
           prop.name, prop.gcnArchName, (double)prop.totalGlobalMem / (1024.0*1024.0*1024.0),
           prop.multiProcessorCount);

    benchmark_launch_latency(10000);

    /* 1. LM Head: [151936 x 4096], Q8G (1 byte/elem) = ~593.5 MB */
    benchmark_shape("LM Head (Q8G)", 151936, 4096, 1, 20);

    /* 2. Self-Attention Projection: [4096 x 4096], Q8G = 16.0 MB */
    benchmark_shape("Attention Proj (Q8G)", 4096, 4096, 1, 50);

    /* 3. Attention Projection: [4096 x 4096], Q4G (0.5 byte/elem) = 8.0 MB */
    benchmark_shape("Attention Proj (Q4G)", 4096, 2048, 1, 50);

    /* 4. MoE Router Gate: [256 x 4096], Q8G = 1.0 MB */
    benchmark_shape("MoE Router Gate", 256, 4096, 1, 100);

    printf("\n=== Micro-Harness Benchmark Complete ===\n");
    return 0;
}
