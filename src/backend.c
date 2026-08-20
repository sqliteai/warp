/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * backend.c — runtime CPU detection and kernel dispatch.
 * See waste_backend.h. Structure follows sqlite-vector's
 * init_distance_functions(): baseline first, best backend overwrites.
 */

#include "waste_backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#if defined(_WIN32)
  #include <windows.h>
#endif

#if defined(__APPLE__)
  #include <sys/sysctl.h>
#elif defined(__linux__) && (defined(__aarch64__) || defined(__arm__))
  #include <sys/auxv.h>
  #include <asm/hwcap.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define WASTE_X86 1
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
  #endif
#endif

waste_kernels waste_k;

static const char *g_backend_name = "uninitialized";
static int g_initialized = 0;
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_feature_mu = PTHREAD_MUTEX_INITIALIZER;

/* Backend registration functions, linked in only when their translation
 * unit is compiled. Each returns a name, or NULL to decline at runtime. */
const char *waste_kda_register_cpu(waste_kernels *t);
#if defined(__ARM_NEON) || defined(__aarch64__)
const char *waste_kda_register_neon(waste_kernels *t);
#endif
#if defined(__x86_64__) || defined(_M_X64)
const char *waste_register_avx2(waste_kernels *t);
const char *waste_register_avx512(waste_kernels *t);
#endif
#if defined(WASTE_ENABLE_METAL)
const char *waste_register_metal(waste_kernels *t);
#endif
#if defined(WASTE_ENABLE_CUDA)
const char *waste_register_cuda(waste_kernels *t);
#endif
#if defined(WASTE_ENABLE_BLAS)
const char *waste_register_blas(waste_kernels *t);
#endif

/* ---- CPU feature detection --------------------------------------------- */

#if WASTE_X86
static void run_cpuid(unsigned leaf, unsigned sub, unsigned r[4])
{
  #if defined(_MSC_VER)
    int i[4]; __cpuidex(i, (int)leaf, (int)sub);
    r[0] = i[0]; r[1] = i[1]; r[2] = i[2]; r[3] = i[3];
  #else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
  #endif
}

static uint64_t run_xgetbv(unsigned idx)
{
  #if defined(_MSC_VER)
    return _xgetbv(idx);
  #else
    unsigned lo, hi;
    __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(lo), "=d"(hi) : "c"(idx));
    return ((uint64_t)hi << 32) | lo;
  #endif
}
#endif

#if defined(__APPLE__) && defined(__aarch64__)
static int sysctl_flag(const char *name)
{
    int v = 0; size_t sz = sizeof(v);
    return sysctlbyname(name, &v, &sz, NULL, 0) == 0 && v != 0;
}
#endif

uint32_t waste_cpu_features(void)
{
    static uint32_t cached = 0;
    static int done = 0;
    pthread_mutex_lock(&g_feature_mu);
    if (done) { pthread_mutex_unlock(&g_feature_mu); return cached; }

    uint32_t f = 0;

#if WASTE_X86
    unsigned r[4];
    run_cpuid(0, 0, r);
    const unsigned maxleaf = r[0];
    if (maxleaf >= 1) {
        run_cpuid(1, 0, r);
        if (r[3] & (1u << 26)) f |= WASTE_CPU_SSE2;
        const int osxsave = (r[2] & (1u << 27)) != 0;
        const int fma     = (r[2] & (1u << 12)) != 0;
        /* AVX state must be enabled by the OS, not just present in silicon */
        const uint64_t xcr0 = osxsave ? run_xgetbv(0) : 0;
        const int ymm_ok = osxsave && ((xcr0 & 0x6) == 0x6);
        const int zmm_ok = ymm_ok && ((xcr0 & 0xE0) == 0xE0);
        if (fma && ymm_ok) f |= WASTE_CPU_FMA;
        if (maxleaf >= 7) {
            run_cpuid(7, 0, r);
            if (ymm_ok && (r[1] & (1u << 5)))  f |= WASTE_CPU_AVX2;
            if (zmm_ok && (r[1] & (1u << 16))) f |= WASTE_CPU_AVX512F;
            if (zmm_ok && (r[1] & (1u << 30))) f |= WASTE_CPU_AVX512BW;
            if (zmm_ok && (r[2] & (1u << 1)))  f |= WASTE_CPU_AVX512VBMI;
            run_cpuid(7, 1, r);
            if (ymm_ok && (r[0] & (1u << 4)))  f |= WASTE_CPU_AVX_VNNI;
        }
    }
#elif defined(__aarch64__)
    f |= WASTE_CPU_NEON;              /* mandatory on aarch64 */
  #if defined(__APPLE__)
    if (sysctl_flag("hw.optional.arm.FEAT_DotProd")) f |= WASTE_CPU_DOTPROD;
    if (sysctl_flag("hw.optional.arm.FEAT_I8MM"))    f |= WASTE_CPU_I8MM;
  #elif defined(__linux__)
    {
        unsigned long hw = getauxval(AT_HWCAP);
    #ifdef HWCAP_ASIMDDP
        if (hw & HWCAP_ASIMDDP) f |= WASTE_CPU_DOTPROD;
    #endif
    #ifdef HWCAP_SVE
        if (hw & HWCAP_SVE) f |= WASTE_CPU_SVE;
    #endif
    #ifdef AT_HWCAP2
        {
            unsigned long hw2 = getauxval(AT_HWCAP2);
        #ifdef HWCAP2_I8MM
            if (hw2 & HWCAP2_I8MM) f |= WASTE_CPU_I8MM;
        #endif
        }
    #endif
    }
  #elif defined(_WIN32)
    /* Windows on ARM64: dot product is guaranteed on supported SoCs */
    if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE))
        f |= WASTE_CPU_DOTPROD;
  #endif
#elif defined(__ARM_NEON)
    f |= WASTE_CPU_NEON;
#endif

    cached = f; done = 1;
    pthread_mutex_unlock(&g_feature_mu);
    return f;
}

/* ---- init -------------------------------------------------------------- */

void waste_backend_init(unsigned flags)
{
    pthread_mutex_lock(&g_init_mu);
    if (g_initialized) { pthread_mutex_unlock(&g_init_mu); return; }

    /* 1. universal baseline — always present, always correct */
    g_backend_name = waste_kda_register_cpu(&waste_k);
    g_initialized = 1;

    const char *env = getenv("WASTE_BACKEND");
    if (env && strcmp(env, "cpu") == 0) flags |= WASTE_BE_FORCE_CPU;
    if (flags & WASTE_BE_FORCE_CPU) { pthread_mutex_unlock(&g_init_mu); return; }

    /* 2. best in-process SIMD backend for this machine */
    const uint32_t f = waste_cpu_features();
    (void)f;
#if defined(__ARM_NEON) || defined(__aarch64__)
    if (f & WASTE_CPU_NEON) g_backend_name = waste_kda_register_neon(&waste_k);
#endif
#if defined(__x86_64__) || defined(_M_X64)
    /* Widest first: a machine with AVX-512 also reports AVX2. FMA is
     * checked with AVX2 because the kernels use fmadd. */
    if ((f & WASTE_CPU_AVX512F) && (f & WASTE_CPU_AVX512BW))
        g_backend_name = waste_register_avx512(&waste_k);
    else if ((f & WASTE_CPU_AVX2) && (f & WASTE_CPU_FMA))
        g_backend_name = waste_register_avx2(&waste_k);
#endif
    /* AVX2 / AVX-512 / SVE / RVV modules register here as they land, in
     * best-first order, each overwriting only what it implements. */

    /* 3. accelerators: compiled in only when enabled at build time, and
     * still free to decline if this machine has no usable device. */
    if (flags & WASTE_BE_NO_DEVICE) { pthread_mutex_unlock(&g_init_mu); return; }
    const char *n = NULL;
#if defined(WASTE_ENABLE_CUDA)
    if ((n = waste_register_cuda(&waste_k))) g_backend_name = n;
#endif
#if defined(WASTE_ENABLE_METAL)
    if ((n = waste_register_metal(&waste_k))) g_backend_name = n;
#endif
#if defined(WASTE_ENABLE_BLAS)
    if ((n = waste_register_blas(&waste_k))) g_backend_name = n;
#endif
    (void)n;
    pthread_mutex_unlock(&g_init_mu);
}

const char *waste_backend_name(void)
{
    /* Always pass through the init mutex: an unsynchronised read of the
     * completion flag could observe a half-published dispatch table. */
    waste_backend_init(WASTE_BE_AUTO);
    return g_backend_name;
}

void waste_backend_release_host_buffers(void)
{
#if defined(WASTE_ENABLE_METAL)
    extern void waste_metal_release_host_buffers(void);
    waste_metal_release_host_buffers();
#endif
}
