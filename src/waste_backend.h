/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste_backend.h — kernel dispatch, modelled on sqlite-vector.
 *
 * sqlite-vector keeps one global table of function pointers, fills it with
 * a baseline CPU implementation that is always compiled in, then lets the
 * best backend detected at runtime overwrite the entries it implements
 * (see src/distance-cpu.c:init_distance_functions there). WASTE uses the
 * same discipline — and, like sqlite-vector, no dynamic loading at all:
 * every backend is selected by conditional compilation and enabled at
 * runtime by feature detection.
 *
 *   - SIMD (NEON, AVX2, AVX-512, SVE, RVV) is compiled in when the target
 *     architecture supports it and chosen by runtime CPU detection, so one
 *     binary adapts to the machine it lands on.
 *   - Accelerators (CUDA, Metal, BLAS) are build-time options
 *     (-DWASTE_ENABLE_CUDA=1, ...). A build without them has no link
 *     dependency on them; a build with them still probes at runtime and
 *     declines when no usable device is present.
 *
 * Invariants:
 *   - the universal CPU path is ALWAYS available and always correct; every
 *     other backend is an optimization that must produce the same result;
 *   - a backend may implement any subset of kernels — unimplemented slots
 *     keep the CPU version;
 *   - dispatch is resolved once at init, never per call in a hot loop;
 *   - WASTE_BACKEND=cpu (env) or waste_backend_init(WASTE_BE_FORCE_CPU)
 *     disables acceleration, for bisecting numeric differences.
 *
 * Because WASTE's kernels have heterogeneous signatures (unlike
 * sqlite-vector's distance functions, which all share one), the table is a
 * struct of function pointers rather than a 2-D array. Same idea.
 */

#ifndef WASTE_BACKEND_H
#define WASTE_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- capability bits --------------------------------------------------- */

typedef enum {
    WASTE_CPU_NEON     = 1u << 0,
    WASTE_CPU_DOTPROD  = 1u << 1,   /* ARM SDOT/UDOT                        */
    WASTE_CPU_I8MM     = 1u << 2,   /* ARM SMMLA                            */
    WASTE_CPU_SVE      = 1u << 3,
    WASTE_CPU_SSE2     = 1u << 8,
    WASTE_CPU_AVX2     = 1u << 9,
    WASTE_CPU_FMA      = 1u << 10,
    WASTE_CPU_AVX512F  = 1u << 11,
    WASTE_CPU_AVX512BW = 1u << 12,
    WASTE_CPU_AVX_VNNI = 1u << 13,
    WASTE_CPU_AVX512VBMI = 1u << 14,
    WASTE_CPU_RVV      = 1u << 20,
} waste_cpu_feature;

/* Detected once; safe to call repeatedly. */
uint32_t waste_cpu_features(void);

/* ---- the dispatch table ------------------------------------------------ */

typedef struct {
    /* KDA (see kda.h for semantics) */
    void (*kda_step)(int H, int K, int V,
                     const float *q, const float *k, const float *v,
                     const float *g_log, const float *beta,
                     float *S, float *o, float *scratch);
    void (*short_conv_step)(int C, int KS, const float *w, const float *bias,
                            float *ring, const float *x, float *y);
    void (*rmsnorm_gated)(int C, const float *x, const float *gate,
                          const float *weight, float eps, float *y);

    /* The range kernels that carry the arithmetic (see simd.h). They
     * take (begin, end, arg) because that is what the thread pool hands
     * out, so dispatch costs one indirect call per range, not per row. */
    void (*mvq_rows_f32)(int b, int e, void *arg);
    void (*lutb_range)(int lo, int hi, void *arg);
    void (*vq_rows_p6)(int b, int e, void *arg);

    /* Set by a backend that wants the whole row range in one call — a GPU
     * dispatch must not be split across pool threads. Call sites use
     * waste_run_rows() rather than waste_parallel_for() directly. */
    int on_device;

    /* A backend implementing only some slots leaves the rest at CPU. */
} waste_kernels;

/* The live table. Read it after waste_backend_init(); never mutate it from
 * outside a backend's registration function. */
extern waste_kernels waste_k;

/* ---- init / introspection ---------------------------------------------- */

typedef enum {
    WASTE_BE_AUTO      = 0,
    WASTE_BE_FORCE_CPU = 1u << 0,  /* baseline only — the numeric reference */
    WASTE_BE_NO_DEVICE = 1u << 1,  /* skip GPU/accelerator backends         */
} waste_backend_flags;

/* Fills the table: CPU baseline first, then the best backend compiled into
 * this build and supported by this machine. Idempotent. Honours the
 * WASTE_BACKEND environment variable ("cpu" to pin the baseline). */
void waste_backend_init(unsigned flags);

/* e.g. "CPU", "NEON+dotprod", "AVX-512", "Metal". Never NULL. */
const char *waste_backend_name(void);

/* Release backend objects that borrow model-owned host allocations before
 * a model is freed.  A no-op for backends that do not cache such objects. */
void waste_backend_release_host_buffers(void);

/* Each backend module defines a registration function with this shape:
 * it overwrites the slots it implements and returns a name, or NULL to
 * decline (compiled in, but no usable device on this machine). They are
 * called directly from waste_backend_init under #if guards — there is no
 * plugin ABI and nothing is resolved by name at runtime. */

#ifdef __cplusplus
}
#endif
#endif /* WASTE_BACKEND_H */
