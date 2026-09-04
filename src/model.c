/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * model.c — container loading + forward pass. See model.h.
 *
 * One token per call: prefill is just repeated steps, which keeps the
 * decode path (the one that matters for a streaming engine) as the only
 * path, and makes the KDA/conv/KV state handling uniform.
 */

#define _GNU_SOURCE
#include "model.h"

#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>          /* MinGW ships one too: open/close/lseek */

#include "crc32.h"
#include "ecache.h"
#include "json.h"
#include "platform.h"
#include "threads.h"
#include "kda.h"
#include "qwen_gdn.h"
#include "qwen_hc.h"
#include "qwen_moe.h"
#include "qwen_ple.h"
#include "qwen_qsa.h"
#include "simd.h"
#include "waste_backend.h"
#include "waste_metal.h"
#include "waste_format.h"

#define MAXP 512

/* ---- lightweight phase profiling (WASTE_PROFILE=1) --------------------- */
#include <time.h>
double waste_prof[16];
uint64_t waste_prof_n[16];
uint64_t waste_tmv_bytes;
int *waste_route_cap; int waste_route_n, waste_route_cap_n;
/* WASTE_TRUNK_CHECK=1: run the f32 reference beside whichever quantized
 * trunk kernel is selected and accumulate the relative error on real
 * activations. Two kernels can agree with the reference to 4e-5 on one
 * model and disagree with each other by 0.13 of a logit norm on another
 * (docs/EXP1.md §2c), so the only way to rank them is to measure them
 * where they run, on the tensors they run on. */
static int trunk_check = 0;
double waste_tcheck_num, waste_tcheck_den, waste_tcheck_max;
unsigned long long waste_tcheck_n;
/* matvec_t by call size: [<1MB, <8MB, <32MB, rest] */
double waste_tmv_t[4];
uint64_t waste_tmv_b[4], waste_tmv_c[4];
enum { P_LUTB, P_KDA, P_MLA, P_ROUTE, P_EDEQ, P_EMM, P_HEAD, P_LUTA, P_MM,
       P_TMV, P_KDAK };
static int prof_on = -1;
static pthread_mutex_t prof_mu = PTHREAD_MUTEX_INITIALIZER;
static double pnow(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
#define PROF_START(b) double _t##b = prof_on ? pnow() : 0
#define PROF_END(b)   do { if (prof_on) { pthread_mutex_lock(&prof_mu); \
    waste_prof[b] += pnow() - _t##b; waste_prof_n[b]++; \
    pthread_mutex_unlock(&prof_mu); } } while (0)

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    /* ftell says -1 on a directory or a pipe, and (size_t)(-1) + 1 is a
     * zero-byte malloc that the fread below then writes past. */
    if (n < 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    b[n] = 0;
    fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

static int bank_open(const char *path, size_t rec_bytes, int want, int *direct);

/* pread until the whole range lands; short reads are legal.
 *
 * The offset is int64_t and not `long` because `long` is 32 bits on
 * Windows: a trunk is 57 GB on K3, so a truncating offset would read the
 * wrong tensor rather than fail. */
static int pread_all(int fd, void *dst, size_t n, int64_t off)
{
    uint8_t *p = (uint8_t *)dst;
    while (n) {
        const int64_t r = waste_pread(fd, p, n, off);
        if (r <= 0) return -1;
        p += r; n -= (size_t)r; off += r;
    }
    return 0;
}

const waste_tensor *waste_find(const waste_model *m, const char *name)
{
    for (int i = 0; i < m->n_tensors; i++)
        if (strcmp(m->t[i].name, name) == 0) return &m->t[i];
    return NULL;
}

/* Formats a tensor name. Rotates over several buffers because callers pass
 * two or three of these to the same function, and C does not order
 * argument evaluation — a single static buffer would make them all alias. */
__attribute__((format(printf, 1, 2)))
static const char *tname(const char *fmt, ...)
{
    /* Distinct contexts may validate/execute concurrently.  The rotation
     * solves same-expression aliasing; thread-local storage solves the
     * otherwise-racy process-wide cursor and buffers. */
    static _Thread_local char buf[8][160];
    static _Thread_local unsigned turn;
    char *b = buf[turn++ & 7];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(b, 160, fmt, ap);
    va_end(ap);
    return b;
}

__attribute__((format(printf, 2, 3)))
static const float *T(const waste_model *m, const char *fmt, ...)
{
    char buf[160];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    const waste_tensor *t = waste_find(m, buf);
    return t ? t->data : NULL;
}

#define f16_to_f32 waste_f16

/* ---- kernels used only here (dispatchable later) ----------------------- */

void waste_rmsnorm(float *o, const float *x, const float *w, int n, float eps)
{
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(s / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
static inline float dotf(const float *a, const float *b, int n)
{
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(s0, s1));
    for (; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#else
static inline float dotf(const float *a, const float *b, int n)
{
    float acc = 0;
    for (int i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}
#endif

static int q8_off = 1;     /* 1 = keep the trunk stored as int8          */
static int sdot_on = 0;    /* 1 = also quantize activations (SDOT path)  */
/* Which kernel the Q4G trunk matvec uses. The trunk is 28.0 GB of Q4G on
 * K3 and every byte is read once per token, so this one choice is ~46% of
 * a decode step (docs/EXP1.md §1). See model_opts_init for what each mode
 * costs in accuracy. */
enum { TK_F32 = 0, TK_SDOT = 1, TK_I8MM = 2, TK_SMLAL = 3 };
static int trunk_kern = TK_F32;   /* WASTE_TRUNK_KERNEL                   */
static int sdot4_sg = 32;  /* TK_SDOT only: activations per int8 scale    */
static int i8mm_on = 0;    /* SMMLA batched matmul; costs activation int8 */
static const char *dump_route = NULL;  /* WASTE_DUMP_ROUTE, see moe_layer */
static const char *dump_dsa = NULL;    /* WASTE_DUMP_DSA, see dsa_select   */
static const char *dump_scores = NULL; /* WASTE_DUMP_SCORES, see moe_layer */
static float ccr_lambda = 0.0f;        /* WASTE_CCR_LAMBDA, see moe_layer */
/* Absolute position of the first token of the pass being routed. The dump
 * names each row by the token it belongs to rather than leaving a reader
 * to infer it from where the layer index wraps — which is a heuristic
 * that has to be rewritten correctly in every script that reads one. */
static int dump_pos0 = 0;
static int lookahead_n = 0;            /* WASTE_LOOKAHEAD, see moe_layer  */
static int p6_chunk = 4;               /* WASTE_P6_CHUNK, see vq_apply    */
/* -1 = decide per layer from what is already in the cache, 0 = never,
 * 1 = always. WASTE_XPAR sets the last two; the default is the first, and
 * moe_layer explains what it decides on. */
static int xpar_on = -1;
static int metal_moe = 0;              /* WASTE_METAL_MOE, see moe_layer    */
static int vq8_on = 0;                 /* WASTE_VQ8: int8 VQ3R table        */
/* Which kernels are cut for the performance cores rather than the whole
 * pool. A bitmask because LEARNED §47's finding is per kernel, not per
 * machine: bit 0 the VQ apply, bit 1 the quantized trunk matvec, bit 2 the
 * LUT build. 0 is the pool for everything, which is what every number in
 * this repo before now was measured on. */
enum { WIDE_VQ = 1, WIDE_TRUNK = 2, WIDE_LUTB = 4 };
/* On by default, which LEARNED §47 explicitly declined to do — and the
 * difference is what "on" means. §47 measured *capping the pool* at the
 * performance-core count: 25% on one model and -34% on the other, because
 * the kernel that wants every core loses them all. This takes cores away
 * from nothing. The VQ apply and the LUT build are cut for the fast group
 * and everything else still gets the whole machine, so the trunk matvec is
 * unaffected and only the two kernels whose barrier was waiting on an
 * efficiency core change. Measured positive on both models, and the split
 * is by row so the logits are bit-identical either way (checked). The
 * trunk matvec is deliberately *not* in the default: it measures neutral
 * on both, so it is a switch and not a choice. WASTE_WIDE=0 turns it off. */
static int wide_mask = WIDE_VQ | WIDE_LUTB;   /* WASTE_WIDE */

/* One call site's dispatch: the fast group when this kernel is in the
 * mask, the whole pool otherwise. */
static inline void pf_wide(int bit, int n, int min_chunk, waste_range_fn fn,
                           void *arg)
{
    if (wide_mask & bit) waste_parallel_for_fast(n, min_chunk, fn, arg);
    else waste_parallel_for(n, min_chunk, fn, arg);
}
static int xpar_batch = 4;             /* WASTE_XPAR_BATCH, see moe_layer  */
static pthread_once_t model_opts_once = PTHREAD_ONCE_INIT;

static void model_opts_init(void)
{
    const char *e = getenv("WASTE_PROFILE");
    prof_on = e && *e != '0';
    e = getenv("WASTE_Q8");
    if (e && *e == '0') q8_off = 0;
    e = getenv("WASTE_SDOT");
    sdot_on = e && *e != '0';
    /* The 4-bit trunk through SDOT. K3's trunk is 28.0 GB of Q4G and every
     * byte is read once per token, so this kernel is ~40% of a decode step.
     * The f32 path reaches 60 GB/s on this machine against a 217 GB/s
     * streaming ceiling (tools/mvqbw.c) because a 4-bit weight has to
     * become a float before it can be multiplied; quantizing the
     * activations too puts it at 202 GB/s, i.e. at the machine. What it
     * costs is that the arithmetic is no longer the f32 reference's —
     * hence a switch, and hence WASTE_SDOT4_SG, which sets how many
     * activations share one int8 scale (32 by default; the weights' own
     * group is 128). */
    /* WASTE_TRUNK_KERNEL: 0 f32 (the reference), 1 SDOT, 2 i8mm, 3 SMLAL.
     *
     * The f32 path reaches 65 GB/s on this machine against a 188 GB/s
     * streaming ceiling (tools/mvqbw.c) because a 4-bit weight has to
     * become a float before it can be multiplied. Quantizing the
     * activations too removes that, and the three ways of doing it are a
     * straight speed-for-accuracy line:
     *
     *   1 SDOT   int8 activations       173 GB/s   max|d| 0.695
     *   2 i8mm   15-bit, two planes     144 GB/s   max|d| 0.016
     *   3 SMLAL  int16 activations      113 GB/s   max|d| 0.005
     *   0 f32    exact                   78 GB/s   0
     *
     * SDOT is fastest and is not payable: on K3 it measures KL 0.289
     * teacher-forced over twelve positions, against §56's KL 0.118 for a
     * truncation that repo calls broken. The error is small at one
     * position (rel L2 0.03) and the KDA recurrence accumulates it — which
     * is also why the same kernel measures KL 0.0013 on Kimi-Linear's 27
     * layers. i8mm buys 43x the accuracy for 83% of the speed. */
    e = getenv("WASTE_TRUNK_KERNEL");
    trunk_kern = e ? atoi(e) : TK_F32;
    if (trunk_kern < 0 || trunk_kern > TK_SMLAL) trunk_kern = TK_F32;
    if ((trunk_kern == TK_SDOT || trunk_kern == TK_I8MM) &&
        !(waste_cpu_features() & WASTE_CPU_DOTPROD)) trunk_kern = TK_F32;
    if (trunk_kern == TK_I8MM && !(waste_cpu_features() & WASTE_CPU_I8MM))
        trunk_kern = TK_SMLAL;
    e = getenv("WASTE_TRUNK_CHECK");
    trunk_check = e && *e != '0';
    e = getenv("WASTE_SDOT4_SG");
    sdot4_sg = e ? atoi(e) : 32;
    if (sdot4_sg != 32 && sdot4_sg != 64 && sdot4_sg != 128) sdot4_sg = 32;
    /* Here with the rest of them: read per-load, this was the one env
     * switch still written by every concurrent waste_model_load.
     * waste_cpu_features() is self-caching and does not need the backend. */
    e = getenv("WASTE_I8MM");
    i8mm_on = e ? (*e != '0')
                : 0;     /* off by default until it earns it — see below */
    if (i8mm_on && !(waste_cpu_features() & WASTE_CPU_I8MM)) i8mm_on = 0;
    /* Read once rather than per layer per token: moe_layer runs 92
     * times a token and getenv is not free. */
    dump_dsa = getenv("WASTE_DUMP_DSA");
    if (dump_dsa && !*dump_dsa) dump_dsa = NULL;
    dump_route = getenv("WASTE_DUMP_ROUTE");
    dump_scores = getenv("WASTE_DUMP_SCORES");
    { const char *s = getenv("WASTE_CCR_LAMBDA");
      ccr_lambda = s ? (float)atof(s) : 0.0f;
      if (!(ccr_lambda > 0.0f)) ccr_lambda = 0.0f; }   /* also catches NaN */
    /* How many of the next layer's experts to fetch on the router's guess.
     * The layer boundary holds about six reads and the prediction's
     * precision falls off past there, so that is the default. 0 is off. */
    /* Rows per parallel_for chunk in the VQ4P apply, in whole index
     * blocks. The kernel is several times faster than the VQ3R gather it
     * replaces, so the same chunk size hands the pool work that finishes
     * before the dispatch that scheduled it: at Kimi-Linear's shapes one
     * apply is ~3us of arithmetic spread over ten threads. Bigger chunks,
     * fewer of them. */
    /* One task per routed expert instead of one per row range.
     *
     * Off by default, and that is a measurement rather than caution: it is
     * worth ~1.18x on Kimi-Linear, where one apply is microseconds and the
     * step is ~900 fork-joins, and it is a regression on K3, where the
     * batch that gives it its parallelism is the same batch that barriers
     * the read-ahead. No batch size wins both — docs/LEARNED.md §44 has
     * the sweep. Which regime a run is in depends on the model and the
     * machine, not on anything the container states, so it is a switch. */
    /* The routed experts' VQ applies as one device batch per layer phase.
     * One apply is too small to fill this GPU — 3072 rows measures 12 GB/s
     * of index against the CPU's 24.5 — and a layer's sixteen of them in
     * one concurrent command buffer measure 126. So the switch is not
     * "use the GPU", it is "batch the layer", and it costs the barrier
     * docs/LEARNED.md §44 describes: every record held before any
     * arithmetic starts. Off by default until that trade is measured on
     * more than one machine. */
    /* The VQ3R apply through an int8 table held in registers rather than
     * a float one walked in memory. 1.76x on the kernel single-threaded
     * (tools/lutbw.c kernel E). Off by default: it quantizes the table,
     * which makes the path discontinuous in §43's sense. */
    { const char *e2 = getenv("WASTE_WIDE");
      if (e2) wide_mask = atoi(e2);
      if (wide_mask < 0) wide_mask = 0; }
    { const char *e2 = getenv("WASTE_VQ8");
      vq8_on = e2 && *e2 != '0'; }
    { const char *e2 = getenv("WASTE_METAL_MOE");
      metal_moe = e2 && *e2 != '0'; }
    { const char *e2 = getenv("WASTE_XPAR");
      xpar_on = e2 ? (*e2 != '0') : -1; }
    /* Experts held — and so barriered — at a time. Small keeps the reads
     * overlapping the arithmetic; large gives the pool more to chew on. */
    { const char *e2 = getenv("WASTE_XPAR_BATCH");
      xpar_batch = e2 ? atoi(e2) : 4;
      if (xpar_batch < 1) xpar_batch = 1;
      if (xpar_batch > WASTE_PF_MAX) xpar_batch = WASTE_PF_MAX; }
    { const char *e2 = getenv("WASTE_P6_CHUNK");
      p6_chunk = e2 ? atoi(e2) : 16;
      if (p6_chunk < 1) p6_chunk = 1;
      if (p6_chunk > 64) p6_chunk = 64; }
    { const char *e2 = getenv("WASTE_LOOKAHEAD");
      lookahead_n = e2 ? atoi(e2) : 6;
      if (lookahead_n < 0) lookahead_n = 0;
      if (lookahead_n > 64) lookahead_n = 64; }
}

/* One weight from a 3-bit stream: values sit LSB-first at bit offset 3*i,
 * biased by +4, with a guard byte so two loads are always safe. Identical
 * indexing to tools/convert.py's quantize_q3g. */
#define q3_at waste_q3_at

/* ---- int8 x int8 matvec (SDOT / dotprod) --------------------------------
 * The trunk is already stored Q8G: int8 weights with one fp16 scale per
 * group of 128 inputs. Keeping it that way (instead of expanding to f32 at
 * load) saves ~6 GB of RAM and lets the dot run on ARM SDOT / x86 VNNI.
 * Activations are quantized per group with the same geometry, so a group
 * contributes scale_w * scale_x * <int32 dot>.
 */

static inline int32_t idot(const int8_t *a, const int8_t *b, int n)
{
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(acc);
    for (; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)a[i] * b[i];
    return s;
#endif
}

/* int8 weights, f32 activations: dequantize the row inline. Keeps the
 * 4x memory saving of int8 storage while producing exactly the numbers the
 * f32 path does — no activation quantization error. */
void waste_mvq_rows_f32(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    const float *x = (const float *)a->xs;      /* raw activations */
    int8_t *unp = NULL;
    if (a->bits != 8) {
        unp = (int8_t *)malloc((size_t)g);
        if (!unp) return;
    }
    (void)unp;
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
#if defined(__ARM_NEON) || defined(__aarch64__)
            /* Q4G, whole group: unpack in registers instead of staging the
             * nibbles through a byte buffer. Same FMAs into the same 4-lane
             * accumulator in the same order, so the result is bit for bit
             * what the staged path produces — measured with tools/mvqbw.c,
             * where dropping the staging buffer is worth 1.24x on K3's
             * [12288 x 7168] trunk shape at 18 threads. The staged path
             * stays for 3-bit rows, for a group that is not a multiple of
             * 32, and for the ragged last group. */
            if (a->bits == 4 && (k + 1) * g <= a->in && (g & 31) == 0) {
                const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
                const float *xx = x + (size_t)k * g;
                const uint8x16_t m0f = vdupq_n_u8(0x0f);
                const int8x16_t  m8  = vdupq_n_s8(8);
                float32x4_t s0 = vdupq_n_f32(0);
                for (int j = 0; j < g / 2; j += 16) {
                    const uint8x16_t by = vld1q_u8(p4 + j);
                    const int8x16_t lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                    const int8x16_t hi =
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                    const int8x16x2_t z = vzipq_s8(lo, hi);
                    const float *xp = xx + 2 * j;
                    for (int h = 0; h < 2; h++) {
                        const int16x8_t l16 = vmovl_s8(vget_low_s8(z.val[h]));
                        const int16x8_t h16 = vmovl_s8(vget_high_s8(z.val[h]));
                        s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l16))),
                                       vld1q_f32(xp + 16 * h + 0));
                        s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l16))),
                                       vld1q_f32(xp + 16 * h + 4));
                        s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h16))),
                                       vld1q_f32(xp + 16 * h + 8));
                        s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h16))),
                                       vld1q_f32(xp + 16 * h + 12));
                    }
                }
                acc += f16_to_f32(ws[k]) * vaddvq_f32(s0);
                continue;
            }
#endif
            const int8_t *w;
            if (a->bits == 3) {
                const uint8_t *p3 = (const uint8_t *)row;
                for (int i = 0; i < g; i++) unp[i] = (int8_t)q3_at(p3, (long)k * g + i);
                w = unp;
            } else if (a->bits == 4) {
                /* two signed nibbles per byte, low first */
                const uint8_t *p4 = (const uint8_t *)row + (size_t)k * g / 2;
                for (int i = 0; i < g / 2; i++) {
                    const uint8_t byte = p4[i];
                    unp[2 * i]     = (int8_t)(byte & 0x0F) - 8;
                    unp[2 * i + 1] = (int8_t)(byte >> 4) - 8;
                }
                w = unp;
            } else {
                w = row + (size_t)k * g;
            }
            const float *xx = x + (size_t)k * g;
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
#if defined(__ARM_NEON) || defined(__aarch64__)
            float32x4_t s0 = vdupq_n_f32(0);
            int i = 0;
            for (; i + 8 <= lim; i += 8) {
                const int16x8_t w16 = vmovl_s8(vld1_s8(w + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))),
                               vld1q_f32(xx + i));
                s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))),
                               vld1q_f32(xx + i + 4));
            }
            float part = vaddvq_f32(s0);
            for (; i < lim; i++) part += (float)w[i] * xx[i];
#else
            float part = 0;
            for (int i = 0; i < lim; i++) part += (float)w[i] * xx[i];
#endif
            acc += f16_to_f32(ws[k]) * part;
        }
        a->y[o] = acc;
    }
    free(unp);
}

/* ---- Q4G x int8 activations (SDOT) --------------------------------------
 * Two nibbles per byte, low first, so the even and odd elements of a group
 * face different halves of the byte. Rather than interleaving the weights
 * back together, quant_act4 writes the activations deinterleaved — the
 * group's even elements first, then its odd ones — so each half of the
 * unpacked byte meets a contiguous int8 vector and the kernel is two
 * vdotq_s32 per sixteen bytes with nothing in between.
 *
 * `sg` activations share one scale. The weights' group of 128 sets how
 * many weight scales there are and cannot change; the activation scale is
 * ours to choose, and a hidden state has outliers, so a finer one is the
 * cheap half of the accuracy. It costs one horizontal add and one fmul per
 * sub-group.
 */
static void mvq4_rows_sdot(int b, int e, void *p)
{
#if defined(__ARM_FEATURE_DOTPROD)
    const mvq4_arg *a = (const mvq4_arg *)p;
    const int ng = a->ng, g = a->group, ns = a->ns, half = a->sg / 2;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    for (int o = b; o < e; o++) {
        const uint8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *p4 = row + (size_t)k * g / 2;
            const int8_t *xe = a->xq + (size_t)k * g;
            const int8_t *xo = xe + g / 2;
            const float wsf = f16_to_f32(ws[k]);
            const float *xsc = a->xs + (size_t)k * ns;
            for (int t = 0; t < ns; t++) {
                int32x4_t d = vdupq_n_s32(0);
                const int j0 = t * half;
                for (int j = j0; j < j0 + half; j += 16) {
                    const uint8x16_t by = vld1q_u8(p4 + j);
                    const int8x16_t lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                    const int8x16_t hi =
                        vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                    d = vdotq_s32(d, lo, vld1q_s8(xe + j));
                    d = vdotq_s32(d, hi, vld1q_s8(xo + j));
                }
                acc += wsf * xsc[t] * (float)vaddvq_s32(d);
            }
        }
        a->y[o] = acc;
    }
#else
    (void)b; (void)e; (void)p;
#endif
}

/* The i8mm kernel is its own translation unit: FEAT_I8MM is not in the
 * baseline the portable ARM build targets, so arm_neon.h hides vmmlaq_s32
 * unless the file asks for it. src/simd_i8mm.c does, the Makefile gives
 * only that file the flag, and this call is guarded by the runtime bit —
 * the same arrangement simd_avx2.c has on x86, and for the same reason.
 * See docs/EXP1.md §2 for what it buys: 144 GB/s against the f32 path's
 * 78 and 43x SDOT's accuracy.
 *
 * The guard is the link, not the ISA. src/simd_i8mm.c already defines this
 * for every ARM build — a real SMMLA kernel where the flag took, a zeroing
 * refusal where it did not — but the Makefile adds that file to SRC only
 * for `arm%|aarch64%`, so on x86 and on Windows the symbol does not exist
 * at all. Declaring and calling it unguarded there is an undefined
 * reference at link time, which is how 0.7.0 shipped unable to build on
 * three of five CI platforms. The Makefile's own comment records the same
 * failure once before, from the other direction: a findstring that left
 * kda_neon.c out while backend.c still called into it. A dispatcher and
 * the list of files that satisfies it have to be guarded on the same
 * predicate, and this is that predicate. */
#if defined(__ARM_NEON) || defined(__aarch64__)
void waste_mvq4_rows_i8mm(int b, int e, void *p);
#endif

/* ---- Q4G x int16 activations (SMLAL) ------------------------------------
 * No i8mm needed and 15 bits of activation exactly as SMMLA gets, at 113
 * GB/s instead of 144. It is here as the portable member of the family and
 * as the fallback when FEAT_I8MM is absent. Activations are deinterleaved
 * even|odd inside the group, as for SDOT. */
static void mvq4_rows_smlal(int b, int e, void *p)
{
#if defined(__ARM_NEON) || defined(__aarch64__)
    const mvq4_arg *a = (const mvq4_arg *)p;
    const int ng = a->ng, g = a->group;
    const int16_t *x16 = (const int16_t *)a->xq;
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    const int8x16_t  m8  = vdupq_n_s8(8);
    for (int o = b; o < e; o++) {
        const uint8_t *row = a->W + (size_t)o * a->rowbytes;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++) {
            const uint8_t *p4 = row + (size_t)k * g / 2;
            const int16_t *xe = x16 + (size_t)k * g;
            const int16_t *xo = xe + g / 2;
            int32x4_t d0 = vdupq_n_s32(0), d1 = vdupq_n_s32(0);
            for (int j = 0; j < g / 2; j += 16) {
                const uint8x16_t by = vld1q_u8(p4 + j);
                const int8x16_t lo =
                    vsubq_s8(vreinterpretq_s8_u8(vandq_u8(by, m0f)), m8);
                const int8x16_t hi =
                    vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(by, 4)), m8);
                const int16x8_t l0 = vmovl_s8(vget_low_s8(lo)),
                                l1 = vmovl_s8(vget_high_s8(lo));
                const int16x8_t h0 = vmovl_s8(vget_low_s8(hi)),
                                h1 = vmovl_s8(vget_high_s8(hi));
                d0 = vmlal_s16(d0, vget_low_s16(l0),  vld1_s16(xe + j + 0));
                d1 = vmlal_s16(d1, vget_high_s16(l0), vld1_s16(xe + j + 4));
                d0 = vmlal_s16(d0, vget_low_s16(l1),  vld1_s16(xe + j + 8));
                d1 = vmlal_s16(d1, vget_high_s16(l1), vld1_s16(xe + j + 12));
                d0 = vmlal_s16(d0, vget_low_s16(h0),  vld1_s16(xo + j + 0));
                d1 = vmlal_s16(d1, vget_high_s16(h0), vld1_s16(xo + j + 4));
                d0 = vmlal_s16(d0, vget_low_s16(h1),  vld1_s16(xo + j + 8));
                d1 = vmlal_s16(d1, vget_high_s16(h1), vld1_s16(xo + j + 12));
            }
            acc += f16_to_f32(ws[k]) * a->xs[k] *
                   (float)(vaddvq_s32(d0) + vaddvq_s32(d1));
        }
        a->y[o] = acc;
    }
#else
    (void)b; (void)e; (void)p;
#endif
}

/* Activations for mvq4_rows_i8mm: one amax per weight group, a base-128
 * split into two int8 planes, laid out the way the B operand is read.
 * Inside the same guard as its only caller, or -Wunused-function fires on
 * every build that cannot reach it. */
#if defined(__ARM_NEON) || defined(__aarch64__)
static void quant_act4_mm(const float *x, int n, int g, int8_t *q, float *sc)
{
    const int ng = (n + g - 1) / g;
    for (int k = 0; k < ng; k++) {
        const int beg = k * g, end = (beg + g < n) ? beg + g : n;
        float amax = 0;
        for (int i = beg; i < end; i++) {
            const float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }
        const float s = amax > 0 ? amax / 16383.0f : 1.0f;
        sc[k] = s;
        const float inv = 1.0f / s;
        int8_t *base = q + (size_t)k * 2 * g;
        memset(base, 0, (size_t)2 * g);
        for (int i = beg; i < beg + g; i++) {
            int v = 0;
            if (i < end) {
                v = (int)lrintf(x[i] * inv);
                v = v > 16383 ? 16383 : (v < -16383 ? -16383 : v);
            }
            int hi = (v + 64) >> 7;
            if (hi > 127) hi = 127; else if (hi < -128) hi = -128;
            const int lo = v - 128 * hi;
            const int r = i - beg, h = r >> 1;
            int8_t *pl = base + ((r & 1) ? g : 0);
            pl[(h >> 3) * 16 + (h & 7)]     = (int8_t)hi;
            pl[(h >> 3) * 16 + 8 + (h & 7)] = (int8_t)lo;
        }
    }
}

#endif

/* Activations for mvq4_rows_smlal: int16, one amax per weight group,
 * deinterleaved even|odd inside the group. */
static void quant_act4_16(const float *x, int n, int g, int8_t *q, float *sc)
{
    const int ng = (n + g - 1) / g;
    int16_t *x16 = (int16_t *)q;
    for (int k = 0; k < ng; k++) {
        const int beg = k * g, end = (beg + g < n) ? beg + g : n;
        float amax = 0;
        for (int i = beg; i < end; i++) {
            const float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }
        const float s = amax > 0 ? amax / 32767.0f : 1.0f;
        sc[k] = s;
        const float inv = 1.0f / s;
        for (int i = beg; i < beg + g; i++) {
            int v = 0;
            if (i < end) {
                v = (int)lrintf(x[i] * inv);
                v = v > 32767 ? 32767 : (v < -32767 ? -32767 : v);
            }
            const int r = i - beg;
            x16[(size_t)k * g + (r >> 1) + ((r & 1) ? g / 2 : 0)] = (int16_t)v;
        }
    }
}

/* int8 activations for the kernel above: per sub-group amax, deinterleaved
 * within each weight group, zero-padded past `n` so a ragged last group
 * contributes nothing. */
static void quant_act4(const float *x, int n, int g, int sg, int8_t *q, float *sc)
{
    const int ng = (n + g - 1) / g, ns = g / sg;
    for (int k = 0; k < ng; k++) {
        for (int t = 0; t < ns; t++) {
            const int beg = k * g + t * sg;
            const int end = (beg + sg < n) ? beg + sg : (beg < n ? n : beg);
            float amax = 0;
            for (int i = beg; i < end; i++) {
                const float v = fabsf(x[i]);
                if (v > amax) amax = v;
            }
            const float s = amax > 0 ? amax / 127.0f : 1.0f;
            sc[k * ns + t] = s;
            const float inv = 1.0f / s;
            for (int i = beg; i < beg + sg; i++) {
                int v = 0;
                if (i < end) {
                    v = (int)lrintf(x[i] * inv);
                    v = v > 127 ? 127 : (v < -127 ? -127 : v);
                }
                const int r = i - k * g;
                q[k * g + (r >> 1) + ((r & 1) ? g / 2 : 0)] = (int8_t)v;
            }
        }
    }
}

static void mvq_rows(int b, int e, void *p)
{
    mvq_arg *a = (mvq_arg *)p;
    const int ng = a->ng, g = a->group;
    for (int o = b; o < e; o++) {
        const int8_t *row = a->W + (size_t)o * ng * g;
        const uint16_t *ws = a->ws + (size_t)o * ng;
        float acc = 0;
        for (int k = 0; k < ng; k++)
            acc += f16_to_f32(ws[k]) * a->xs[k] *
                   (float)idot(row + (size_t)k * g, a->xq + (size_t)k * g, g);
        a->y[o] = acc;
    }
}

typedef struct { float *y; const float *W, *x; int in; } mv_arg;

static void mv_rows(int b, int e, void *p)
{
    mv_arg *a = (mv_arg *)p;
    for (int o = b; o < e; o++)
        a->y[o] = dotf(a->W + (size_t)o * a->in, a->x, a->in);
}

/* y[out] = W[out][in] . x[in]; row split, so results do not depend on
 * the thread count. */
static void matvec(float *y, const float *W, const float *x, int out, int in)
{
    mv_arg a = { y, W, x, in };
    waste_parallel_for(out, 64, mv_rows, &a);
}

/* A device backend takes the whole range at once; the pool would otherwise
 * split one GPU dispatch into a dozen. Below `device_min_bytes` the
 * dispatch floor is larger than the work, so the pool keeps it — with the
 * CPU kernel, which is a different function pointer since the accelerator
 * has overwritten the one the pool would otherwise have called. */
static inline void run_rows(int n, int min_chunk, waste_range_fn fn, void *arg,
                            size_t bytes)
{
    if (waste_k.on_device && bytes >= waste_k.device_min_bytes) { fn(0, n, arg); return; }
    waste_parallel_for_work(n, min_chunk,
                            waste_k.on_device ? waste_k.mvq_rows_cpu : fn, arg,
                            bytes);
}

/* Quantize x into per-group int8 (same grouping as the weights). */
static void quant_act(const float *x, int n, int g, int8_t *q, float *sc)
{
    const int ng = (n + g - 1) / g;
    for (int k = 0; k < ng; k++) {
        const int beg = k * g, end = (beg + g < n) ? beg + g : n;
        float amax = 0;
        for (int i = beg; i < end; i++) {
            const float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }
        const float s = amax > 0 ? amax / 127.0f : 1.0f;
        sc[k] = s;
        const float inv = 1.0f / s;
        for (int i = beg; i < end; i++) {
            int v = (int)lrintf(x[i] * inv);
            q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        for (int i = end; i < beg + g; i++) q[i] = 0;
    }
}

/* Matvec against a trunk tensor, quantized path when available. */
static void matvec_t_inner(waste_model *m, float *y, const waste_tensor *t,
                           const float *x, int out, int in);

/* Every dense trunk projection lands here, and on K3 that is 28.0 GB of
 * Q4G read once per token — the single largest byte term in a decode step
 * (docs/LEARNED.md §59). It has no bucket of its own in the profile, which
 * is why "kda 29%" was being read as if it were all recurrence. */
static void matvec_t(waste_model *m, float *y, const waste_tensor *t,
                     const float *x, int out, int in)
{
    if (!prof_on) { matvec_t_inner(m, y, t, x, out, in); return; }
    const double t0 = pnow();
    matvec_t_inner(m, y, t, x, out, in);
    const double dt = pnow() - t0;
    const uint64_t nb = t ? (uint64_t)out * t->rowbytes : 0;
    const int bk = nb < (1u<<20) ? 0 : nb < (8u<<20) ? 1 : nb < (32u<<20) ? 2 : 3;
    pthread_mutex_lock(&prof_mu);
    waste_prof[P_TMV] += dt; waste_prof_n[P_TMV]++;
    waste_tmv_bytes += nb;
    waste_tmv_t[bk] += dt; waste_tmv_b[bk] += nb; waste_tmv_c[bk]++;
    pthread_mutex_unlock(&prof_mu);
}

static void matvec_t_inner(waste_model *m, float *y, const waste_tensor *t,
                           const float *x, int out, int in)
{
    if (!t || (!t->q && !t->data)) { memset(y, 0, (size_t)out * sizeof(float)); return; }
    if (!t->q) { matvec(y, t->data, x, out, in); return; }
    const int g = t->group, ng = (in + g - 1) / g;
    if (trunk_kern != TK_F32 && t->bits == 4 && (g & 31) == 0) {
        mvq4_arg a = { y, (const uint8_t *)t->q, t->qs, m->xq, m->xs,
                       in, ng, g, sdot4_sg, g / sdot4_sg, t->rowbytes };
        waste_range_fn fn = NULL;
        if (trunk_kern == TK_SDOT && g % sdot4_sg == 0) {
            quant_act4(x, in, g, sdot4_sg, m->xq, m->xs);
            fn = mvq4_rows_sdot;
#if defined(__ARM_NEON) || defined(__aarch64__)
        } else if (trunk_kern == TK_I8MM) {
            quant_act4_mm(x, in, g, m->xq, m->xs);
            fn = waste_mvq4_rows_i8mm;
#endif
        } else if (trunk_kern == TK_SMLAL) {
            quant_act4_16(x, in, g, m->xq, m->xs);
            fn = mvq4_rows_smlal;
        }
        if (fn) {
            waste_parallel_for_work(out, 64, fn, &a,
                                    (size_t)out * t->rowbytes);
            if (trunk_check) {
                float *ref = (float *)malloc((size_t)out * sizeof(float));
                if (ref) {
                    mvq_arg r = { ref, t->q, t->qs, NULL, x, in, ng, g,
                                  t->bits, t->rowbytes };
                    waste_parallel_for(out, 64, waste_k.mvq_rows_cpu, &r);
                    double num = 0, den = 0;
                    for (int i = 0; i < out; i++) {
                        const double d = (double)y[i] - ref[i];
                        num += d * d; den += (double)ref[i] * ref[i];
                        if (fabs(d) > waste_tcheck_max) waste_tcheck_max = fabs(d);
                    }
                    if (den > 0) {
                        waste_tcheck_num += num / den;
                        waste_tcheck_den += 1.0;
                        waste_tcheck_n++;
                    }
                    free(ref);
                }
            }
            return;
        }
    }
    if (sdot_on && t->bits == 8) {
        quant_act(x, in, g, m->xq, m->xs);
        mvq_arg a = { y, t->q, t->qs, m->xq, m->xs, in, ng, g, 8, (size_t)ng * g };
        waste_parallel_for_work(out, 64, mvq_rows, &a,
                                (size_t)out * t->rowbytes);
    } else {
        mvq_arg a = { y, t->q, t->qs, NULL, x, in, ng, g, t->bits, t->rowbytes };
        run_rows(out, 64, waste_k.mvq_rows_f32, &a,
                 (size_t)out * t->rowbytes);
    }
}

/* Dequantize one row of a trunk tensor into dst[cols].
 *
 * matvec_t fuses this with the dot product, which is right when every row
 * feeds the same activation vector. MLA's absorbed path does not: each of
 * kv_b_proj's rows is scaled by a different query component and summed the
 * other way round, so the row has to be materialized first. */
void waste_deq_row(const waste_tensor *t, long r, int cols, float *dst)
{
    if (!t->q && !t->data) {          /* skipped at load, e.g. vision off */
        memset(dst, 0, (size_t)cols * sizeof(float));
        return;
    }
    if (!t->q) {
        memcpy(dst, t->data + (size_t)r * cols, (size_t)cols * sizeof(float));
        return;
    }
    const int g = t->group, ng = (cols + g - 1) / g;
    const int8_t *row = t->q + (size_t)r * t->rowbytes;
    const uint16_t *ws = t->qs + (size_t)r * ng;
    for (int k = 0; k < ng; k++) {
        const float s = f16_to_f32(ws[k]);
        const int base = k * g;
        const int lim = (base + g <= cols) ? g : cols - base;
        if (t->bits == 8) {
            const int8_t *w = row + (size_t)base;
            for (int i = 0; i < lim; i++) dst[base + i] = s * (float)w[i];
        } else if (t->bits == 4) {
            const uint8_t *p4 = (const uint8_t *)row + (size_t)base / 2;
            for (int i = 0; i < lim; i++) {
                const uint8_t byte = p4[i >> 1];
                const int v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
                dst[base + i] = s * (float)v;
            }
        } else {
            const uint8_t *p3 = (const uint8_t *)row;
            for (int i = 0; i < lim; i++)
                dst[base + i] = s * (float)q3_at(p3, (long)base + i);
        }
    }
}

/* One row of a tensor that was left on disk, into the model's row scratch.
 * Falls through to the resident pointers when the tensor is in RAM, so
 * callers do not branch. */
static void trunk_row(waste_model *m, const waste_tensor *t, long row,
                      const int8_t **q, const uint16_t **qs)
{
    const int ng = (t->shape[t->ndim - 1] + t->group - 1) / t->group;
    if (!t->on_disk) {
        *q  = t->q  + (size_t)row * t->rowbytes;
        *qs = t->qs + (size_t)row * ng;
        return;
    }
    if (pread_all(m->trunk_fd, m->embrow, t->rowbytes,
                  t->file_off + row * (long)t->rowbytes) ||
        pread_all(m->trunk_fd, m->embsc, (size_t)ng * sizeof(uint16_t),
                  t->file_scale_off + row * (long)ng * 2)) {
        memset(m->embrow, 0, t->rowbytes);
        memset(m->embsc, 0, (size_t)ng * sizeof(uint16_t));
    }
    *q = m->embrow; *qs = m->embsc;
}

static int clamp_token(const waste_model *m, int token);

/* One embedding row into dst. The table is often left quantized — at
 * 163840 x 7168 it is 1.1 GB even at four bits — so this is a dequantize
 * as often as it is a copy, and both prefill and single-token decode need
 * exactly the same answer. Exported because a caller comparing image
 * embeddings against text ones has no other way to ask what scale the
 * model's own vocabulary sits at. */
int waste_embed_row(waste_model *m, int token, float *dst)
{
    const int hid = m->cfg.hidden;
    const waste_tensor *emb = waste_find(m, tname("%smodel.embed_tokens.weight",
                                                  m->cfg.prefix));
    if (!emb) return -1;
    if (emb->data) {
        memcpy(dst, emb->data + (size_t)clamp_token(m, token) * hid,
               (size_t)hid * sizeof(float));
        return 0;
    }
    const int g = emb->group, ng = (hid + g - 1) / g;
    const int8_t *row; const uint16_t *sc;
    trunk_row(m, emb, clamp_token(m, token), &row, &sc);
    for (int k = 0; k < ng; k++) {
        const float sv = f16_to_f32(sc[k]);
        for (int i = 0; i < g && k * g + i < hid; i++) {
            int v;
            if (emb->bits == 3) v = q3_at((const uint8_t *)row, (long)k * g + i);
            else if (emb->bits == 4) {
                const uint8_t byte = ((const uint8_t *)row)[(k * g + i) / 2];
                v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
            } else v = row[k * g + i];
            dst[k * g + i] = (float)v * sv;
        }
    }
    return 0;
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

/* SiTU (K3): beta*tanh(g/beta)*sigmoid(g) * [linear_beta*tanh(u/linear_beta)]
 * — replaces SiLU-and-multiply, and unlike it the "up" half is squashed too. */
float waste_situ_pair(float g, float u, float beta, float lbeta)
{
    const float a = beta * tanhf(g / beta) / (1.0f + expf(-g));
    return a * (lbeta > 0.0f ? lbeta * tanhf(u / lbeta) : u);
}

/* SwiGLU and the two variants of it this family ships, over a whole range.
 *
 * One function rather than the if/else that used to sit at each of its
 * seven call sites: GLM-5.3-Flash adds a third form and the sites are
 * exactly the places a new one gets forgotten. `g` is the gate half and is
 * overwritten with the product; `u` is the up half.
 *
 * The clamp is not cosmetic. GLM applies it to gate above and to up on both
 * sides *before* the SiLU, and at limit 10 it fires on real activations —
 * dropping it leaves a model that looks right and drifts. */
void waste_act_pair_range(const waste_config *c, float *g, const float *u, int n)
{
    if (c->act_situ) {
        for (int i = 0; i < n; i++)
            g[i] = waste_situ_pair(g[i], u[i], c->situ_beta, c->situ_linear_beta);
    } else if (c->swiglu_limit > 0.0f) {
        const float lim = c->swiglu_limit;
        for (int i = 0; i < n; i++) {
            const float a = g[i] > lim ? lim : g[i];
            const float b = u[i] > lim ? lim : (u[i] < -lim ? -lim : u[i]);
            g[i] = silu(a) * b;
        }
    } else {
        for (int i = 0; i < n; i++) g[i] = silu(g[i]) * u[i];
    }
}

static void softmax(float *x, int n)
{
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ---- loading ----------------------------------------------------------- */

/* Reads the trunk one tensor at a time.
 *
 * This used to slurp trunk.bin whole and then copy each tensor out of the
 * buffer, so peak RSS during load was twice the trunk — 57 GB on K3, over
 * the configured budget before the first token was produced, and enough to
 * push a 64 GB machine into memory compression. Nothing needed the whole
 * file resident at once: every tensor knows its own offset. */
static int load_trunk(waste_model *m, const char *dir, const js_doc *d, int trunk)
{
    char path[MAXP];
    snprintf(path, sizeof path, "%s/trunk.bin", dir);
    const int fd = open(path, O_RDONLY | WASTE_O_BINARY);
    if (fd < 0) return -1;
#define TRUNK_FAIL do { close(fd); return -1; } while (0)
    /* Every offset and length below comes from the manifest, so bound them
     * by the one thing that is not a claim: how big the file actually is.
     * Without it a declared shape of [2^20, 2^20] asks for a 4 TB
     * allocation before anything notices. */
    const int64_t fsize = waste_file_size(fd);
    if (fsize < 0) TRUNK_FAIL;

    m->n_tensors = js_size(d, trunk);
    m->t = (waste_tensor *)calloc((size_t)m->n_tensors, sizeof *m->t);
    /* n_tensors is set before the allocation is known to have worked, and
     * waste_model_free walks it — so a failure here has to leave the two
     * agreeing rather than a count pointing at NULL. */
    if (!m->t) { m->n_tensors = 0; TRUNK_FAIL; }

    for (int i = 0; i < m->n_tensors; i++) {
        int e = js_at(d, trunk, i);
        waste_tensor *t = &m->t[i];
        js_str(d, js_get(d, e, "name"), t->name, sizeof t->name);
        const int fmt = (int)js_int(d, js_get(d, e, "fmt"), 0);
        const int64_t off = js_int(d, js_get(d, e, "off"), 0);
        const int g = (int)js_int(d, js_get(d, e, "group"), 128);
        const int64_t soff = js_int(d, js_get(d, e, "scale_off"), 0);
        int sh = js_get(d, e, "shape");
        t->ndim = js_size(d, sh);
        /* Shapes and the group size divide and index below, so a manifest
         * declaring [] or [0] or group 0 has to stop here rather than reach
         * `t->n / N`. A fuzzer found all three. */
        if (t->ndim < 1 || t->ndim > 4 || g < 1) TRUNK_FAIL;
        if (off < 0 || off > fsize || soff < 0 || soff > fsize) TRUNK_FAIL;
        t->n = 1;
        for (int k = 0; k < t->ndim && k < 4; k++) {
            t->shape[k] = (int)js_int(d, js_at(d, sh, k), 1);
            if (t->shape[k] < 1 || t->shape[k] > (1 << 26)) TRUNK_FAIL;
            t->n *= (size_t)t->shape[k];
            /* One element per byte is already generous — nothing in this
             * format stores less than a bit per weight — so a tensor with
             * more elements than the file has bytes is a lie. */
            if (t->n > (size_t)fsize) TRUNK_FAIL;
        }
        /* K3 is multimodal and the container carries its vision tower and
         * projector, 223 MB of them. The engine implements the text path
         * only — every lookup goes through cfg.prefix — so loading them
         * spends the one resource the whole design is fighting for. The
         * bytes stay in trunk.bin for the day vision lands. */
        /* Record the storage format before the skip below: embed_tokens
         * and the vision tower never reach the code that sets t->bits, and
         * `info` reports the trunk's formats from this. */
        if (!m->cfg.prefix[0] ||
            !strncmp(t->name, m->cfg.prefix, strlen(m->cfg.prefix)))
            m->trunk_fmts |= 1u << (fmt & 31);

        const int is_vision = !strncmp(t->name, "vision_tower.", 13) ||
                              !strncmp(t->name, "mm_projector.", 13);
        /* The tower is skipped unless it was asked for, and that has to be
         * its own test rather than a corollary of the prefix one: K3's
         * container has a prefix and every non-prefixed tensor is the
         * tower, so the two coincided there. GLM's container has no prefix
         * at all, and the coincidence made its 282 MB resident on every
         * open, image or no image. */
        if (is_vision && !m->want_vision) {
            t->on_disk = 1;
            t->file_off = off;
            t->file_scale_off = soff;
            continue;
        }
        if (m->cfg.prefix[0] && !is_vision &&
            strncmp(t->name, m->cfg.prefix, strlen(m->cfg.prefix)) != 0) {
            t->on_disk = 1;
            t->file_off = off;
            t->file_scale_off = soff;
            continue;
        }

        if (fmt == 0) {                                   /* F32 */
            t->data = (float *)waste_dio_alloc(t->n * sizeof(float));
            if (!t->data) TRUNK_FAIL;
            if (pread_all(fd, t->data, t->n * sizeof(float), off)) TRUNK_FAIL;
        } else {                              /* Q8G, Q4G or Q3G */
            const int N = t->shape[t->ndim - 1];
            const int64_t rows = (int64_t)(t->n / (size_t)N);
            const int ng = (N + g - 1) / g;
            t->group = g;
            t->bits = (fmt == 3) ? 4 : (fmt == 7) ? 3 : 8;
            /* 3-bit rows are a bitstream plus one guard byte */
            t->rowbytes = (t->bits == 3)
                        ? (size_t)((ng * g * 3 + 7) / 8 + 1)
                        : (size_t)ng * g * t->bits / 8;
            const size_t payload = (size_t)rows * t->rowbytes;
            /* The embedding table is the one big tensor of which a single
             * row is read per token. Keeping 1.11 GB resident to touch 7 KB
             * of it is a bad trade against the expert cache, so leave it on
             * disk and pread the row. */
            if (strstr(t->name, "embed_tokens.weight") ||
                strstr(t->name, "ngram_head.")) {
                t->on_disk = 1;
                t->file_off = off;
                t->file_scale_off = soff;
                continue;
            }
            t->q = (int8_t *)waste_dio_alloc(payload);
            t->qs = (uint16_t *)waste_dio_alloc((size_t)rows * ng * sizeof(uint16_t));
            if (!t->q || !t->qs) TRUNK_FAIL;
            if (pread_all(fd, t->q, payload, off) ||
                pread_all(fd, t->qs, (size_t)rows * ng * sizeof(uint16_t), soff))
                TRUNK_FAIL;

            /* WASTE_Q8=0 dequantizes the trunk once here instead of at every
             * use, and it runs *after* the quantized load so it can hand the
             * rows to waste_deq_row — the same unpacker the compute path
             * calls. A width this code does not itself know cannot be
             * mis-read at load while working everywhere else.
             *
             * It used to be a branch of its own, ahead of this one, with a
             * private copy of the Q8 unpacking that read `rows * ng * g`
             * bytes: one byte per weight, true of Q8G alone, while the
             * branch caught every non-F32 format. A Q4G trunk — what
             * --trunk-bits 4, the default, produces — asked for twice the
             * bytes it occupies and failed to load. Only the synthetic
             * container, Q8G/F32 throughout, took the shape it handled,
             * which is why CI never saw it.
             *
             * embed_tokens stays on disk above either way. Its rows go
             * through this same unpacker when read, so the logits are
             * identical, 1.41 GiB stays out of the resident set, and the
             * f32-equivalence check differs from the default path in the
             * storage width alone — which is what it claims to compare. */
            if (!q8_off) {
                float *f32 = (float *)waste_dio_alloc(t->n * sizeof(float));
                if (!f32) TRUNK_FAIL;
                for (int64_t r = 0; r < rows; r++)
                    waste_deq_row(t, (long)r, N, f32 + (size_t)r * (size_t)N);
                waste_dio_free(t->q);  t->q  = NULL;
                waste_dio_free(t->qs); t->qs = NULL;
                /* Indistinguishable from a tensor that was F32 on disk:
                 * the kernels branch on bits without consulting q, and
                 * m->t is calloc'd, so a real F32 tensor carries zeroes in
                 * all three. wire_trunk reads data and q independently and
                 * skips the quantized half on group <= 0. */
                t->bits = 0; t->rowbytes = 0; t->group = 0;
                t->data = f32;
            }
        }
    }
    m->trunk_fd = fd;   /* on-disk tensors read through it */
#undef TRUNK_FAIL
    return 0;
}

static int tensor_matrix_ok(const waste_model *m, const char *name, int rows, int cols)
{
    const waste_tensor *t = waste_find(m, name);
    return t && t->ndim == 2 && t->shape[0] == rows && t->shape[1] == cols &&
           (t->data || t->q || t->on_disk);
}

static int tensor_vector_ok(const waste_model *m, const char *name, int n)
{
    const waste_tensor *t = waste_find(m, name);
    return t && t->n == (size_t)n && t->data;
}

static int tensor_data_ok(const waste_model *m, const char *name, size_t n)
{
    const waste_tensor *t = waste_find(m, name);
    return t && t->n >= n && t->data;
}

static int bad_tensor(const char *name)
{
    fprintf(stderr, "waste: required tensor is missing or has the wrong shape: %s\n",
            name);
    return 0;
}

#define REQUIRE_MATRIX(name, rows, cols) \
    do { const char *rn_ = (name); if (!tensor_matrix_ok(m, rn_, (rows), (cols))) \
        return bad_tensor(rn_); } while (0)
#define REQUIRE_VECTOR(name, n) \
    do { const char *rn_ = (name); if (!tensor_vector_ok(m, rn_, (n))) \
        return bad_tensor(rn_); } while (0)
#define REQUIRE_DATA(name, n) \
    do { const char *rn_ = (name); if (!tensor_data_ok(m, rn_, (n))) \
        return bad_tensor(rn_); } while (0)

static int validate_qwen_tensors(waste_model *m)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, hc = c->hc_count, lr = c->hc_lowrank;
    const int H = hc * hid;
    REQUIRE_MATRIX(tname("%smodel.embed_tokens.weight", c->prefix), c->vocab, hid);
    REQUIRE_MATRIX(tname("%slm_head.weight", c->prefix), c->vocab, hid);
    REQUIRE_VECTOR(tname("%smodel.hyper_connection_mixer.hc_norm.weight", c->prefix), H);
    REQUIRE_MATRIX(tname("%smodel.hyper_connection_mixer.input_mix_weight_down.weight", c->prefix), lr, H);
    REQUIRE_MATRIX(tname("%smodel.hyper_connection_mixer.input_mix_weight_up.weight", c->prefix), H, lr);

    const int Hk = c->gdn_k_heads, Hv = c->gdn_v_heads;
    const int Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
    const int qkv = 2 * Hk * Dk + Hv * Dv;
    const int qd = c->n_heads * c->qsa_head_dim;
    const int kvd = c->qsa_n_kv * c->qsa_head_dim;
    const int idxd = (c->idx_n_heads + c->idx_kv_heads) * c->idx_head_dim;
    const int shared = c->shared_inter ? c->shared_inter : c->moe_inter;

    for (int L = 0; L < c->n_layers; L++) {
        const char *side[2] = { "attn_hyper_connection", "mlp_hyper_connection" };
        for (int s = 0; s < 2; s++) {
            REQUIRE_VECTOR(tname("%smodel.layers.%d.%s.hc_norm.weight", c->prefix, L, side[s]), H);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.%s.block_inject_weight.weight", c->prefix, L, side[s]), hc, H);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.%s.input_mix_weight_down.weight", c->prefix, L, side[s]), lr, H);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.%s.input_mix_weight_up.weight", c->prefix, L, side[s]), H, lr);
        }
        if (!c->qwen_full[L]) {
            REQUIRE_DATA(tname("%smodel.layers.%d.linear_attn.A_log", c->prefix, L), Hv);
            REQUIRE_DATA(tname("%smodel.layers.%d.linear_attn.dt_bias", c->prefix, L), Hv);
            REQUIRE_DATA(tname("%smodel.layers.%d.linear_attn.conv1d.weight", c->prefix, L),
                         (size_t)qkv * c->conv_k);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.linear_attn.in_proj_qkv.weight", c->prefix, L), qkv, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.linear_attn.in_proj_z.weight", c->prefix, L), Hv * Dv, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.linear_attn.in_proj_a.weight", c->prefix, L), Hv, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.linear_attn.in_proj_b.weight", c->prefix, L), Hv, hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.linear_attn.norm.weight", c->prefix, L), Dv);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.linear_attn.out_proj.weight", c->prefix, L), hid, Hv * Dv);
        } else {
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.q_proj.weight", c->prefix, L), qd * 2, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.k_proj.weight", c->prefix, L), kvd, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.v_proj.weight", c->prefix, L), kvd, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L), hid, qd);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.q_norm.weight", c->prefix, L), c->qsa_head_dim);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.k_norm.weight", c->prefix, L), c->qsa_head_dim);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.indexer.index_qk_proj.weight", c->prefix, L), idxd, hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.indexer.q_layernorm.weight", c->prefix, L), c->idx_head_dim);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.indexer.k_layernorm.weight", c->prefix, L), c->idx_head_dim);
        }
        REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.gate.weight", c->prefix, L), c->n_experts, hid);
        REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.shared_expert.gate_proj.weight", c->prefix, L), shared, hid);
        REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.shared_expert.up_proj.weight", c->prefix, L), shared, hid);
        REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.shared_expert.down_proj.weight", c->prefix, L), hid, shared);
        REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.shared_expert_gate.weight", c->prefix, L), 1, hid);
        if (L == c->ple_layer) {
            REQUIRE_MATRIX(tname("%smodel.layers.%d.ple.key_proj.weight", c->prefix, L), H, c->ple_embed ? c->ple_embed : hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.ple.value_proj.weight", c->prefix, L), hid, c->ple_embed ? c->ple_embed : hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.ple.norm_key.weight", c->prefix, L), H);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.ple.norm_query.weight", c->prefix, L), H);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.ple.norm_conv.weight", c->prefix, L), H);
            REQUIRE_DATA(tname("%smodel.layers.%d.ple.conv1d.weight", c->prefix, L),
                         (size_t)H * (c->ple_conv_k > 0 ? c->ple_conv_k : 4));
            for (int h = 0; h < WASTE_QWEN_PLE_HEADS; h++) {
                const int rows = c->ple_sz[h] > 0 ? (int)c->ple_sz[h] : 1;
                const int width = (c->ple_embed && c->heads_per_ngram)
                    ? c->ple_embed / ((c->ngram_size - 1) * c->heads_per_ngram) : 8;
                REQUIRE_MATRIX(tname("%smodel.layers.%d.ple.ple_embedding.ngram_head.%d.weight",
                                     c->prefix, L, h), rows, width);
            }
        }
    }
    return 1;
}

/* Validate every tensor shape the text forward pass indexes.  Kernel calls
 * receive dimensions from config rather than from the tensor, so merely
 * checking that a name exists is not enough: a shorter, correctly named
 * tensor is an out-of-bounds read. */
static int validate_text_tensors(waste_model *m)
{
    if (m->cfg.arch_qwen) return validate_qwen_tensors(m);
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    REQUIRE_MATRIX(tname("%smodel.embed_tokens.weight", c->prefix), c->vocab, hid);
    REQUIRE_VECTOR(tname("%smodel.norm.weight", c->prefix), hid);
    REQUIRE_MATRIX(tname("%slm_head.weight", c->prefix), c->vocab, hid);

    for (int L = 0; L < c->n_layers; L++) {
        REQUIRE_VECTOR(tname("%smodel.layers.%d.input_layernorm.weight", c->prefix, L), hid);
        REQUIRE_VECTOR(tname("%smodel.layers.%d.post_attention_layernorm.weight", c->prefix, L), hid);

        if (c->kda_layer[L]) {
            const int H = c->kda_heads, D = c->kda_dim, C = H * D;
            const char *kind[3] = { "q", "k", "v" };
            for (int i = 0; i < 3; i++) {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.%s_proj.weight",
                                     c->prefix, L, kind[i]), C, hid);
                REQUIRE_DATA(tname("%smodel.layers.%d.self_attn.%s_conv1d.weight",
                                   c->prefix, L, kind[i]), (size_t)C * c->conv_k);
            }
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.f_a_proj.weight", c->prefix, L), D, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.f_b_proj.weight", c->prefix, L), C, D);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.b_proj.weight", c->prefix, L), H, hid);
            REQUIRE_DATA(tname("%smodel.layers.%d.self_attn.A_log", c->prefix, L), H);
            REQUIRE_DATA(tname("%smodel.layers.%d.self_attn.dt_bias", c->prefix, L), C);
            if (c->full_rank_gate) {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.g_proj.weight", c->prefix, L), C, hid);
            } else {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.g_a_proj.weight", c->prefix, L), D, hid);
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.g_b_proj.weight", c->prefix, L), C, D);
            }
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.o_norm.weight", c->prefix, L), D);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L), hid, C);
        } else {
            const int qd = c->qk_nope + c->qk_rope;
            if (c->q_lora) {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.q_a_proj.weight", c->prefix, L), c->q_lora, hid);
                REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.q_a_layernorm.weight", c->prefix, L), c->q_lora);
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.q_b_proj.weight", c->prefix, L), c->n_heads * qd, c->q_lora);
            } else {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.q_proj.weight", c->prefix, L), c->n_heads * qd, hid);
            }
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.kv_a_proj_with_mqa.weight", c->prefix, L), c->kv_lora + c->qk_rope, hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attn.kv_a_layernorm.weight", c->prefix, L), c->kv_lora);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.kv_b_proj.weight", c->prefix, L), c->n_heads * (c->qk_nope + c->v_head), c->kv_lora);
            if (c->mla_output_gate)
                REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.g_proj.weight", c->prefix, L), c->n_heads * c->v_head, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L), hid, c->n_heads * c->v_head);
        }

        if (c->hc_mult) {
            const int nmix = (2 + c->hc_mult) * c->hc_mult;
            const char *site[2] = { "attn", "ffn" };
            for (int i = 0; i < 2; i++) {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.hc_%s_fn", c->prefix, L, site[i]),
                               nmix, c->hc_mult * hid);
                REQUIRE_DATA(tname("%smodel.layers.%d.hc_%s_base", c->prefix, L, site[i]),
                             (size_t)nmix);
                REQUIRE_DATA(tname("%smodel.layers.%d.hc_%s_scale", c->prefix, L, site[i]), 3);
            }
        }
        if (c->index_topk && !c->kda_layer[L]) {
            const char *ix = "%smodel.layers.%d.self_attn.indexer.%s";
            REQUIRE_MATRIX(tname(ix, c->prefix, L, "wq_b.weight"),
                           c->index_heads * c->index_dim, c->q_lora);
            REQUIRE_MATRIX(tname(ix, c->prefix, L, "wk.weight"), c->index_dim, hid);
            REQUIRE_VECTOR(tname(ix, c->prefix, L, "k_norm.weight"), c->index_dim);
            REQUIRE_VECTOR(tname(ix, c->prefix, L, "k_norm.bias"), c->index_dim);
            REQUIRE_MATRIX(tname(ix, c->prefix, L, "weights_proj.weight"), c->index_heads, hid);
            REQUIRE_MATRIX(tname(ix, c->prefix, L, "index_kpool_compress_gate"),
                           c->index_dim, hid);
            REQUIRE_DATA(tname(ix, c->prefix, L, "index_kpool_compress_ape"),
                         (size_t)c->index_kpool * c->index_dim);
        }

        if (c->attn_res_block) {
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attention_res_norm.weight", c->prefix, L), hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.self_attention_res_proj.weight", c->prefix, L), hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.mlp_res_norm.weight", c->prefix, L), hid);
            REQUIRE_VECTOR(tname("%smodel.layers.%d.mlp_res_proj.weight", c->prefix, L), hid);
        }

        if (c->n_experts && L >= c->first_dense) {
            const int lat = c->latent_dim ? c->latent_dim : hid;
            const int shared = c->moe_inter * (c->n_shared ? c->n_shared : 1);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L), c->n_experts, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", c->prefix, L), shared, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", c->prefix, L), shared, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", c->prefix, L), hid, shared);
            if (c->latent_dim) {
                REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight", c->prefix, L), lat, hid);
                REQUIRE_MATRIX(tname("%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight", c->prefix, L), hid, lat);
                if (c->latent_norm)
                    REQUIRE_VECTOR(tname("%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight", c->prefix, L), lat);
            }
        } else {
            REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.gate_proj.weight", c->prefix, L), c->dense_inter, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.up_proj.weight", c->prefix, L), c->dense_inter, hid);
            REQUIRE_MATRIX(tname("%smodel.layers.%d.mlp.down_proj.weight", c->prefix, L), hid, c->dense_inter);
        }
    }
    if (c->attn_res_block) {
        REQUIRE_VECTOR(tname("%smodel.output_attn_res_norm.weight", c->prefix), hid);
        REQUIRE_VECTOR(tname("%smodel.output_attn_res_proj.weight", c->prefix), hid);
    }
    return 1;
}

#undef REQUIRE_MATRIX
#undef REQUIRE_VECTOR
#undef REQUIRE_DATA

/* A manifest is untrusted input, and these numbers size allocations and
 * bound loops that index fixed arrays. A config claiming 200 layers walks
 * off the end of waste_model's [WASTE_MAX_LAYERS] arrays; one claiming
 * zero of anything produces empty allocations that later get written.
 * Neither is a wrong answer — both are memory corruption — so refuse. */
static int cfg_sane(const waste_config *c)
{
    if (c->n_layers < 1 || c->n_layers > WASTE_MAX_LAYERS) return 0;
    if (c->hidden   < 1 || c->hidden   > (1 << 20)) return 0;
    if (c->vocab    < 1 || c->vocab    > (1 << 24)) return 0;
    if (c->n_heads  < 1 || c->n_heads  > (1 << 16)) return 0;
    if (c->eps <= 0.0f || !(c->eps < 1.0f)) return 0;      /* also catches NaN */
    /* MoE is optional, but if there are experts the routing has to make
     * sense: top_k above the pool overruns the per-token index array. */
    if (c->n_experts < 0 || c->n_experts > (1 << 20)) return 0;
    if (c->n_experts && (c->top_k < 1 || c->top_k > c->n_experts ||
                         c->top_k > 64)) return 0;
    if (c->moe_inter < 0 || c->moe_inter > (1 << 20)) return 0;
    if (c->dense_inter < 0 || c->dense_inter > (1 << 20)) return 0;
    if (c->kda_heads < 0 || c->kda_heads > (1 << 16)) return 0;
    if (c->kda_dim   < 0 || c->kda_dim   > (1 << 16)) return 0;
    if (c->conv_k    < 0 || c->conv_k    > 64) return 0;
    if (c->kv_lora < 0 || c->kv_lora > (1 << 20) ||
        c->q_lora < 0 || c->q_lora > (1 << 20)) return 0;
    if (c->qk_nope < 0 || c->qk_nope > (1 << 20) ||
        c->qk_rope < 0 || c->qk_rope > (1 << 20) ||
        c->v_head < 0 || c->v_head > (1 << 20)) return 0;
    if (c->first_dense < 0 || c->first_dense > c->n_layers) return 0;
    if (c->n_shared  < 0 || c->n_shared  > 64) return 0;
    if (c->latent_dim < 0 || c->latent_dim > (1 << 20)) return 0;
    if (c->attn_res_block < 0 || c->attn_res_block > c->n_layers) return 0;
    int n_kda = 0;
    for (int L = 0; L < c->n_layers; L++) n_kda += !!c->kda_layer[L];
    if (n_kda) {
        if (c->kda_heads < 1 || c->kda_dim < 1 || c->conv_k < 1) return 0;
        if ((int64_t)c->kda_heads * c->kda_dim > INT_MAX) return 0;
    }
    if (!c->arch_qwen && n_kda < c->n_layers) {
        const int64_t qd = (int64_t)c->qk_nope + c->qk_rope;
        if (c->kv_lora < 1 || qd < 1 || c->v_head < 1) return 0;
        if ((int64_t)c->n_heads * qd > INT_MAX ||
            (int64_t)c->n_heads * c->v_head > INT_MAX ||
            (int64_t)c->n_heads * (c->qk_nope + c->v_head) > INT_MAX)
            return 0;
    }
    /* mHC multiplies the resident residual stream and every buffer sized
     * from it, so it is bounded like a dimension rather than a flag. */
    if (c->hc_mult < 0 || c->hc_mult > 16) return 0;
    if (c->hc_mult) {
        if (c->hc_iters < 1 || c->hc_iters > 1024) return 0;
        if (!(c->hc_eps > 0.0f) || !(c->hc_eps < 1.0f)) return 0;
        if ((int64_t)c->hc_mult * c->hidden > INT_MAX) return 0;
    }
    if (!(c->swiglu_limit >= 0.0f)) return 0;              /* also NaN */
    /* The indexer is all-or-nothing: a container that states a topk without
     * the shapes to score with would silently attend over nothing. */
    if (c->index_kpool < 1 || c->index_kpool > 64) return 0;
    if (c->index_topk < 0 || c->index_topk > (1 << 24)) return 0;
    if (c->index_heads < 0 || c->index_heads > (1 << 16)) return 0;
    if (c->index_dim < 0 || c->index_dim > (1 << 20)) return 0;
    if (c->index_topk) {
        if (c->index_heads < 1 || c->index_dim < 1) return 0;
        if (c->index_topk % c->index_kpool) return 0;
        if ((int64_t)c->index_heads * c->index_dim > INT_MAX) return 0;
    } else if (c->index_heads || c->index_dim) {
        return 0;
    }
    /* Qwen states its shapes in its own keys, and every one of them sizes
     * an allocation or indexes a loop below. A container that omits one is
     * refused here rather than opened and read out of bounds. */
    if (c->arch_qwen) {
        if (c->qwen_n_layer_types != c->n_layers) return 0;
        if (c->gdn_k_heads < 1 || c->gdn_v_heads < 1 ||
            c->gdn_k_dim < 1 || c->gdn_v_dim < 1) return 0;
        if (c->gdn_v_heads % c->gdn_k_heads != 0) return 0;
        if (c->hc_count < 1 || c->hc_count > 16 ||
            c->hc_lowrank < 1 || c->hc_lowrank > (1 << 16)) return 0;
        if (c->qsa_head_dim < 1 || c->qsa_n_kv < 1) return 0;
        if (c->idx_n_heads < 1 || c->idx_head_dim < 1 ||
            c->idx_compress < 1 || c->idx_budget < 1) return 0;
        if (c->idx_budget % c->idx_compress != 0) return 0;
        if (c->idx_kv_heads != 1) return 0;
        if (c->n_heads % c->qsa_n_kv != 0) return 0;
        if (c->ngram_size < 1 || c->ngram_size > 8) return 0;
        if (c->rotary_dim < 0 || c->rotary_dim > 256) return 0;
        if (c->rotary_dim / 2 > WASTE_MAX_ROPE_HALF) return 0;
        if ((int64_t)c->hc_count * c->hidden > INT_MAX) return 0;
        if ((int64_t)c->gdn_v_heads * c->gdn_k_dim * c->gdn_v_dim > INT_MAX)
            return 0;
        /* The QSA query buffers are n_heads * head_dim and the indexer's
         * work is idx_n_heads * idx_head_dim; both are computed as int
         * before they reach a size_t, so bound the products, not just the
         * factors. */
        if ((int64_t)c->n_heads * c->qsa_head_dim > INT_MAX / 4) return 0;
        if ((int64_t)c->idx_n_heads * c->idx_head_dim > INT_MAX / 4) return 0;
        if ((int64_t)c->idx_budget + c->idx_compress > INT_MAX / 4) return 0;
        if (c->ple_layer >= 0) {
            for (int h = 0; h < WASTE_QWEN_PLE_HEADS; h++)
                if (c->ple_sz[h] <= 0) return 0;
        }
    }
    if (c->n_experts && c->moe_inter < 1) return 0;
    if ((!c->n_experts || c->first_dense) && c->dense_inter < 1) return 0;
    if ((int64_t)c->moe_inter * (c->n_shared ? c->n_shared : 1) > INT_MAX)
        return 0;
    return 1;
}

/* inv_freq for the rope dims, plus YaRN's factor on the attention scale.
 *
 * Both follow DeepseekV3YarnRotaryEmbedding. Two details do not survive
 * paraphrase:
 *   - mscale appears twice with different meanings. cos/sin carry
 *     mscale / mscale_all_dim, which is 1 whenever the two are equal (K2 sets
 *     both to 1), so they are left alone here. The attention scale carries
 *     mscale_all_dim SQUARED, which is 1.8133x on K2.
 *   - YaRN rescales inv_freq globally, so it applies from position 0. It is
 *     not a long-context-only correction that a short prompt can ignore.
 *
 * A shape this does not implement leaves a reason in c->rope_err and the
 * load refuses on it. Falling through to plain RoPE instead would be the
 * same failure this function was added to fix: not a degraded answer but an
 * unordered one, and one that looks like weight-shaped logits.
 */
static void rope_init(waste_config *c, const js_doc *d, int cfg)
{
    const double PI = 3.14159265358979323846;
    c->att_mul = 1.0f;
    c->rope_err[0] = 0;
    /* By value, not by presence: a container carrying "mla_use_nope": false
     * has to rotate. The presence idiom used for the other flags costs a
     * feature when it misreads; here it costs the sequence order.
     *
     * Present but not a boolean is refused rather than defaulted, because
     * defaulting picks the sequence order from a manifest that did not say
     * which one it wanted, and picks it silently. */
    const int nope = js_get(d, cfg, "mla_use_nope");
    c->mla_nope = 0;
    if (nope >= 0 && js_typeof(d, nope) != JS_BOOL) {
        snprintf(c->rope_err, sizeof c->rope_err,
                 "mla_use_nope is present but is not true or false");
        return;
    }
    c->mla_nope = js_bool(d, nope, 0);
    const int dim = c->qk_rope, half = dim / 2;
    if (c->mla_nope || half <= 0) return;
    if (half > WASTE_MAX_ROPE_HALF) {
        snprintf(c->rope_err, sizeof c->rope_err,
                 "qk_rope_head_dim %d needs rotation, this build holds %d",
                 dim, 2 * WASTE_MAX_ROPE_HALF);
        return;
    }

    const double base = js_num(d, js_get(d, cfg, "rope_theta"), 10000.0);
    for (int j = 0; j < half; j++)
        c->rope_inv_freq[j] = (float)(1.0 / pow(base, (double)(2 * j) / dim));

    /* A key that is absent, null or {} all mean no scaling, and js_size is 0
     * for each — the plain-RoPE table above is already the whole answer.
     * null is how HF configs spell it and convert.py copies them verbatim,
     * so this is the common shape, not the corner. */
    const int rs = js_get(d, cfg, "rope_scaling");
    if (rs < 0 || js_size(d, rs) == 0) return;
    char type[24];
    int ty = js_get(d, rs, "type");
    if (ty < 0) ty = js_get(d, rs, "rope_type");   /* HF renamed the key */
    js_str(d, ty, type, sizeof type);              /* "" if absent or not a string */
    if (!type[0]) {
        snprintf(c->rope_err, sizeof c->rope_err,
                 "rope_scaling carries no type string, and only yarn is "
                 "implemented");
        return;
    }
    if (strcmp(type, "yarn") != 0) {
        snprintf(c->rope_err, sizeof c->rope_err,
                 "rope_scaling type \"%s\" is not implemented, only yarn", type);
        return;
    }
    /* factor <= 1 is not a refusal: YaRN's ramp is the identity there and
     * both mscales collapse to 1, so plain RoPE is the right answer. */
    const double factor = js_num(d, js_get(d, rs, "factor"), 1.0);
    if (factor <= 1.0) return;
    /* Unequal mscales put a ratio on cos/sin that nothing here applies.
     * V3, R1, K2 and V2 all ship them equal; HF's defaults (1 and 0) are
     * not, so an omitted mscale_all_dim lands here too. */
    const double m_one = js_num(d, js_get(d, rs, "mscale"), 1.0);
    const double m_dim = js_num(d, js_get(d, rs, "mscale_all_dim"), 0.0);
    if (m_one != m_dim) {
        snprintf(c->rope_err, sizeof c->rope_err,
                 "rope_scaling mscale %g != mscale_all_dim %g, and the ratio "
                 "on cos/sin is not implemented", m_one, m_dim);
        return;
    }

    const double orig = js_num(d, js_get(d, rs, "original_max_position_embeddings"), 4096.0);
    const double bf = js_num(d, js_get(d, rs, "beta_fast"), 32.0);
    const double bs = js_num(d, js_get(d, rs, "beta_slow"), 1.0);
    double low = floor(dim * log(orig / (bf * 2.0 * PI)) / (2.0 * log(base)));
    double high = ceil(dim * log(orig / (bs * 2.0 * PI)) / (2.0 * log(base)));
    if (low < 0.0) low = 0.0;
    if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;             /* upstream's singularity guard */
    for (int j = 0; j < half; j++) {
        double ramp = ((double)j - low) / (high - low);
        ramp = ramp < 0.0 ? 0.0 : ramp > 1.0 ? 1.0 : ramp;
        const double mask = 1.0 - ramp;         /* 1 = extrapolate, 0 = interpolate */
        const double extra = c->rope_inv_freq[j];
        c->rope_inv_freq[j] = (float)((extra / factor) * (1.0 - mask) + extra * mask);
    }
    if (m_dim != 0.0) {
        const double ms = 0.1 * m_dim * log(factor) + 1.0;
        c->att_mul = (float)(ms * ms);
    }
}

static void cfg_from_json(waste_config *c, const js_doc *d, int cfg)
{
    c->n_layers = (int)js_int(d, js_get(d, cfg, "num_hidden_layers"), 0);
    c->hidden = (int)js_int(d, js_get(d, cfg, "hidden_size"), 0);
    c->n_experts = (int)js_int(d, js_get(d, cfg, "num_experts"), 0);
    c->top_k = (int)js_int(d, js_get(d, cfg, "num_experts_per_token"), 0);
    if (!c->top_k)
        c->top_k = (int)js_int(d, js_get(d, cfg, "num_experts_per_tok"), 0);
    c->moe_inter = (int)js_int(d, js_get(d, cfg, "moe_intermediate_size"), 0);
    c->dense_inter = (int)js_int(d, js_get(d, cfg, "intermediate_size"), 0);
    c->n_shared = (int)js_int(d, js_get(d, cfg, "num_shared_experts"), 0);
    c->first_dense = (int)js_int(d, js_get(d, cfg, "first_k_dense_replace"), 0);
    c->vocab = (int)js_int(d, js_get(d, cfg, "vocab_size"), 0);
    c->eos_token_id = (int)js_int(d, js_get(d, cfg, "eos_token_id"), 0);
    c->n_heads = (int)js_int(d, js_get(d, cfg, "num_attention_heads"), 0);
    c->kv_lora = (int)js_int(d, js_get(d, cfg, "kv_lora_rank"), 0);
    c->q_lora = (int)js_int(d, js_get(d, cfg, "q_lora_rank"), 0);
    c->qk_nope = (int)js_int(d, js_get(d, cfg, "qk_nope_head_dim"), 0);
    c->qk_rope = (int)js_int(d, js_get(d, cfg, "qk_rope_head_dim"), 0);
    c->v_head = (int)js_int(d, js_get(d, cfg, "v_head_dim"), 0);
    c->eps = (float)js_num(d, js_get(d, cfg, "rms_norm_eps"), 1e-5);
    c->routed_scale = (float)js_num(d, js_get(d, cfg, "routed_scaling_factor"), 1.0);
    c->renorm = js_get(d, cfg, "moe_renormalize") >= 0;

    c->latent_dim = (int)js_int(d, js_get(d, cfg, "routed_expert_hidden_size"), 0);
    c->latent_norm = js_get(d, cfg, "latent_moe_use_norm") >= 0;
    c->attn_res_block = (int)js_int(d, js_get(d, cfg, "attn_res_block_size"), 0);
    c->mla_output_gate = js_get(d, cfg, "mla_use_output_gate") >= 0;
    {
        char act[32];
        js_str(d, js_get(d, cfg, "hidden_act"), act, sizeof act);
        c->act_situ = strcmp(act, "situ") == 0;
    }
    c->situ_beta = (float)js_num(d, js_get(d, cfg, "activation_situ_beta"), 1.0);
    c->situ_linear_beta = (float)js_num(d, js_get(d, cfg, "activation_situ_linear_beta"), 0.0);

    /* K3 is a KimiK3ForConditionalGeneration wrapped around a Kimi-Linear
     * block, and the converter keeps the wrapper's config under `_outer`.
     * The inner `architectures` says KimiLinearForCausalLM on both models,
     * so the outer one is the only place K3 names itself. */
    {
        int a = js_get(d, js_get(d, cfg, "_outer"), "architectures");
        if (a < 0) a = js_get(d, cfg, "architectures");
        js_str(d, js_at(d, a, 0), c->arch, sizeof c->arch);
    }

    rope_init(c, d, cfg);

    /* GLM-5.3-Flash: mHC, the clamped SwiGLU and the DSA indexer. Absent on
     * every Kimi container, where all of these read back 0 and the ordinary
     * paths run unchanged. */
    c->hc_mult  = (int)js_int(d, js_get(d, cfg, "hc_mult"), 0);
    c->hc_iters = (int)js_int(d, js_get(d, cfg, "hc_sinkhorn_iters"), 0);
    c->hc_eps   = (float)js_num(d, js_get(d, cfg, "hc_eps"), 1e-6);
    c->swiglu_limit = (float)js_num(d, js_get(d, cfg, "swiglu_limit"), 0.0);
    c->index_topk  = (int)js_int(d, js_get(d, cfg, "index_topk"), 0);
    c->index_kpool = (int)js_int(d, js_get(d, cfg, "index_kpool"), 1);
    c->index_heads = (int)js_int(d, js_get(d, cfg, "index_n_heads"), 0);
    c->index_dim   = (int)js_int(d, js_get(d, cfg, "index_head_dim"), 0);
    c->index_tail  = js_get(d, cfg, "index_kpool_always_select_tail") >= 0;
    c->tok_han_split = js_bool(d, js_get(d, cfg, "tokenizer_han_split"), 1);
    c->tok_digit_run = (int)js_int(d, js_get(d, cfg, "tokenizer_digit_run"), 3);

    int lac = js_get(d, cfg, "linear_attn_config");
    c->full_rank_gate = js_get(d, lac, "use_full_rank_gate") >= 0;
    c->gate_lower_bound = (float)js_num(d, js_get(d, lac, "gate_lower_bound"), 0.0);
    c->kda_heads = (int)js_int(d, js_get(d, lac, "num_heads"), 0);
    c->kda_dim = (int)js_int(d, js_get(d, lac, "head_dim"), 0);
    c->conv_k = (int)js_int(d, js_get(d, lac, "short_conv_kernel_size"), 4);
    memset(c->kda_layer, 0, sizeof c->kda_layer);
    int kl = js_get(d, lac, "kda_layers");
    for (int i = 0; i < js_size(d, kl); i++) {
        int v = (int)js_int(d, js_at(d, kl, i), -1) - 1;   /* list is 1-based */
        if (v >= 0 && v < 128) c->kda_layer[v] = 1;
    }

    c->arch_qwen = 0;
    c->ple_layer = -1;
    {
        char mt[40];
        js_str(d, js_get(d, cfg, "model_type"), mt, sizeof mt);
        if (strcmp(mt, "qwen4_exp_text") == 0 ||
            strstr(c->arch, "Qwen4Exp") != NULL)
            c->arch_qwen = 1;
    }
    if (!c->arch_qwen) return;

    /* Qwen is not Kimi: do not fill kda_layer from a missing linear_attn_config. */
    memset(c->kda_layer, 0, sizeof c->kda_layer);
    c->qsa_n_kv = (int)js_int(d, js_get(d, cfg, "num_key_value_heads"), 0);
    c->qsa_head_dim = (int)js_int(d, js_get(d, cfg, "head_dim"), 0);
    c->gdn_k_heads = (int)js_int(d, js_get(d, cfg, "linear_num_key_heads"), 0);
    c->gdn_v_heads = (int)js_int(d, js_get(d, cfg, "linear_num_value_heads"), 0);
    c->gdn_k_dim = (int)js_int(d, js_get(d, cfg, "linear_key_head_dim"), 0);
    c->gdn_v_dim = (int)js_int(d, js_get(d, cfg, "linear_value_head_dim"), 0);
    c->conv_k = (int)js_int(d, js_get(d, cfg, "linear_conv_kernel_dim"), 4);
    c->hc_count = (int)js_int(d, js_get(d, cfg, "hc_count"), 0);
    c->hc_lowrank = (int)js_int(d, js_get(d, cfg, "hc_lowrank"), 0);
    c->idx_n_heads = (int)js_int(d, js_get(d, cfg, "indexer_n_heads"), 4);
    c->idx_kv_heads = (int)js_int(d, js_get(d, cfg, "indexer_kv_heads"), 1);
    c->idx_head_dim = (int)js_int(d, js_get(d, cfg, "indexer_head_dim"), 128);
    c->idx_budget = (int)js_int(d, js_get(d, cfg, "indexer_budget"), 2048);
    c->idx_compress = (int)js_int(d, js_get(d, cfg, "indexer_compress_ratio"), 4);
    c->ngram_size = (int)js_int(d, js_get(d, cfg, "ngram_size"), 3);
    c->heads_per_ngram = (int)js_int(d, js_get(d, cfg, "heads_per_ngram"), 8);
    c->ple_embed = (int)js_int(d, js_get(d, cfg, "ple_embed_dim"), 0);
    c->ple_conv_k = (int)js_int(d, js_get(d, cfg, "ple_conv_kernel_size"), 4);
    c->shared_inter = (int)js_int(d, js_get(d, cfg, "shared_expert_intermediate_size"),
                                  c->moe_inter);
    /* Not read from the config: Qwen4ExpTextTopKRouter renormalizes
     * unconditionally, so a container that happens to omit the key must
     * still renormalize. */
    c->renorm = 1;
    {
        const int lt = js_get(d, cfg, "layer_types");
        memset(c->qwen_full, 0, sizeof c->qwen_full);
        /* Kept so cfg_sane can insist on one entry per layer. All-zero is
         * a valid-looking answer that means "every layer is GDN", and a
         * container whose `layer_types` is missing or short would attend
         * with a recurrence on layers that need sparse attention — wrong
         * everywhere and diagnosable nowhere. */
        c->qwen_n_layer_types = js_size(d, lt);
        for (int i = 0; i < js_size(d, lt) && i < WASTE_MAX_LAYERS; i++) {
            char kind[32];
            js_str(d, js_at(d, lt, i), kind, sizeof kind);
            c->qwen_full[i] = (strcmp(kind, "full_attention") == 0);
        }
    }
    {
        const int ids = js_get(d, cfg, "ple_layer_ids");
        if (js_size(d, ids) > 0) {
            const int one = (int)js_int(d, js_at(d, ids, 0), 0);
            c->ple_layer = one > 0 ? one - 1 : -1;
        }
    }
    {
        const int off = js_get(d, cfg, "ple_head_offsets");
        const int sz = js_get(d, cfg, "ple_head_vocab_sizes");
        const int mul = js_get(d, cfg, "ple_layer_multipliers");
        for (int h = 0; h < WASTE_QWEN_PLE_HEADS; h++) {
            c->ple_off[h] = js_int(d, js_at(d, off, h), 0);
            c->ple_sz[h] = js_int(d, js_at(d, sz, h), 0);
        }
        for (int i = 0; i < 8; i++)
            c->ple_mult[i] = js_int(d, js_at(d, mul, i), 0);
    }
    {
        const int rp = js_get(d, cfg, "rope_parameters");
        const double pf = js_num(d, js_get(d, rp, "partial_rotary_factor"),
                                 js_num(d, js_get(d, cfg, "partial_rotary_factor"), 0.25));
        const int hd = c->qsa_head_dim ? c->qsa_head_dim : 256;
        c->rotary_dim = (int)(hd * pf);
        const int sec = js_get(d, rp, "mrope_section");
        c->mrope_section[0] = (int)js_int(d, js_at(d, sec, 0), 11);
        c->mrope_section[1] = (int)js_int(d, js_at(d, sec, 1), 11);
        c->mrope_section[2] = (int)js_int(d, js_at(d, sec, 2), 10);
        const double base = js_num(d, js_get(d, rp, "rope_theta"),
                                   js_num(d, js_get(d, cfg, "rope_theta"), 10000000.0));
        const int half = c->rotary_dim / 2;
        if (half > 0 && half <= WASTE_MAX_ROPE_HALF) {
            for (int j = 0; j < half; j++)
                c->rope_inv_freq[j] = (float)(1.0 / pow(base, (double)(2 * j) / c->rotary_dim));
            c->rope_err[0] = 0;
        }
    }
}

/* Defined below, next to record_check; the cache needs it at load. */
static int bank_fetch(void *user, int layer, int expert, uint8_t *dst);
static void start_readers(waste_model *m);
static void start_fill(waste_model *m);
static void stop_fill(waste_model *m);

/* Wire the resident trunk. The cache is the cold part of this engine —
 * 19 to 30% hit — and this is the hot one: 27.5 GB on K3, read in full
 * every token, so it is the eviction that costs most. LEARNED.md §30
 * measured wiring the cache and found it bought reproducibility rather
 * than speed; this is the other half of that experiment.
 *
 * Sizes are derived here rather than recorded at each malloc, because the
 * trunk is allocated in four places in load_trunk and a fifth would be a
 * place to forget. A tensor left on disk has nothing to wire. */
static void wire_trunk(waste_model *m)
{
    size_t done = 0;
    int failed = 0;
    for (int i = 0; i < m->n_tensors; i++) {
        const waste_tensor *t = &m->t[i];
        if (t->on_disk || t->ndim < 1) continue;
        if (t->data) {
            const size_t n = t->n * sizeof(float);
            if (waste_wire(t->data, n)) done += n; else failed++;
        }
        const int N = t->shape[t->ndim - 1];
        if (!t->q || N <= 0 || t->group <= 0) continue;
        const size_t rows = t->n / (size_t)N;
        const size_t payload = rows * t->rowbytes;
        if (waste_wire(t->q, payload)) done += payload; else failed++;
        if (t->qs) {
            const size_t sb = rows * (size_t)((N + t->group - 1) / t->group) *
                              sizeof(uint16_t);
            if (waste_wire(t->qs, sb)) done += sb; else failed++;
        }
    }
    fprintf(stderr, "waste: wired %.2f GB of trunk%s\n",
            (double)done / (double)(1u << 30),
            failed ? " (some buffers were refused)" : "");
}

int waste_model_load(waste_model *m, const char *dir, int kv_cap,
                     const waste_load_opts *opt)
{
    /* Named rather than positional: this used to be `{ 0, 0, 0, 0, 1 }`,
     * and a field added in the middle of the struct silently moved the 1
     * from direct_io onto cache_policy. */
    static const waste_load_opts defaults = { .direct_io = 1 };
    if (!opt) opt = &defaults;
    const size_t cache_bytes = opt->cache_bytes;
    memset(m, 0, sizeof *m);
    pthread_mutex_init(&m->fetch_mu, NULL);
    m->trunk_fd = -1;
    for (int L = 0; L < WASTE_MAX_LAYERS; L++) {
        for (int s = 0; s < WASTE_MAX_SHARDS; s++) m->bank[L].fd[s] = -1;
        m->bank[L].n_shards = 1;
    }
    m->want_vision = opt->want_vision;
    m->want_direct = opt->direct_io;
    pthread_once(&model_opts_once, model_opts_init);
    m->kv_cap = kv_cap;
    m->direct_io = 1;
    waste_backend_init(WASTE_BE_AUTO);
    {
        /* An explicit count is the caller's decision and wins; the
         * environment is the escape hatch for when there is no caller to
         * ask, so it only fills in a 0. The pool is process-wide and
         * initialises once — the first model in wins, which is why the
         * header says so. */
        const char *e = getenv("WASTE_THREADS");
        /* Same rule for the cpuset, and the same escape hatch. waste_open
         * already refused a list that is malformed or that this platform
         * cannot bind, so anything but OK here means the caller came in
         * through waste_model_load directly: leave placement to the OS
         * rather than guess at what half of a bad list meant. */
        waste_cpumask cpus;
        const int cr = waste_cpus_resolve(opt->cpus, &cpus);
        waste_pool_init(opt->n_threads > 0 ? opt->n_threads
                                           : (e ? atoi(e) : 0),
                        cr == WASTE_CPUS_OK ? &cpus : NULL);
    }

    char path[MAXP];
    snprintf(path, sizeof path, "%s/manifest.json", dir);
    char *src = slurp(path, NULL);
    if (!src) return -1;
    js_doc d;
    if (js_parse(&d, src) < 0) { free(src); return -1; }

    /* Refuse a container this build does not know how to read.
     *
     * The field has been written since the first converter and read by
     * nobody, which is the dangerous shape: the day the layout changes, an
     * old engine would parse a new container against the old rules and
     * produce plausible wrong numbers instead of an error. Accept only the
     * versions listed here, and reject a manifest with no version at all —
     * that is either not a WASTE container or predates the guarantee. */
    {
        const int fv_tok = js_get(&d, 0, "format_version");
        const long fv = fv_tok >= 0 ? js_int(&d, fv_tok, -1) : -1;
        if (fv != WASTE_FORMAT_VERSION) {
            fprintf(stderr,
                    "waste: container format version %s, this build reads %d\n",
                    fv_tok >= 0 ? "mismatch" : "missing", WASTE_FORMAT_VERSION);
            if (fv_tok >= 0)
                fprintf(stderr, "waste: manifest says %ld\n", fv);
            js_free(&d); free(src);
            return -2;                       /* -> WASTE_E_FORMAT */
        }
    }

    {
        int cfg = js_get(&d, 0, "config");
        /* The converter flattens K3's nested text_config into `config` and
         * records the tensor-name prefix separately, so read that — probing
         * for a nested text_config here finds nothing and silently leaves
         * every tensor lookup one prefix short. */
        js_str(&d, js_get(&d, 0, "tensor_prefix"), m->cfg.prefix,
               sizeof m->cfg.prefix);
        const int tc = js_get(&d, cfg, "text_config");   /* raw HF config */
        if (tc >= 0) {
            cfg = tc;
            if (!m->cfg.prefix[0])
                snprintf(m->cfg.prefix, sizeof m->cfg.prefix, "language_model.");
        }
        cfg_from_json(&m->cfg, &d, cfg);
        /* Qwen containers are accepted once the kernels and planner exist.
         * Kimi still never sees Qwen tensors: arch_qwen selects a distinct
         * forward, not KDA/MLA/AttnRes. */
    }
    if (!cfg_sane(&m->cfg)) {
        fprintf(stderr, "waste: manifest config is out of range "
                        "(%d layers, hidden %d, vocab %d, %d experts top-%d)\n",
                m->cfg.n_layers, m->cfg.hidden, m->cfg.vocab,
                m->cfg.n_experts, m->cfg.top_k);
        js_free(&d); free(src);
        return -2;                        /* -> WASTE_E_FORMAT */
    }
    /* rope_init leaves no table for a shape it does not implement. Running
     * anyway would apply no rotation, which is not a degraded result but an
     * unordered one, so refuse instead. */
    if (m->cfg.rope_err[0]) {
        fprintf(stderr, "waste: %s\n", m->cfg.rope_err);
        js_free(&d); free(src);
        return -2;                        /* -> WASTE_E_FORMAT */
    }
    const waste_config *c = &m->cfg;

    int eq = js_get(&d, 0, "expert_quant");
    m->stages = (int)js_int(&d, js_get(&d, eq, "stages"), 3);
    m->vec_dim = (int)js_int(&d, js_get(&d, eq, "vec_dim"), 8);
    m->cb_entries = (int)js_int(&d, js_get(&d, eq, "entries"), 256);
    /* Absent in every v0 container written before VQ4P, and 8 is what those
     * mean: one whole byte of index per stage. */
    m->index_bits = (int)js_int(&d, js_get(&d, eq, "index_bits"), 8);
    /* cfg_sane covers `config`; this block is the other half of the
     * manifest and nothing checked it. All three size the LUT and two of
     * them divide: `"vec_dim": 0` was a division by zero in three places,
     * silent on ARM (which yields 0) and a SIGFPE on x86, which is to say
     * on Linux and Windows. The codebook index is a byte, so entries
     * above 256 could never be addressed anyway. */
    if (m->stages < 1 || m->stages > 8 ||
        m->vec_dim < 1 || m->vec_dim > 64 ||
        m->cb_entries < 1 || m->cb_entries > 256) {
        fprintf(stderr, "waste: manifest expert_quant is out of range "
                        "(%d stages, vec_dim %d, %d entries)\n",
                m->stages, m->vec_dim, m->cb_entries);
        js_free(&d); free(src);
        return -2;                        /* -> WASTE_E_FORMAT */
    }
    /* The packing and its unpack are 4x6-into-3 and nothing else. Refuse
     * any other combination here rather than let a mis-declared container
     * reach a kernel that would read the right number of bytes and decode
     * the wrong indices out of them. */
    if (m->index_bits != 8 &&
        (m->index_bits != 6 || m->stages != 4 || m->cb_entries != 64)) {
        fprintf(stderr, "waste: manifest expert_quant index_bits %d is only "
                        "supported as 6 with 4 stages and 64 entries "
                        "(got %d stages, %d entries)\n",
                m->index_bits, m->stages, m->cb_entries);
        js_free(&d); free(src);
        return -2;                        /* -> WASTE_E_FORMAT */
    }

    if (load_trunk(m, dir, &d, js_get(&d, 0, "trunk")) < 0) { js_free(&d); free(src); return -1; }

    /* vision config, when the container carries a tower */
    {
        snprintf(path, sizeof path, "%s/vision.json", dir);
        char *vs = slurp(path, NULL);
        if (vs) {
            js_doc vd;
            if (js_parse(&vd, vs) >= 0) {
                waste_vision_cfg *v = &m->vcfg;
                {   /* GLM states its tower under the HF names, K3 under
                     * the converter's `vt_` ones. Read the tower first and
                     * then the geometry it implies, rather than defaulting
                     * every field twice. */
                    char tw[32];
                    js_str(&vd, js_get(&vd, 0, "tower"), tw, sizeof tw);
                    v->tower = strcmp(tw, "glm5-next") == 0
                             ? WASTE_TOWER_GLM : WASTE_TOWER_K3;
                }
                if (v->tower == WASTE_TOWER_GLM) {
                    v->hidden      = (int)js_int(&vd, js_get(&vd, 0, "hidden_size"), 1024);
                    v->heads       = (int)js_int(&vd, js_get(&vd, 0, "num_heads"), 16);
                    v->qkv_hidden  = v->hidden;      /* qkv is 3x hidden   */
                    v->inter       = (int)js_int(&vd, js_get(&vd, 0, "intermediate_size"), 4096);
                    v->layers      = (int)js_int(&vd, js_get(&vd, 0, "depth"), 24);
                    v->merge       = (int)js_int(&vd, js_get(&vd, 0, "spatial_merge_size"), 2);
                    v->temporal    = (int)js_int(&vd, js_get(&vd, 0, "temporal_patch_size"), 2);
                    v->out_hidden  = (int)js_int(&vd, js_get(&vd, 0, "out_hidden_size"), c->hidden);
                    v->proj_inter  = (int)js_int(&vd, js_get(&vd, 0, "projection_intermediate_size"), 10240);
                    v->swiglu_limit = (float)js_num(&vd, js_get(&vd, 0, "swiglu_limit"), 0.0);
                    v->img_start   = (int)js_int(&vd, js_get(&vd, 0, "image_start_token_id"), -1);
                    v->img_end     = (int)js_int(&vd, js_get(&vd, 0, "image_end_token_id"), -1);
                    v->text_hidden = v->out_hidden;
                    v->patch       = (int)js_int(&vd, js_get(&vd, 0, "patch_size"), 14);
                    /* Its norms are Glm5NextRMSNorm(dim, eps=rms_norm_eps),
                     * so unlike K3's the eps is stated and is not float
                     * epsilon. The merger's is a LayerNorm at torch's
                     * default. */
                    v->eps         = (float)js_num(&vd, js_get(&vd, 0, "rms_norm_eps"), 1e-5);
                    v->proj_eps    = 1e-5f;
                    /* pos_h/pos_w belong to K3's learned grid and are not
                     * read; keep them in range for cfg_sane. */
                    v->pos_h = v->pos_w = 1;
                    v->media_token = (int)js_int(&vd, js_get(&vd, 0,
                                         "media_placeholder_token_id"), -1);
                    v->max_patches = (int)js_int(&vd, js_get(&vd, 0,
                                         "max_patches"), 1024);
                    v->min_tokens  = (int)js_int(&vd, js_get(&vd, 0,
                                         "min_image_tokens"), 1);
                    goto vision_pixels;
                }
                v->hidden      = (int)js_int(&vd, js_get(&vd, 0, "vt_hidden_size"), 1024);
                v->heads       = (int)js_int(&vd, js_get(&vd, 0, "vt_num_attention_heads"), 12);
                v->qkv_hidden  = (int)js_int(&vd, js_get(&vd, 0, "qkv_hidden_size"), 1536);
                v->inter       = (int)js_int(&vd, js_get(&vd, 0, "vt_intermediate_size"), 4096);
                v->layers      = (int)js_int(&vd, js_get(&vd, 0, "vt_num_hidden_layers"), 27);
                v->pos_h       = (int)js_int(&vd, js_get(&vd, 0, "init_pos_emb_height"), 64);
                v->pos_w       = (int)js_int(&vd, js_get(&vd, 0, "init_pos_emb_width"), 64);
                v->text_hidden = (int)js_int(&vd, js_get(&vd, 0, "text_hidden_size"), c->hidden);
                v->patch       = (int)js_int(&vd, js_get(&vd, 0, "patch_size"), 14);
                /* the tower's RMSNorms are built as nn.RMSNorm(dim) with no
                 * eps, so PyTorch uses finfo(float32).eps */
                v->eps         = 1.1920928955078125e-07f;
                v->proj_eps    = 1e-5f;
                v->media_token = (int)js_int(&vd, js_get(&vd, 0,
                                     "media_placeholder_token_id"), -1);
                v->max_patches = (int)js_int(&vd, js_get(&vd, 0,
                                     "max_patches"), 1024);
            vision_pixels:
                {   /* K3 normalizes to [-1, 1]: preprocessor_config.json
                     * carries mean = std = 0.5 under `media_proc_cfg`, and
                     * kimi_k3_vision_processing.py applies exactly those.
                     * These used to be the CLIP constants, on the belief
                     * that the release shipped no preprocessor config. It
                     * does. The converter writes the real values into
                     * vision.json, which overrides this; the fallback is
                     * for a container converted without the file present. */
                    static const float dm[3] = {0.5f, 0.5f, 0.5f};
                    static const float ds[3] = {0.5f, 0.5f, 0.5f};
                    const int mi = js_get(&vd, 0, "image_mean");
                    const int si = js_get(&vd, 0, "image_std");
                    for (int k = 0; k < 3; k++) {
                        v->mean[k] = mi >= 0 ? (float)js_num(&vd, js_at(&vd, mi, k), dm[k]) : dm[k];
                        v->std[k]  = si >= 0 ? (float)js_num(&vd, js_at(&vd, si, k), ds[k]) : ds[k];
                    }
                }
                js_free(&vd);
            }
            free(vs);
        }
    }
    if (m->want_vision && waste_find(m, "vision_tower.patch_embed.proj.weight")) {
        const waste_vision_cfg *v = &m->vcfg;
        int sane = v->hidden > 0 && v->hidden <= (1 << 20) &&
                   v->heads > 0 && v->heads <= (1 << 16) &&
                   v->qkv_hidden > 0 && v->qkv_hidden <= (1 << 20) &&
                   v->qkv_hidden % v->heads == 0 &&
                   v->inter > 0 && v->inter <= (1 << 22) &&
                   v->layers > 0 && v->layers <= 256 &&
                   v->pos_h > 0 && v->pos_h <= (1 << 16) &&
                   v->pos_w > 0 && v->pos_w <= (1 << 16) &&
                   v->text_hidden == c->hidden && v->patch == 14 &&
                   (v->tower != WASTE_TOWER_GLM ||
                    (v->merge >= 1 && v->merge <= 8 &&
                     v->temporal >= 1 && v->temporal <= 8 &&
                     v->out_hidden > 0 && v->out_hidden <= (1 << 20) &&
                     v->proj_inter > 0 && v->proj_inter <= (1 << 22) &&
                     v->hidden % v->heads == 0 &&
                     isfinite(v->swiglu_limit) && v->swiglu_limit >= 0.0f)) &&
                   v->media_token >= 0 && v->media_token < c->vocab &&
                   v->max_patches >= 4 && v->max_patches <= (1 << 20) &&
                   isfinite(v->eps) && v->eps > 0.0f &&
                   isfinite(v->proj_eps) && v->proj_eps > 0.0f;
        for (int k = 0; k < 3; k++)
            sane = sane && isfinite(v->mean[k]) && isfinite(v->std[k]) &&
                   v->std[k] > 0.0f;
        if (!sane) {
            fprintf(stderr, "waste: vision.json geometry is unsupported or out of range\n");
            js_free(&d); free(src);
            return -2;
        }
    }

    /* codebooks */
    snprintf(path, sizeof path, "%s/codebooks.bin", dir);
    size_t cblen;
    char *cb = slurp(path, &cblen);
    if (!cb) { js_free(&d); free(src); return -1; }
    const size_t rec = 16 + (size_t)m->cb_entries * m->vec_dim * 2;
    m->n_books = (int)(cblen / rec);
    m->codebooks = (float *)malloc((size_t)m->n_books * m->cb_entries * m->vec_dim * sizeof(float));
    if (!m->codebooks) { free(cb); js_free(&d); free(src); return -1; }
    for (int b = 0; b < m->n_books; b++) {
        const uint16_t *h = (const uint16_t *)(cb + b * rec + 16);
        for (int i = 0; i < m->cb_entries * m->vec_dim; i++) {
            const uint16_t v = h[i];
            const uint32_t sign = (uint32_t)(v >> 15) << 31;
            uint32_t exp = (v >> 10) & 0x1f, man = v & 0x3ff, bits;
            bits = exp ? (sign | ((exp + 112u) << 23) | (man << 13)) : sign;
            memcpy(&m->codebooks[(size_t)b * m->cb_entries * m->vec_dim + i], &bits, 4);
        }
    }
    free(cb);
    /* Transposed copy, [book][dim][entry]. The LUT build wants
     * dst[c] = sum_d x[d]*C[c][d] for 256 entries at once; with the entries
     * innermost that is an axpy per dimension instead of 256 eight-element
     * dot products each ending in a cross-lane reduction. 6.5 MB on K3. */
    {
        const size_t per = (size_t)m->cb_entries * m->vec_dim;
        m->codebooksT = (float *)malloc((size_t)m->n_books * per * sizeof(float));
        if (!m->codebooksT) { js_free(&d); free(src); return -1; }
        for (int b = 0; b < m->n_books; b++) {
            const float *src_b = m->codebooks + (size_t)b * per;
            float *dst_b = m->codebooksT + (size_t)b * per;
            for (int c = 0; c < m->cb_entries; c++)
                for (int dim = 0; dim < m->vec_dim; dim++)
                    dst_b[(size_t)dim * m->cb_entries + c] = src_b[(size_t)c * m->vec_dim + dim];
        }
    }

    /* expert banks. The expert's input width is the latent one on a model
     * that has it (K3), and the hidden everywhere else — the same choice
     * moe_layer makes, and the one that decides how many per-channel
     * scales a record ends with. */
    {
        const int lat = c->latent_dim ? c->latent_dim : c->hidden;
        m->expert_m[0] = m->expert_m[1] = c->moe_inter;
        m->expert_n[0] = m->expert_n[1] = lat;
        m->expert_m[2] = lat; m->expert_n[2] = c->moe_inter;
    }
    /* Checksum verification is off unless asked for — it costs a pass over
     * every record on every miss, ~5% on Kimi-Linear. WASTE_VERIFY=1 turns
     * it on for a process; waste_cfg.verify_records turns it on for one
     * context, and waste.c sets this again after load. Either switch is
     * enough; neither can turn it off, because off is where it starts. */
    { const char *e = getenv("WASTE_VERIFY"); m->verify = e && *e != '0'; }
    int layers = js_get(&d, 0, "layers");
    for (int L = 0; L < c->n_layers; L++) {
        char key[16];
        snprintf(key, sizeof key, "%d", L);
        int e = js_get(&d, layers, key);
        const int expected = c->n_experts && L >= c->first_dense;
        if (e < 0) {
            if (expected) { js_free(&d); free(src); return -2; }
            continue;
        }
        char fn[64];
        js_str(&d, js_get(&d, e, "file"), fn, sizeof fn);
        snprintf(path, sizeof path, "%s/%s", dir, fn);
        m->bank[L].n_experts = (int)js_int(&d, js_get(&d, e, "experts"), 0);
        m->bank[L].cb_base = (int)js_int(&d, js_get(&d, e, "codebook_base"), 0);
        /* A layer's bank passes 2 GB well before K3's largest does, so the
         * division that produces rec_bytes has to happen in 64 bits. */
        const int64_t bytes = js_int(&d, js_get(&d, e, "bytes"), 0);
        if (!expected || m->bank[L].n_experts != c->n_experts || bytes <= 0 ||
            bytes % m->bank[L].n_experts != 0 || m->bank[L].cb_base < 0 ||
            3 * m->stages > m->n_books ||
            m->bank[L].cb_base > m->n_books - 3 * m->stages) {
            js_free(&d); free(src); return -2;
        }
        m->bank[L].rec_bytes = m->bank[L].n_experts ? bytes / m->bank[L].n_experts : 0;
        /* Unstriped open is the N=1 case of the striped one: one fd on the
         * bank's own file. A WASTE_BANK_SHARDS manifest (comma-separated
         * directories) reopens the bank as N shard files named after the
         * bank's basename in each directory.
         *
         * Every shard must hold its whole share of experts, and that is
         * checked here rather than discovered later: one WASTE_ALIGN read at
         * the last record's offset, which is 4 KiB-aligned like every record
         * and so legal under O_DIRECT. A shard short by whole records — an
         * interrupted copy, the ordinary way a split goes wrong — refuses
         * the load instead of failing the first time the router happens to
         * pick a high-numbered expert, which on K3 can be thousands of
         * tokens in. A shard truncated *inside* its last record still fails
         * at that record's read, REC_E_READ, and a wrong N is caught by the
         * expert id in the record header: neither can be served as a
         * different expert's weights. */
        {
            const char *sh = getenv("WASTE_BANK_SHARDS");
            char dirs[1024];
            int n_sh = 1;
            if (sh && *sh) {
                /* Truncating the list would silently drop shards, and a
                 * count that disagrees with the split that produced them is
                 * a different layout, not a smaller one. */
                if ((size_t)snprintf(dirs, sizeof dirs, "%s", sh) >= sizeof dirs) {
                    js_free(&d); free(src); return -2;
                }
                n_sh = 1;
                for (const char *p = dirs; *p; p++) if (*p == ',') n_sh++;
                if (n_sh > WASTE_MAX_SHARDS) { js_free(&d); free(src); return -2; }
            }
            const char *base = strrchr(fn, '/'); base = base ? base + 1 : fn;
            for (int s = 0; s < n_sh; s++) {
                char spath[1152];
                if (n_sh == 1)
                    snprintf(spath, sizeof spath, "%s/%s", dir, fn);
                else {
                    char *dstart = dirs;
                    for (int k = 0; k < s && dstart; k++) {
                        dstart = strchr(dstart, ',');
                        if (dstart) dstart++;
                    }
                    if (!dstart || !*dstart) { js_free(&d); free(src); return -2; }
                    char *comma = strchr(dstart, ',');
                    size_t dlen = comma ? (size_t)(comma - dstart) : strlen(dstart);
                    snprintf(spath, sizeof spath, "%.*s/%s", (int)dlen, dstart, base);
                }
                int sfd = bank_open(spath, m->bank[L].rec_bytes, m->want_direct,
                                    &m->direct_io);
                if (sfd < 0) { js_free(&d); free(src); return -1; }
                /* Experts round-robin, so shard s holds ids s, s+N, s+2N ...
                 * and that is ceil((n_experts - s) / N) records.
                 *
                 * Two things can be wrong with a shard, and they need
                 * different instruments. The size is the whole-file claim:
                 * a stale or mismatched split leaves a shard the layout
                 * does not fit. The manifest bound the offsets already;
                 * this bounds them against the file, and it needs no read,
                 * so it also covers containers whose record size is not a
                 * whole number of blocks — where the probe below
                 * deliberately skips.
                 *
                 * What it is *not* protecting against is a wrong record
                 * being served as the right one. Measured rather than
                 * assumed: two shards padded to the sizes N=2 expects but
                 * carrying a 3-way split's records refuse on the first
                 * read, "expert 7 of layer 1: record header is not what the
                 * bank index describes". Every record carries the expert it
                 * belongs to and record_check reads it, so a misplaced one
                 * is a refusal and never a substitution. The value here is
                 * *when*: a long shard loaded happily and generated correct
                 * output before this check, since the extra records are
                 * simply never read, and a short one waited for the router
                 * to reach the missing expert. Both are now a refusal at
                 * load, which is the difference between a split you can
                 * trust and one you have not disproved yet. */
                const int recs = (m->bank[L].n_experts - s + n_sh - 1) / n_sh;
                if (n_sh > 1 && recs > 0) {
                    const int64_t want_bytes =
                        (int64_t)recs * m->bank[L].rec_bytes;
                    const int64_t have_bytes = waste_file_size(sfd);
                    if (have_bytes < 0 || have_bytes != want_bytes) {
                        close(sfd);
                        js_free(&d); free(src); return -1;
                    }
                }
                /* The probe is the other half: a file can be exactly the
                 * right length and still not be readable the way the engine
                 * will read it.
                 *
                 * waste_dio_alloc, not a stack array. O_DIRECT rejects every
                 * transfer whose *buffer* is unaligned, not only its offset
                 * and length -- the rule bank_probe below is written for, and
                 * the reason it allocates rather than declaring. A stack
                 * buffer here passed on macOS, whose F_NOCACHE has no buffer
                 * requirement, and on the x86_64 CI runner, whose filesystem
                 * declines O_DIRECT and falls back to buffered. On
                 * linux-arm64, where the bypass is real, it turned every
                 * container load into an I/O error: 25 passed, 29 failed.
                 * Two platforms agreeing is not coverage when only the third
                 * exercises the path.
                 *
                 * Skipped unless a record is a whole number of blocks, since
                 * then the offset itself is one O_DIRECT would refuse, and a
                 * refusal the device made is not evidence about the shard. */
                if (recs > 0 && m->bank[L].rec_bytes >= (int64_t)WASTE_ALIGN &&
                    m->bank[L].rec_bytes % (int64_t)WASTE_ALIGN == 0) {
                    void *probe = waste_dio_alloc(WASTE_ALIGN);
                    if (!probe) { close(sfd); js_free(&d); free(src); return -1; }
                    const int64_t last = (int64_t)(recs - 1) * m->bank[L].rec_bytes;
                    const int64_t got = waste_pread(sfd, probe, WASTE_ALIGN, last);
                    waste_dio_free(probe);
                    if (got != (int64_t)WASTE_ALIGN) {
                        close(sfd);
                        js_free(&d); free(src); return -1;
                    }
                }
                m->bank[L].fd[s] = sfd;
            }
            m->bank[L].n_shards = n_sh;
        }
        if (m->bank[L].fd[0] < 0) { js_free(&d); free(src); return -1; }
    }
    js_free(&d);
    free(src);

    if (!validate_text_tensors(m)) return -2;      /* -> WASTE_E_FORMAT */

    /* state + scratch */
    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    /* Pools a full context can produce. The +1 is the pool the last token
     * closes: kv_cap tokens make exactly kv_cap/kpool of them, and rounding
     * up costs one vector and removes a bound to get wrong. */
    m->pool_cap = c->index_kpool ? kv_cap / c->index_kpool + 1 : 0;
    if (c->arch_qwen) {
        const int Hv = c->gdn_v_heads, Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
        const int Hk = c->gdn_k_heads;
        const int qkv = 2 * Hk * Dk + Hv * Dv;
        const int nkv = c->qsa_n_kv, hd = c->qsa_head_dim;
        const int idim = c->idx_head_dim;
        const int compress = c->idx_compress > 0 ? c->idx_compress : 4;
        const int nblk = compress > 0 ? (kv_cap + compress - 1) / compress : 0;
        const int max_sel = c->idx_budget + compress;
        const int rot = c->rotary_dim > 0 ? c->rotary_dim : 1;
        for (int L = 0; L < c->n_layers; L++) {
            if (!c->qwen_full[L]) {
                m->S[L] = (float *)calloc((size_t)Hv * Dk * Dv, sizeof(float));
                m->conv[L] = (float *)calloc((size_t)qkv * (c->conv_k > 0 ? c->conv_k - 1 : 0),
                                             sizeof(float));
            } else {
                m->has_qsa = 1;
                m->qsa_k[L] = (uint16_t *)calloc((size_t)kv_cap * nkv * hd, 2);
                m->qsa_v[L] = (uint16_t *)calloc((size_t)kv_cap * nkv * hd, 2);
                m->qsa_rawk[L] = (float *)calloc((size_t)kv_cap * idim, sizeof(float));
            }
        }
        m->hcx = (float *)calloc((size_t)c->hc_count * c->hidden, sizeof(float));
        {
            const int R = (c->ple_conv_k > 1 && c->ngram_size > 0)
                ? (c->ple_conv_k - 1) * c->ngram_size : 0;
            m->ple_ring = (float *)calloc((size_t)c->hc_count * c->hidden * (R > 0 ? R : 1),
                                          sizeof(float));
        }
        {
            const int pe = c->ple_embed ? c->ple_embed : c->hidden;
            m->ple_emb = (float *)calloc((size_t)(pe > 0 ? pe : 1), sizeof(float));
        }
        m->gdn_g = (float *)calloc((size_t)(Hv > 0 ? Hv : 1), sizeof(float));
        {
            const int qd = c->n_heads * hd;
            m->qsa_q = (float *)calloc((size_t)(qd > 0 ? qd : 1), sizeof(float));
            m->qsa_gate = (float *)calloc((size_t)(qd > 0 ? qd : 1), sizeof(float));
            m->qsa_attn = (float *)calloc((size_t)(qd > 0 ? qd : 1), sizeof(float));
            const size_t compact = (size_t)(max_sel > 0 ? max_sel : 1) * (size_t)nkv * hd;
            m->qsa_kf = (float *)calloc(compact > 0 ? compact : 1, sizeof(float));
            m->qsa_vf = (float *)calloc(compact > 0 ? compact : 1, sizeof(float));
            m->qsa_scr = (float *)calloc((size_t)(max_sel > 0 ? max_sel : 1), sizeof(float));
            m->qsa_sel = (int *)calloc((size_t)(max_sel > 0 ? max_sel : 1), sizeof(int));
            const size_t work = (size_t)nblk * idim + (size_t)nblk + (size_t)idim;
            m->qsa_work = (float *)calloc(work > 0 ? work : 1, sizeof(float));
            m->qsa_taken = (int *)calloc((size_t)(nblk > 0 ? nblk : 1), sizeof(int));
            m->qsa_cs = (float *)calloc((size_t)2 * kv_cap * rot, sizeof(float));
        }
        m->moe_prob = (float *)calloc((size_t)(c->n_experts > 0 ? c->n_experts : 1),
                                      sizeof(float));
        m->moe_used = (uint8_t *)calloc((size_t)(c->n_experts > 0 ? c->n_experts : 1), 1);
        for (int i = 0; i < 8; i++) m->ple_prev[i] = c->eos_token_id;
    } else {
        for (int L = 0; L < c->n_layers; L++) {
            if (c->kda_layer[L]) {
                m->S[L] = (float *)calloc((size_t)H * D * D, sizeof(float));
                m->conv[L] = (float *)calloc((size_t)3 * C * (c->conv_k - 1), sizeof(float));
            } else {
                m->has_mla = 1;      /* this is what makes kv_cap a real bound */
                m->latcache[L] = (float *)calloc(
                    (size_t)kv_cap * (c->kv_lora + c->qk_rope), sizeof(float));
                if (c->index_topk) {
                    m->idxpool[L] = (float *)calloc(
                        (size_t)m->pool_cap * c->index_dim, sizeof(float));
                    m->idxbuf[L] = (float *)calloc(
                        (size_t)c->index_kpool * 2 * c->index_dim, sizeof(float));
                    if (!m->idxpool[L] || !m->idxbuf[L]) return -1;
                }
            }
        }
    }
    if (c->index_topk && m->has_mla) {
        /* One selection at a time, shared by every layer: it is consumed by
         * the head loop before the next layer records anything. */
        m->idxsel = (int *)calloc((size_t)c->index_topk + c->index_kpool,
                                  sizeof(int));
        m->idxscore = (float *)calloc((size_t)m->pool_cap, sizeof(float));
        m->idxrank = (int *)calloc((size_t)m->pool_cap, sizeof(int));
        m->idxq = (float *)calloc((size_t)c->index_heads * c->index_dim +
                                  c->index_heads, sizeof(float));
        if (!m->idxsel || !m->idxscore || !m->idxrank || !m->idxq) return -1;
    }
    int big = c->hidden > C ? c->hidden : C;
    if (c->arch_qwen) {
        const int qkv = 2 * c->gdn_k_heads * c->gdn_k_dim +
                        c->gdn_v_heads * c->gdn_v_dim;
        const int hcH = c->hc_count * c->hidden;
        const int qsa = c->n_heads * c->qsa_head_dim * 2;
        if (qkv > big) big = qkv;
        if (hcH > big) big = hcH;
        if (qsa > big) big = qsa;
    }
    /* mHC keeps hc_mult residual streams instead of one. Every other user
     * of m->x reads stream 0, which is where the single-stream models put
     * the whole thing, so the multiplier is confined to this allocation
     * and to the three places that walk the buffer whole. */
    const int hc = c->hc_mult ? c->hc_mult : 1;
    m->x = (float *)calloc((size_t)hc * c->hidden, sizeof(float));
    if (c->hc_mult) {
        const int nmix = (2 + c->hc_mult) * c->hc_mult;
        m->hcflat = (float *)calloc((size_t)hc * c->hidden, sizeof(float));
        m->hccol  = (float *)calloc((size_t)c->hidden, sizeof(float));
        /* the mapping's outputs, then the `pre` weights hc_collapse keeps
         * past them rather than on its caller's stack */
        m->hcmix  = (float *)calloc((size_t)nmix + c->hc_mult, sizeof(float));
        if (!m->hcflat || !m->hccol || !m->hcmix) return -1;
    }
    m->h = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->tmp = (float *)calloc((size_t)8 * big + 8 * c->moe_inter + 8 * c->dense_inter
                             + (size_t)4 * c->n_heads * (c->v_head + c->qk_nope + c->qk_rope
                                                         + c->qsa_head_dim)
                             + (size_t)2 * (c->q_lora ? c->q_lora : 1) + 256,
                             sizeof(float));
    /* Sized for every user of the buffer, not just the one it is named
     * after — see WASTE_ATT_ROUTER_OFF in model.h. */
    {
        size_t need = (size_t)kv_cap * (size_t)c->n_heads;   /* MLA/QSA scores */
        const size_t kda = (size_t)c->kda_heads * (size_t)c->kda_dim;
        const size_t gdn = (size_t)c->gdn_v_heads * (size_t)c->gdn_k_dim;
        const size_t route = WASTE_ATT_ROUTER_OFF + 2u * (size_t)c->n_experts;
        if (kda > need) need = kda;
        if (gdn > need) need = gdn;
        if (route > need) need = route;
        m->att = (float *)calloc(need + 1024, sizeof(float));
    }
    {   /* int8 activations for the SMMLA batched matmul: a full chunk of
         * the widest input the trunk has (the dense FFN's 33792) */
        int widest = c->dense_inter > c->hidden ? c->dense_inter : c->hidden;
        /* Same bound as m->xq below, for the same reason: the batched
         * matmul quantizes rows of whatever width it is handed. */
        if ((int64_t)(c->hc_mult ? c->hc_mult : 1) * c->hidden > widest)
            widest = (c->hc_mult ? c->hc_mult : 1) * c->hidden;
        m->mmx_cap = (size_t)WASTE_CHUNK_MAX * widest;
        m->mms_cap = (size_t)WASTE_CHUNK_MAX * ((widest + 127) / 128 + 1);
        m->mmxq = (int8_t *)malloc(m->mmx_cap);
        m->mmxs = (float *)malloc(m->mms_cap * sizeof(float));
    }
    {   /* one row of whichever tensors were left on disk */
        size_t rb = 0, sb = 0;
        for (int i = 0; i < m->n_tensors; i++) {
            const waste_tensor *t = &m->t[i];
            /* group is 0 on the tensors the loader skipped — the vision
             * tower, and anything outside cfg.prefix — because that skip
             * sets on_disk and continues before the quantized branch
             * assigns it. Dividing by it is undefined, and the architecture
             * decides what that means: arm64's sdiv quietly yields 0, x86's
             * idiv raises #DE, so `waste info` on K3 was an instant SIGFPE
             * on every x86 build while every check here stayed green.
             * Skipping is also what those tensors want — rowbytes 0 and no
             * scale rows, so no row scratch to size. issue #10. */
            if (!t->on_disk || !t->group) continue;
            const int ng = (t->shape[t->ndim - 1] + t->group - 1) / t->group;
            if (t->rowbytes > rb) rb = t->rowbytes;
            if ((size_t)ng > sb) sb = (size_t)ng;
        }
        m->embrow = rb ? (int8_t *)malloc(rb) : NULL;
        m->embsc  = sb ? (uint16_t *)malloc(sb * sizeof(uint16_t)) : NULL;
    }
    {   /* MLA absorption scratch: one q~, one accumulated latent and one
         * dequantized weight row per head, so the head loop needs no locks */
        const size_t n = (size_t)c->n_heads * (c->kv_lora ? c->kv_lora : 1);
        m->qabs = (float *)calloc(n, sizeof(float));
        m->cacc = (float *)calloc(n, sizeof(float));
        m->mrow = (float *)calloc(n, sizeof(float));
    }
    m->logits = (float *)calloc((size_t)c->vocab, sizeof(float));
    m->ff = (float *)calloc((size_t)2 * (c->dense_inter > c->moe_inter ? c->dense_inter : c->moe_inter), sizeof(float));
    m->e_gate = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_up = (float *)malloc((size_t)c->moe_inter * c->hidden * sizeof(float));
    m->e_down = (float *)malloc((size_t)c->hidden * c->moe_inter * sizeof(float));
    {   /* AttnRes history + the latent-MoE staging buffers */
        const int nb = c->attn_res_block ? c->n_layers / c->attn_res_block + 2 : 0;
        m->blockres = nb ? (float *)calloc((size_t)nb * c->hidden, sizeof(float)) : NULL;
        m->prefix_sum = (float *)calloc((size_t)c->hidden, sizeof(float));
        m->ares = (float *)calloc((size_t)(2 * c->hidden + 2 * (c->latent_dim
                                  ? c->latent_dim : c->hidden)), sizeof(float));
    }
    {
        int nmax = c->hidden > c->dense_inter ? c->hidden : c->dense_inter;
        /* mHC's mapping reads the whole flattened stream vector, which is
         * hc_mult times the hidden size and on GLM the widest quantized
         * matvec in the model: 16384 against the dense FFN's 12288. Sized
         * from the FFN alone this buffer is 4096 activations short of what
         * hc_collapse quantizes into it, and the 1024 bytes of slack below
         * hide that at test scale and not at model scale. Qwen's
         * HyperConnection mix reads the same shape under its own key. */
        const int64_t hcw = (int64_t)(c->hc_mult ? c->hc_mult :
                                      c->arch_qwen && c->hc_count ? c->hc_count : 1)
                          * c->hidden;
        if (hcw > nmax) nmax = (int)hcw;
        /* Two bytes per activation: the i8mm path writes two int8 planes
         * and the SMLAL path writes int16, both over the padded group
         * count rather than over `in`. */
        m->xq = (int8_t *)calloc((size_t)nmax * 2 + 1024, 1);
        m->xs = (float *)calloc((size_t)nmax / 32 + 64, sizeof(float));
    }

    {   /* LUT: [max_nv][stages][entries], three of them (gate, up, down).
         * Sized from the container's own vec_dim/stages/entries — it used
         * to divide by a literal 8 and so was only correct for the one
         * vec_dim the converter happens to write. */
        const int lt = c->latent_dim ? c->latent_dim : c->hidden;
        int nmax = c->hidden > c->moe_inter ? c->hidden : c->moe_inter;
        if (lt > nmax) nmax = lt;
        const size_t nvmax = (size_t)(nmax / m->vec_dim + 1);
        /* Page-aligned: a device backend wraps these with no copy, and
         * newBufferWithBytesNoCopy refuses anything else. */
        m->lut_bytes = (size_t)3 * nvmax * m->stages * m->cb_entries * sizeof(float);
        m->lut = (float *)waste_dio_alloc((size_t)3 * nvmax *
                                 m->stages * m->cb_entries * sizeof(float));
        /* An int8 shadow of m->lut, region for region, filled by
         * vq_build_lut. Same element count as the float table and a
         * quarter of the bytes. */
        const size_t reg = nvmax * (size_t)m->stages * m->cb_entries;
        const size_t nsc = nvmax / WASTE_VQ_LUT_BLK + 2;
        /* The int8 shadow of the table. VQ4P needs it; VQ3R uses it only
         * under WASTE_VQ8, and the allocation is 1 MB on K3, so it is
         * simpler to have it than to make every consumer test twice. */
        if (m->index_bits == 6 || vq8_on) {
            m->lut8 = (int8_t *)malloc(3 * reg);
            m->lut8_scale = (float *)malloc(3 * nsc * sizeof(float));
        }
        /* Per-expert slices for the expert-parallel path. Declining the
         * allocation is not fatal: moe_layer falls back to the row-parallel
         * loop, which needs none of this. */
        const int K = c->top_k;
        if (K > 1 && K <= WASTE_PF_MAX) {
            const int lt2 = c->latent_dim ? c->latent_dim : c->hidden;
            m->xlut_sz = reg;
            m->xnsc = nsc;
            m->xga  = (float *)malloc((size_t)K * c->moe_inter * sizeof(float));
            m->xub  = (float *)malloc((size_t)K * c->moe_inter * sizeof(float));
            m->xacc = (float *)malloc((size_t)K * lt2 * sizeof(float));
            m->xlut = (float *)waste_dio_alloc((size_t)K * reg * sizeof(float));
            if (m->index_bits == 6 || vq8_on) {
                m->xlut8 = (int8_t *)malloc((size_t)K * reg);
                m->xqs = (float *)malloc((size_t)K * nsc * sizeof(float));
            }
            if (!m->xga || !m->xub || !m->xacc || !m->xlut ||
                ((m->index_bits == 6 || vq8_on) && (!m->xlut8 || !m->xqs))) {
                free(m->xga); free(m->xub); free(m->xacc);
                free(m->xlut); free(m->xlut8); free(m->xqs);
                m->xga = m->xub = m->xacc = m->xlut = m->xqs = NULL;
                m->xlut8 = NULL;
            }
        }
    }
    {   /* expert cache, sized by the caller's budget */
        int64_t rec = 0;
        for (int L = 0; L < c->n_layers; L++)
            if (m->bank[L].rec_bytes > rec) rec = m->bank[L].rec_bytes;
        if (waste_ecache_init(&m->cache, cache_bytes, (size_t)rec, opt->policy))
            return -1;
        m->miss_buf = (uint8_t *)waste_dio_alloc((size_t)rec);
        if (!m->miss_buf) return -1;

        start_readers(m);
    }
    if (waste_mlock_mode() & WASTE_WIRE_TRUNK) wire_trunk(m);

    /* Every buffer the forward pass dereferences without asking, not the
     * five that used to be listed here — a NULL m->tmp or m->att is a
     * crash on the first token rather than a load failure. The per-layer
     * state arrays are checked in the same breath. */
    if (!m->x || !m->h || !m->tmp || !m->att || !m->logits || !m->ff ||
        !m->e_gate || !m->e_up || !m->e_down || !m->lut ||
        !m->xq || !m->xs || !m->qabs || !m->cacc || !m->mrow ||
        !m->prefix_sum || !m->ares || !m->mmxq || !m->mmxs)
        return -1;
    if (m->index_bits == 6 && (!m->lut8 || !m->lut8_scale)) return -1;
    for (int L = 0; L < c->n_layers; L++) {
        if (c->arch_qwen) {
            if (!c->qwen_full[L]) {
                if (!m->S[L] || !m->conv[L]) return -1;
            } else if (!m->qsa_k[L] || !m->qsa_v[L] || !m->qsa_rawk[L]) {
                return -1;
            }
        } else if (c->kda_layer[L]) {
            if (!m->S[L] || !m->conv[L]) return -1;
        } else if (!m->latcache[L]) {
            return -1;
        }
    }
    if (c->arch_qwen && (!m->hcx || !m->ple_ring || !m->ple_emb || !m->gdn_g ||
                         !m->qsa_q || !m->qsa_gate || !m->qsa_attn ||
                         !m->qsa_kf || !m->qsa_vf || !m->qsa_scr || !m->qsa_work ||
                         !m->qsa_cs || !m->qsa_sel || !m->qsa_taken ||
                         !m->moe_prob || !m->moe_used))
        return -1;
    if (c->attn_res_block && !m->blockres) return -1;
    /* Last, so a load that fails leaves no thread reading a model nobody
     * owns — every return above this line is a failure. */
    start_fill(m);
    return 0;
}

void waste_model_free(waste_model *m)
{
    /* Before anything else: the reader threads pread on the bank fds, and
     * those are closed further down. Stopping them here rather than in
     * waste_ecache_free — which runs last — is the difference between a
     * clean shutdown and a read on a descriptor that has been closed and
     * possibly reused. */
    /* Before the readers stop and before the bank descriptors close: the
     * fill thread preads on both. */
    stop_fill(m);
    waste_ecache_io_stop(&m->cache);

    /* Reachable on a partially-built model now that waste_open frees what
     * a failed load left behind, so nothing here may assume the load got
     * as far as its own allocation. */
    if (!m->t) m->n_tensors = 0;
    /* Device backends may hold no-copy wrappers keyed by these host
     * addresses.  Drop them while the allocations are still alive. */
    waste_backend_release_host_buffers();
    for (int i = 0; i < m->n_tensors; i++) {
        waste_dio_free(m->t[i].data); waste_dio_free(m->t[i].q);
        waste_dio_free(m->t[i].qs);
    }
    free(m->t);
    if (m->trunk_fd >= 0) close(m->trunk_fd);
    free(m->embrow); free(m->embsc);
    free(m->mmxq); free(m->mmxs);
    free(m->codebooks);
    free(m->codebooksT);
    for (int L = 0; L < 128; L++) {
        free(m->S[L]); free(m->conv[L]); free(m->latcache[L]);
        free(m->idxpool[L]); free(m->idxbuf[L]);
        free(m->qsa_k[L]); free(m->qsa_v[L]); free(m->qsa_rawk[L]);
        for (int s = 0; s < WASTE_MAX_SHARDS; s++)
            if (m->bank[L].fd[s] >= 0) close(m->bank[L].fd[s]);
    }
    free(m->hcx);
    free(m->ple_ring);
    free(m->ple_emb);
    free(m->gdn_g);
    free(m->qsa_q); free(m->qsa_gate); free(m->qsa_attn);
    free(m->qsa_kf); free(m->qsa_vf); free(m->qsa_scr); free(m->qsa_work);
    free(m->qsa_cs); free(m->qsa_sel); free(m->qsa_taken);
    free(m->moe_prob); free(m->moe_used);
    free(m->x); free(m->h); free(m->tmp); free(m->att); free(m->logits);
    free(m->ff); free(m->e_gate); free(m->e_up); free(m->e_down); waste_dio_free(m->lut);
    free(m->lut8); free(m->lut8_scale);
    free(m->xga); free(m->xub); free(m->xacc);
    waste_dio_free(m->xlut); free(m->xlut8); free(m->xqs);
    free(m->xq); free(m->xs); waste_dio_free(m->miss_buf);
    free(m->blockres); free(m->prefix_sum); free(m->ares);
    free(m->cx); free(m->cnorm); free(m->cresid); free(m->cq); free(m->ckv);
    free(m->clat); free(m->cff); free(m->cexp); free(m->cblockres);
    free(m->cprefix); free(m->croute); free(m->crw); free(m->cused);
    free(m->cq8); free(m->cq8_scale);
    free(m->hcflat); free(m->hccol); free(m->hcmix);
    free(m->idxsel); free(m->idxscore); free(m->idxrank);
    waste_ecache_free(&m->cache);
    pthread_mutex_destroy(&m->fetch_mu);
}

/* ---- expert dequant ---------------------------------------------------- */

/* Opens an expert bank with the page cache out of the way.
 *
 * The whole hit-rate argument depends on this: with a 17 GB container on a
 * 64 GB machine the kernel would cache the banks and every number we
 * measure would be about the kernel's cache, not ours. K3's ~900 GB gets no
 * such help, so the engine must not depend on it.
 *
 * macOS says so with fcntl. Linux needs O_DIRECT and Windows
 * FILE_FLAG_NO_BUFFERING, which are the same mechanism under two names and
 * carry the same contract: the offset, the length and the destination
 * buffer must all be multiples of the device's logical block size. Offsets
 * and lengths are whole 4 KiB pages by construction — but only if this
 * container was written that way, so check rather than assume, because a
 * misaligned record makes every read fail outright rather than merely run
 * slow. The buffers come from waste_dio_alloc. A filesystem may still
 * refuse the bypass (tmpfs does), so fall back and at least turn readahead
 * off.
 *
 * `*direct` is set to 0 if any bank ends up without the bypass, so `waste
 * info` can say the hit rates are being measured against a warm page
 * cache. Not validated on Linux from this machine — see LEARNED.md §14. */
/* O_DIRECT and FILE_FLAG_NO_BUFFERING both accept the open and then fail
 * every transfer whose offset, length or buffer is not aligned to the
 * device's logical block — which the caller cannot know from the file. So
 * the eligibility test below is necessary and not sufficient, and the only
 * honest confirmation is a transfer: one block at offset 0, a page per bank
 * at load. Without it, a device wanting more than WASTE_ALIGN would open
 * unbuffered and then fail on the first expert of the first token. */
#if defined(_WIN32) || (defined(__linux__) && defined(O_DIRECT))
static int bank_probe(int fd)
{
    void *buf = waste_dio_alloc(WASTE_ALIGN);
    if (!buf) return 0;
    const int64_t got = waste_pread(fd, buf, WASTE_ALIGN, 0);
    waste_dio_free(buf);
    return got == (int64_t)WASTE_ALIGN;
}
#endif

static int bank_open(const char *path, size_t rec_bytes, int asked, int *direct)
{
    (void)rec_bytes; (void)direct;
    /* Two separate questions, and conflating them misreports one platform
     * or the other.
     *
     * `want` is whether the caller wants the bypass at all:
     * waste_cfg.use_direct_io, which is on by default, unless WASTE_DIRECT=0
     * vetoes it — the escape hatch for measuring what the page cache is
     * worth. `aligned` is whether O_DIRECT and FILE_FLAG_NO_BUFFERING can
     * be used, since both refuse a transfer that is not a whole number of
     * sectors and a misaligned record fails every read outright rather
     * than merely running slow.
     *
     * macOS asks only the first: F_NOCACHE has no alignment contract, so a
     * container whose records are not page multiples still gets the
     * bypass there and `direct` must not be cleared for it. */
    const char *e = getenv("WASTE_DIRECT");
    const int want = asked && !(e && e[0] == '0');
    /* WASTE_ALIGN, not WASTE_DIO_ALIGN. The two are different questions and
     * conflating them cost every Linux run its bypass: 16 KiB is what the
     * engine aligns its *buffers* to, because Metal wants a whole page,
     * while what O_DIRECT constrains is the offset and the length — and the
     * format guarantees those in 4 KiB units. No container has ever been a
     * 16 KiB multiple (Kimi-Linear's record is 651 pages, K3's is 3029), so
     * this test was false for every model that exists and Linux silently
     * read through the page cache while reporting it. Reported as issue #4;
     * macOS was unaffected, F_NOCACHE having no alignment contract. */
    const int aligned = rec_bytes && rec_bytes % WASTE_ALIGN == 0;
    (void)want; (void)aligned;
#if defined(_WIN32)
    if (want && aligned) {
        const int fd = waste_open_stream(path, 1);
        if (fd >= 0 && bank_probe(fd)) return fd;
        if (fd >= 0) close(fd);
    }
    *direct = 0;
    /* Second call rather than a retry flag: the first CreateFileA failed,
     * and the fallback differs only in the flag it asks for. */
    return waste_open_stream(path, 0);
#else
#if defined(__linux__) && defined(O_DIRECT)
    if (want && aligned) {
        const int fd = open(path, O_RDONLY | O_DIRECT);
        /* tmpfs accepts the flag and refuses the read; so would a device
         * wanting a bigger block than the format aligns to. */
        if (fd >= 0 && bank_probe(fd)) return fd;
        if (fd >= 0) close(fd);
    }
    *direct = 0;
#endif
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return fd;
#ifdef __APPLE__
    if (want) {
        fcntl(fd, F_NOCACHE, 1);      /* the engine owns caching */
        fcntl(fd, F_RDAHEAD, 0);
    } else {
        *direct = 0;
    }
#elif defined(__linux__)
    /* No bypass available: at least stop the kernel reading ahead into
     * pages nothing will ask for. */
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    *direct = 0;
#endif
    return fd;
#endif /* _WIN32 */
}

/* ---- record verification ------------------------------------------------
 *
 * This runs once per record that comes off the disk and never on a cache
 * hit: the unit being checked is "bytes that just entered RAM", and
 * re-checking a record already in the cache would cost a pass over every
 * expert on every token to learn nothing new.
 *
 * The crc32 covers the body only — the converter computes it from byte 48
 * to the end of the per-channel scales — so the header is outside it and
 * has to be checked structurally instead. That is needed anyway: the
 * offsets saying where the body ends are in the header, and following an
 * unvalidated offset to decide how much to checksum is a read past the
 * buffer rather than a check.
 *
 * Everything here is derivable from the manifest, so it is derived rather
 * than believed: the record has to be the bank's stride, has to be the
 * expert that was asked for, and its scales are one f16 per output row of
 * each of the three matrices.
 */

typedef enum {
    REC_OK = 0,
    REC_E_READ,      /* short read                                        */
    REC_E_HEADER,    /* magic, identity, offsets                          */
    REC_E_CRC,       /* the payload is not what the converter wrote       */
    REC_E_NOSLOT,    /* the cache could not free a slot to read into      */
} rec_status;

static rec_status record_check(const waste_model *m, int layer, int expert,
                               const uint8_t *rec)
{
    const size_t rec_bytes = (size_t)m->bank[layer].rec_bytes;
    const waste_expert_hdr *h = (const waste_expert_hdr *)rec;

    if (rec_bytes < sizeof *h) return REC_E_HEADER;
    if (h->magic != WASTE_MAGIC_EXPERT) return REC_E_HEADER;
    /* A bank is a flat array indexed by expert id, so a record naming a
     * different one means the file has been spliced, or truncated by
     * something other than a whole number of records — a pread that lands
     * on the wrong one still succeeds. */
    if (h->layer != (uint16_t)layer || h->expert_id != (uint16_t)expert)
        return REC_E_HEADER;
    if ((size_t)h->rec_4k_blocks * WASTE_ALIGN != rec_bytes) return REC_E_HEADER;
    /* The fmt byte and the manifest have to agree, not merely each be
     * legal: VQ4P and VQ3R records are the same size for the same matrix,
     * so a container that mixes them would read the right bytes and decode
     * the wrong indices without ever failing a bounds check. */
    if (h->fmt != (m->index_bits == 6 ? WQ_VQ4P : WQ_VQ3R) &&
        !(m->index_bits == 8 && h->fmt == WQ_VQ2R))
        return REC_E_HEADER;
    if (h->lowrank_id != 0) return REC_E_HEADER;             /* v0 */
    /* codebook_id indexes the table three stage-groups deep and nothing
     * downstream bounds it. */
    if ((long)h->codebook_id + 3L * m->stages > m->n_books) return REC_E_HEADER;

    if (!(h->gate_off == sizeof *h && h->gate_off < h->up_off &&
          h->up_off < h->down_off && h->down_off < h->chan_corr_off &&
          h->chan_corr_off <= rec_bytes))
        return REC_E_HEADER;

    /* one f16 scale per output row of gate, up and down */
    const size_t scales = 2u * (size_t)(m->expert_m[0] + m->expert_m[1] +
                                        m->expert_m[2]);
    if (scales > rec_bytes - h->chan_corr_off) return REC_E_HEADER;

    /* Only the checksum is optional, and it is off by default. The checks
     * above are O(1) and are the ones keeping an offset from a damaged
     * header out of the arithmetic downstream, so they always run: what
     * the switch buys is the pass over the payload, and what it costs is
     * that pass. Nothing here is ever skipped for memory safety. */
    if (!m->verify) return REC_OK;

    const size_t payload = (size_t)h->chan_corr_off + scales - sizeof *h;
    if (waste_crc32(rec + sizeof *h, payload) != h->crc32) return REC_E_CRC;
    return REC_OK;
}

/* One pread of a 4 KiB-aligned record — what the layout exists for —
 * and then the check that it is the record it says it is.
 * The offset is computed in 64 bits before the multiply: a bank of 384
 * experts crosses 2 GB well before the biggest layer does. */
/* First failure wins: it is the one with a cause worth reporting, and the
 * layer and expert of the twentieth tell nobody anything. With read-ahead
 * on, "first" is decided by the lock rather than by program order — any of
 * the concurrent failures is equally the one to report. */
static int bank_fail(waste_model *m, rec_status st, int layer, int expert)
{
    pthread_mutex_lock(&m->fetch_mu);
    if (!m->read_error) {
        m->read_error = (int)st;
        m->bad_layer = layer;
        m->bad_expert = expert;
    }
    pthread_mutex_unlock(&m->fetch_mu);
    return -1;
}

static int bank_fetch(void *user, int layer, int expert, uint8_t *dst)
{
    waste_model *m = (waste_model *)user;
    /* The ids are not always the engine's own. waste_ecache_warm takes
     * them from usage.waste, which travels next to a container and is
     * read at open without anyone asking for it — and its layer field is
     * 16 bits against a bank array of WASTE_MAX_LAYERS. Indexing it
     * unchecked gave a garbage fd and a garbage record size to a pread
     * aimed at a fixed-size cache slot. */
    const int bad_layer = layer < 0 || layer >= m->cfg.n_layers ||
                          layer >= WASTE_MAX_LAYERS;
    waste_bank *b = bad_layer ? NULL : &m->bank[layer];
    if (bad_layer || expert < 0 || expert >= b->n_experts ||
        b->fd[0] < 0 || b->rec_bytes <= 0)
        return bank_fail(m, REC_E_HEADER, layer, expert);

    /* Striped banks: expert e lives on shard e % n_shards at offset
     * (e / n_shards) * rec_bytes. Round-robin keeps the top-k experts of a
     * single token spread across devices, which is the point — the per-token
     * demand is k experts of ONE layer, so layer-granularity placement would
     * leave every read of a token on one drive. Unstriped is N=1 and the
     * division is exact: shard 0, offset e * rec_bytes. pread is positional,
     * so the reader threads share each shard fd without a seek to race over. */
    const int shard = expert % b->n_shards;
    const int64_t off = ((int64_t)(expert / b->n_shards)) * (int64_t)b->rec_bytes;
    const int64_t got = waste_pread(b->fd[shard], dst, (size_t)b->rec_bytes, off);
    rec_status st = got == (int64_t)b->rec_bytes ? REC_OK : REC_E_READ;
    if (st == REC_OK) st = record_check(m, layer, expert, dst);
    if (st != REC_OK) return bank_fail(m, st, layer, expert);

    pthread_mutex_lock(&m->fetch_mu);
    m->expert_reads++;
    pthread_mutex_unlock(&m->fetch_mu);
    return 0;
}

static const uint8_t *read_expert(waste_model *m, int L, int eid)
{
    if (m->cache.n_slots > 0) {
        const uint8_t *r = waste_ecache_get(&m->cache, L, eid, bank_fetch, m);
        /* A cache that cannot free a slot never reaches bank_fetch, so
         * nothing downstream would have recorded a reason and the layer
         * would go on to sum the experts it did get. The whole point of
         * the read_error channel is that a token computed with missing
         * experts is reported rather than answered. */
        if (!r && !m->read_error) bank_fail(m, REC_E_NOSLOT, L, eid);
        return r;
    }
    m->cache.misses++;
    m->cache.bytes_read += (size_t)m->bank[L].rec_bytes;
    return bank_fetch(m, L, eid, m->miss_buf) == 0 ? m->miss_buf : NULL;
}

const char *waste_model_read_error(const waste_model *m, int *layer, int *expert)
{
    if (!m->read_error) return NULL;
    if (layer) *layer = m->bad_layer;
    if (expert) *expert = m->bad_expert;
    switch (m->read_error) {
        case REC_E_READ:   return "short read";
        case REC_E_HEADER: return "record header is not what the bank index describes";
        case REC_E_NOSLOT: return "expert cache could not free a slot to read into";
        default:           return "checksum mismatch";
    }
}

void waste_model_clear_read_error(waste_model *m)
{
    m->read_error = 0;
    m->ctx_full = 0;
}

/* MLA stores one latent per position, so kv_cap is the whole sequence
 * this model can hold. A container with no MLA layer stores nothing per
 * position and is not bounded here. */
int waste_model_ctx_max(const waste_model *m)
{
    return (m->has_mla || m->has_qsa) ? m->kv_cap : 0;
}

int waste_model_ctx_full(const waste_model *m) { return m->ctx_full; }

/* ---- fused VQ matvec ---------------------------------------------------
 * Never materializes the weights. For a matrix stored as `stages` codebook
 * indices per 8-dim vector, y[row] = scale[row] * sum_v sum_s C_s[i]. x_v.
 * The inner term depends only on (stage, code, vector position), not on
 * the row — so it is tabulated once per matrix and reused by every row,
 * and (for gate/up, whose input is the layer's hidden state) by every
 * routed expert in the layer. Same idea as sqlite-vector's turbo LUT.
 *
 * lut layout: [v][stage][code] — the 3 values a row needs for vector v sit
 * in one contiguous 3 KiB block.
 */

/* dst[c] = sum_d x[d] * C[c][d], for all `entries` codes at once.
 *
 * With the codebook transposed to [dim][entry] this is one axpy per
 * dimension: every lane does useful work and nothing is reduced across
 * lanes. The previous shape — `entries` separate eight-element dot
 * products, each ending in vaddvq_f32 — spent more time folding four
 * lanes into one than multiplying. */
void waste_lutb_range(int lo, int hi, void *p)
{
    const lutb_arg *a = (const lutb_arg *)p;
    const int en = a->entries, vd = a->vec_dim, st = a->stages;
    for (int v = lo; v < hi; v++) {
        const float *xv = a->x + (size_t)v * vd;
        for (int s = 0; s < st; s++) {
            const float *CT = a->booksT + (size_t)(a->cb_base + s) * en * vd;
            float *dst = a->lut + ((size_t)v * st + s) * en;
#if defined(__ARM_NEON) || defined(__aarch64__)
            int c = 0;
            for (; c + 16 <= en; c += 16) {
                float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                            a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                for (int d = 0; d < vd; d++) {
                    const float32x4_t xd = vdupq_n_f32(xv[d]);
                    const float *cr = CT + (size_t)d * en + c;
                    a0 = vfmaq_f32(a0, xd, vld1q_f32(cr));
                    a1 = vfmaq_f32(a1, xd, vld1q_f32(cr + 4));
                    a2 = vfmaq_f32(a2, xd, vld1q_f32(cr + 8));
                    a3 = vfmaq_f32(a3, xd, vld1q_f32(cr + 12));
                }
                vst1q_f32(dst + c, a0);      vst1q_f32(dst + c + 4, a1);
                vst1q_f32(dst + c + 8, a2);  vst1q_f32(dst + c + 12, a3);
            }
            for (; c < en; c++) {
                float t = 0;
                for (int d = 0; d < vd; d++) t += xv[d] * CT[(size_t)d * en + c];
                dst[c] = t;
            }
#else
            for (int c = 0; c < en; c++) {
                float t = 0;
                for (int d = 0; d < vd; d++) t += xv[d] * CT[(size_t)d * en + c];
                dst[c] = t;
            }
#endif
        }
    }
}

static void vq_quant_lut(const float *lut, int nv, int st, int en,
                         int8_t *q, float *scale);

/* q/qs are the int8 shadow of `lut` and its per-block scales, or NULL when
 * the container is not VQ4P. Quantizing here rather than in vq_apply is not
 * a tidiness point: gate and up are built once per token and applied once
 * per routed expert, so doing it there ran the same pass top_k times over
 * a table that had not changed. */
static void vq_build_lut(waste_model *m, float *lut, int cb_base,
                         const float *x, int N, int stages, int entries,
                         int vec_dim, int8_t *q, float *qs)
{
    PROF_START(P_LUTB);
    lutb_arg a = { lut, m->codebooksT, x, cb_base, stages, entries, vec_dim };
    /* Vectors are independent. The build was serial while everything around
     * it used the pool, and on K3 it is 8.0 GFLOP per token — 7.0 of them
     * for the down projections, which are rebuilt once per routed expert. */
    pf_wide(WIDE_LUTB, N / vec_dim, 16, waste_k.lutb_range, &a);
    if (q) vq_quant_lut(lut, N / vec_dim, stages, entries, q, qs);
    PROF_END(P_LUTB);
}

typedef struct {
    float *y; const uint8_t *idx; const uint16_t *scale; const float *lut;
    int nv, stages, entries;
} vq_arg;

/* Row-tiled so the per-position table block is loaded once for a whole
 * tile instead of once per row. The naive row-outer/vector-inner order
 * streams the entire table (884 KB for a 2304-wide matrix) M times; at
 * M=1024 that is ~900 MB of traffic per matrix. With a tile of 64 rows the
 * table is read M/64 times and the tile's indices (55 KB) stay in L1.
 * This is a cache-blocking win, not a SIMD one: the inner op is a gather,
 * which NEON cannot vectorize. */
#define VQ_TILE 64          /* must equal the container's index_block */
#ifndef VQ_SUPER
#define VQ_SUPER 2          /* index blocks handled per pass (swept: 2 wins) */
#endif

/* What this loop actually costs, measured rather than assumed:
 *   - it is NOT table bandwidth. Re-reading the 884 KB table once per
 *     64-row tile works out to 8.2 GB/token, which would need 165 GB/s —
 *     suspiciously exactly this machine's ceiling — but raising VQ_SUPER
 *     to cut that traffic made it *slower* (0.25 -> 0.40 s at SUPER=8),
 *     because the table is shared read-only across threads and stays
 *     cached, while the extra index streams and accumulators do not.
 *   - it is NOT index locality either. Blocking the index layout so a
 *     tile's indices are contiguous measured 1.44x in isolation and
 *     changed nothing here.
 *   - it IS the load -> address -> load dependency of each gather.
 *     Interleaving four independent rows keeps four chains in flight and
 *     is what finally moved it.
 * Swept: VQ_SUPER 1 and 2 tie within noise, 4+ is worse. */
static void vq_rows(int b, int e, void *p)
{
    vq_arg *a = (vq_arg *)p;
    const int nv = a->nv, st = a->stages, en = a->entries;
    float acc[VQ_TILE * VQ_SUPER];

    for (int r0 = b; r0 < e; r0 += VQ_TILE * VQ_SUPER) {
        const int rows = (r0 + VQ_TILE * VQ_SUPER < e) ? VQ_TILE * VQ_SUPER : e - r0;
        const int nblk = (rows + VQ_TILE - 1) / VQ_TILE;
        memset(acc, 0, (size_t)rows * sizeof(float));

        for (int v = 0; v < nv; v++) {
            const float *blk = a->lut + (size_t)v * st * en;
            for (int j = 0; j < nblk; j++) {
                const int nr = (j + 1) * VQ_TILE <= rows ? VQ_TILE
                                                         : rows - j * VQ_TILE;
                const uint8_t *ix = a->idx +
                    ((size_t)(r0 / VQ_TILE + j) * nv + v) * VQ_TILE * st;
                float *ac = acc + (size_t)j * VQ_TILE;
                /* Each gather is load -> address -> load, a ~5-cycle chain.
                 * Four rows are independent, so interleaving them keeps
                 * four chains in flight instead of one.
                 *
                 * The three table loads per row are the algorithm and cannot
                 * go. The index bytes can: eight rows are twenty-four
                 * consecutive bytes, so six word loads and some shifting
                 * replace twenty-four byte loads. Per eight rows that is 64
                 * memory operations down to 46, and eight independent chains
                 * instead of four.
                 *
                 * Worth what it is worth: +3.5% end to end on Kimi-Linear,
                 * where this bucket is most of a step, and 3% of the bucket
                 * on K3, where expert I/O is twice the arithmetic and the
                 * gain does not reach the clock. docs/EFFICIENCY.md §5.
                 *
                 * Little-endian is already assumed throughout — the record
                 * headers are read as structs — so byte k of the word is
                 * shift 8k. */
                int r = 0;
                if (st == 3) {
                    const float *b1 = blk + en, *b2 = blk + 2 * en;
                    for (; r + 8 <= nr; r += 8, ix += 8 * 3) {
                        uint32_t w0, w1, w2, w3, w4, w5;
                        memcpy(&w0, ix,      4); memcpy(&w1, ix +  4, 4);
                        memcpy(&w2, ix +  8, 4); memcpy(&w3, ix + 12, 4);
                        memcpy(&w4, ix + 16, 4); memcpy(&w5, ix + 20, 4);
                        const float t0 = blk[w0 & 0xff]         + b1[(w0 >>  8) & 0xff] + b2[(w0 >> 16) & 0xff];
                        const float t1 = blk[w0 >> 24]          + b1[w1 & 0xff]         + b2[(w1 >>  8) & 0xff];
                        const float t2 = blk[(w1 >> 16) & 0xff] + b1[w1 >> 24]          + b2[w2 & 0xff];
                        const float t3 = blk[(w2 >>  8) & 0xff] + b1[(w2 >> 16) & 0xff] + b2[w2 >> 24];
                        const float t4 = blk[w3 & 0xff]         + b1[(w3 >>  8) & 0xff] + b2[(w3 >> 16) & 0xff];
                        const float t5 = blk[w3 >> 24]          + b1[w4 & 0xff]         + b2[(w4 >>  8) & 0xff];
                        const float t6 = blk[(w4 >> 16) & 0xff] + b1[w4 >> 24]          + b2[w5 & 0xff];
                        const float t7 = blk[(w5 >>  8) & 0xff] + b1[(w5 >> 16) & 0xff] + b2[w5 >> 24];
                        ac[r] += t0; ac[r + 1] += t1; ac[r + 2] += t2; ac[r + 3] += t3;
                        ac[r + 4] += t4; ac[r + 5] += t5; ac[r + 6] += t6; ac[r + 7] += t7;
                    }
                }
                for (; r < nr; r++, ix += st) {
                    float t = blk[ix[0]];
                    for (int s = 1; s < st; s++) t += blk[s * en + ix[s]];
                    ac[r] += t;
                }
            }
        }
        for (int r = 0; r < rows; r++)
            a->y[r0 + r] = acc[r] * f16_to_f32(a->scale[r0 + r]);
    }
}

/* ---- WQ_VQ4P ------------------------------------------------------------
 * Four 6-bit indices per row per vector position, and a 64-entry stage
 * table that is 64 bytes — one vqtbl4q_s8. VQ3R's 256-entry table is 16
 * vector registers on a machine with 32, which is why its gather stayed
 * scalar however it was unrolled (docs/LEARNED.md §25).
 *
 * The table has to be int8 for a byte shuffle to index it, so the fp32 LUT
 * is quantized first. One scale per WASTE_VQ_LUT_BLK positions rather than
 * one global scale: a LUT entry is dot(x_v, centroid) and its magnitude
 * follows ||x_v||, which varies by orders of magnitude across a hidden
 * state. A single scale would round the quiet positions to zero. Blocking
 * also bounds the int16 accumulator, so the fold to fp32 is the only place
 * precision is spent.
 */
typedef struct {
    const float *lut; int8_t *q; float *scale; int nv, st, en;
} lutq_arg;

/* One scale block per call, so the pass rides the same pool as everything
 * around it. It was serial to begin with, which cost more than the kernel
 * it was feeding saved: at Kimi-Linear's shapes the quantization is ~12% of
 * a *threaded* apply, so serial it was over 100% of one. */
static void vq_quant_range(int b, int e, void *p)
{
    const lutq_arg *a = (const lutq_arg *)p;
    const int st = a->st, en = a->en;
    for (int blk = b; blk < e; blk++) {
        const int v0 = blk * WASTE_VQ_LUT_BLK;
        int v1 = v0 + WASTE_VQ_LUT_BLK;
        if (v1 > a->nv) v1 = a->nv;
        const size_t n = (size_t)(v1 - v0) * st * en;
        const float *src = a->lut + (size_t)v0 * st * en;
        int8_t *dst = a->q + (size_t)v0 * st * en;
        size_t i = 0;
        float mx = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        {   /* Scalar this pass was ~0.4s a run on Kimi-Linear — more than
             * the kernel it feeds was saving — and almost all of it was
             * lrintf and the two clamp branches per value. */
            float32x4_t m0 = vdupq_n_f32(0), m1 = vdupq_n_f32(0);
            for (; i + 8 <= n; i += 8) {
                m0 = vmaxq_f32(m0, vabsq_f32(vld1q_f32(src + i)));
                m1 = vmaxq_f32(m1, vabsq_f32(vld1q_f32(src + i + 4)));
            }
            mx = vmaxvq_f32(vmaxq_f32(m0, m1));
        }
#endif
        for (; i < n; i++) {
            const float f = fabsf(src[i]);
            if (f > mx) mx = f;
        }
        a->scale[blk] = mx / 127.0f;
        const float inv = mx > 0 ? 127.0f / mx : 0.0f;
        i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        {
            const float32x4_t vinv = vdupq_n_f32(inv);
            const int32x4_t hi = vdupq_n_s32(127), lo = vdupq_n_s32(-127);
            for (; i + 16 <= n; i += 16) {
                int32x4_t c[4];
                for (int k = 0; k < 4; k++) {
                    /* vcvtnq rounds to nearest, ties to even — the same
                     * rule lrintf follows in the default mode, so the two
                     * paths agree value for value. The explicit clamp is
                     * what keeps -128 out; a saturating narrow alone would
                     * let it through and break the table's symmetry. */
                    const float32x4_t f = vmulq_f32(vld1q_f32(src + i + k * 4), vinv);
                    c[k] = vminq_s32(vmaxq_s32(vcvtnq_s32_f32(f), lo), hi);
                }
                const int16x8_t s0 = vcombine_s16(vmovn_s32(c[0]), vmovn_s32(c[1]));
                const int16x8_t s1 = vcombine_s16(vmovn_s32(c[2]), vmovn_s32(c[3]));
                vst1q_s8(dst + i, vcombine_s8(vmovn_s16(s0), vmovn_s16(s1)));
            }
        }
#endif
        for (; i < n; i++) {
            /* -127 rather than -128 keeps the table symmetric. */
            int t = (int)lrintf(src[i] * inv);
            if (t > 127) t = 127;
            else if (t < -127) t = -127;
            dst[i] = (int8_t)t;
        }
    }
}

/* Serial on purpose. A build is 4 to 9 scale blocks, which vectorized is a
 * few microseconds of work; handing that to the pool measured ~79us of
 * dispatch for ~7us of arithmetic, and 260 builds a token turned a pass
 * that should cost 0.06s into 0.41s. Threading it was tried both ways —
 * the numbers are in docs/LEARNED.md. */
static void vq_quant_lut(const float *lut, int nv, int st, int en,
                         int8_t *q, float *scale)
{
    lutq_arg a = { lut, q, scale, nv, st, en };
    vq_quant_range(0, (nv + WASTE_VQ_LUT_BLK - 1) / WASTE_VQ_LUT_BLK, &a);
}

void waste_vq_rows_p6(int b, int e, void *p)
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
            const float ls = a->lscale[v0 / WASTE_VQ_LUT_BLK];
            for (int r = 0; r < nr; r++) acc[r] += ls * (float)sum[r];
        }
        for (int r = 0; r < nr; r++)
            a->y[r0 + r] = acc[r] * f16_to_f32(a->scale[r0 + r]);
    }
}

/* The same two kernels, run on the calling thread.
 *
 * waste_parallel_for is not reentrant — one global descriptor guarded by
 * g_pool_run_mu — so nothing a worker calls may dispatch. When moe_layer
 * gives each expert its own thread, these are what that thread runs.
 * Splitting is by row either way, so the result does not depend on which
 * of the two was used. */
static void vq_apply_serial(waste_model *m, float *y, const uint8_t *idx,
                            const uint16_t *scale, int M, int N,
                            const float *lut, const int8_t *q, const float *qs)
{
    const int nv = N / m->vec_dim;
    if (m->index_bits == 6) {
        vqp_arg a = { y, idx, scale, q, qs, nv };
        waste_k.vq_rows_p6(0, M, &a);
    } else if (vq8_on && q && waste_k.vq_rows_e) {
        vqp_arg a = { y, idx, scale, q, qs, nv };
        waste_k.vq_rows_e(0, M, &a);
    } else {
        vq_arg a = { y, idx, scale, lut, nv, m->stages, m->cb_entries };
        vq_rows(0, M, &a);
    }
}

static void vq_matvec_serial(waste_model *m, float *y, const uint8_t *idx,
                             const uint16_t *scale, const float *x, int M,
                             int N, int cb_base, float *lut, int8_t *q,
                             float *qs)
{
    lutb_arg a = { lut, m->codebooksT, x, cb_base, m->stages, m->cb_entries,
                   m->vec_dim };
    waste_k.lutb_range(0, N / m->vec_dim, &a);
    if (q) vq_quant_lut(lut, N / m->vec_dim, m->stages, m->cb_entries, q, qs);
    vq_apply_serial(m, y, idx, scale, M, N, lut, q, qs);
}

static void vq_apply(waste_model *m, float *y, const uint8_t *idx,
                     const uint16_t *scale, int M, int N, const float *lut,
                     const int8_t *q, const float *qs)
{
    PROF_START(P_LUTA);
    const int nv = N / m->vec_dim;
    if (m->index_bits == 6) {
        vqp_arg a = { y, idx, scale, q, qs, nv };
        pf_wide(WIDE_VQ, M, VQ_TILE * p6_chunk, waste_k.vq_rows_p6, &a);
    } else if (vq8_on && q && waste_k.vq_rows_e) {
        vqp_arg a = { y, idx, scale, q, qs, nv };
        pf_wide(WIDE_VQ, M, VQ_TILE * VQ_SUPER, waste_k.vq_rows_e, &a);
    } else {
        vq_arg a = { y, idx, scale, lut, nv, m->stages, m->cb_entries };
        /* min_chunk = VQ_TILE keeps every thread's range block-aligned,
         * which the blocked index layout requires. */
        pf_wide(WIDE_VQ, M, VQ_TILE * VQ_SUPER, vq_rows, &a);
    }
    PROF_END(P_LUTA);
}

static void vq_matvec(waste_model *m, float *y, const uint8_t *idx,
                      const uint16_t *scale, const float *x, int M, int N,
                      int cb_base, float *lut, int8_t *q, float *qs)
{
    vq_build_lut(m, lut, cb_base, x, N, m->stages, m->cb_entries,
                 m->vec_dim, q, qs);
    vq_apply(m, y, idx, scale, M, N, lut, q, qs);
}

/* ---- expert-parallel MoE ------------------------------------------------
 * One routed expert per task instead of one row range per task.
 *
 * The row-parallel form dispatches three times per expert — gate, up, down
 * — and at Kimi-Linear's expert shapes each of those is a few microseconds
 * of arithmetic against a fork-join that costs tens. Per token that is ~900
 * dispatches; this is 26, one a layer, and every thread does a whole
 * expert. Records have to be held for the length of the layer for it, which
 * is what waste_ecache_hold exists for.
 *
 * Determinism is unaffected: each expert writes its own slice and the sum
 * over experts is done afterwards in j order, so the result does not depend
 * on which thread ran which expert or on how many there were. */
typedef struct {
    waste_model *m;
    const waste_config *c;
    const uint8_t **recs;
    const float *w;
    int j_off;                       /* first expert of this batch         */
    int inter, lat;
    const float *lut_gate, *lut_up;
    const int8_t *q_gate, *q_up;
    const float *qs_gate, *qs_up;
} xpar_arg;

static void moe_expert_range(int b, int e, void *p)
{
    const xpar_arg *a = (const xpar_arg *)p;
    waste_model *m = a->m;
    const waste_config *c = a->c;
    const int inter = a->inter, lat = a->lat;

    for (int t = b; t < e; t++) {
        const int j = a->j_off + t;
        const uint8_t *rec = a->recs[j];
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *sc = (const uint16_t *)(rec + h->chan_corr_off);
        float *ga = m->xga + (size_t)j * inter;
        float *ub = m->xub + (size_t)j * inter;
        float *acc = m->xacc + (size_t)j * lat;
        float *ld = m->xlut + (size_t)j * m->xlut_sz;
        int8_t *qd = m->xlut8 ? m->xlut8 + (size_t)j * m->xlut_sz : NULL;
        float *qsd = m->xqs ? m->xqs + (size_t)j * m->xnsc : NULL;

        vq_apply_serial(m, ga, rec + h->gate_off, sc, inter, lat,
                        a->lut_gate, a->q_gate, a->qs_gate);
        vq_apply_serial(m, ub, rec + h->up_off, sc + inter, inter, lat,
                        a->lut_up, a->q_up, a->qs_up);
        waste_act_pair_range(c, ga, ub, inter);
        vq_matvec_serial(m, acc, rec + h->down_off, sc + 2 * inter, ga, lat,
                         inter, h->codebook_id + 2 * m->stages, ld, qd, qsd);
        /* acc is left unweighted on purpose: the caller applies w[j] in the
         * same `ysum[i] += w[j] * acc[i]` the serial loop uses, so the
         * compiler contracts it to the same fma and the two paths agree
         * bit for bit. Weighting it here instead rounded the product first,
         * and a 1e-8 difference is not harmless downstream — vq_quant_lut
         * takes its scale from max|lut|, so a perturbation that small
         * changes the scale and moves every entry near a rounding boundary
         * by one LSB. That measured as 0.68 on a logit. */
    }
}

/* ---- layers ------------------------------------------------------------ */

/* Log-space decay gate, in place over [H][D].
 *
 * Kimi-Linear:  g = -exp(A_log) * softplus(z)         unbounded below
 * K3:           g = lower_bound * sigmoid(exp(A_log) * z)
 *
 * The second is not a clamp of the first — it is a different function, and
 * it confines the decay exp(g) to (exp(lower_bound), 1).
 */
/* A_log is one learnable log-scale PER HEAD (tech report eq. 5). K3 ships
 * the tensor padded to head_dim — 96 real values in a 128-slot tensor,
 * indices 96..127 exactly zero — so its length must not be used to infer
 * the layout: reading it per channel silently applies exp(0)=1 to a
 * quarter of the channels. `per_channel` is kept for a model that really
 * does store one per channel, but nothing ships that today. */
void waste_kda_decay_gate_ex(float *g, const float *A_log, const float *dt_bias,
                             int H, int D, float lower_bound, int per_channel)
{
    for (int h = 0; h < H; h++) {
        const float ea_head = per_channel ? 0.0f : expf(A_log[h]);
        for (int j = 0; j < D; j++) {
            const int i = h * D + j;
            const float ea = per_channel ? expf(A_log[j]) : ea_head;
            const float z = g[i] + (dt_bias ? dt_bias[i] : 0.0f);
            if (lower_bound < 0.0f)
                g[i] = lower_bound / (1.0f + expf(-ea * z));
            else {
                const float sp = z > 20.0f ? z : log1pf(expf(z));
                g[i] = -ea * sp;
            }
        }
    }
}

void waste_kda_decay_gate(float *g, const float *A_log, const float *dt_bias,
                          int H, int D, float lower_bound)
{
    waste_kda_decay_gate_ex(g, A_log, dt_bias, H, D, lower_bound, 0);
}

typedef struct {
    int K, V;
    const float *q, *k, *v, *g, *beta;
    float *S, *o, *u;
} kda_par;

static void kda_step_range(int lo, int hi, void *ap)
{
    const kda_par *a = (const kda_par *)ap;
    waste_k.kda_step(hi - lo, a->K, a->V,
                     a->q + (size_t)lo * a->K, a->k + (size_t)lo * a->K,
                     a->v + (size_t)lo * a->V, a->g + (size_t)lo * a->K,
                     a->beta + lo,
                     a->S + (size_t)lo * a->K * a->V,
                     a->o + (size_t)lo * a->V,
                     a->u + (size_t)lo * a->V);
}

static void kda_layer(waste_model *m, int L, const float *in, float *out)
{
    const waste_config *c = &m->cfg;
    const int H = c->kda_heads, D = c->kda_dim, C = H * D, hid = c->hidden;
    float *q = m->tmp, *k = q + C, *v = k + C, *g = v + C, *o = g + C;
    float *beta = o + C, *lo = beta + H, *gate = lo + C;

    const char *nm[3] = { "q", "k", "v" };
    float *dstv[3] = { q, k, v };
    for (int i = 0; i < 3; i++) {
        char b[128];
        snprintf(b, sizeof b, "%smodel.layers.%d.self_attn.%s_proj.weight", c->prefix, L, nm[i]);
        matvec_t(m, dstv[i], waste_find(m, b), in, C, hid);
        snprintf(b, sizeof b, "%smodel.layers.%d.self_attn.%s_conv1d.weight", c->prefix, L, nm[i]);
        waste_k.short_conv_step(C, c->conv_k, waste_find(m, b)->data, NULL,
                                m->conv[L] + (size_t)i * C * (c->conv_k - 1),
                                dstv[i], dstv[i]);
    }

    matvec_t(m, lo, waste_find(m, tname("%smodel.layers.%d.self_attn.f_a_proj.weight", c->prefix, L)), in, D, hid);
    matvec_t(m, g, waste_find(m, tname("%smodel.layers.%d.self_attn.f_b_proj.weight", c->prefix, L)), lo, C, D);
    const waste_tensor *At = waste_find(m, tname("%smodel.layers.%d.self_attn.A_log",
                                                 c->prefix, L));
    const float *dt = T(m, "%smodel.layers.%d.self_attn.dt_bias", c->prefix, L);
    /* A_log is per head, not per channel: the tech report's eq. 5 indexes it
     * by h, and K3 ships it padded to head_dim (indices 96..127 are exactly
     * zero on every layer we checked). */
    waste_kda_decay_gate_ex(g, At->data, dt, H, D, c->gate_lower_bound, 0);
    matvec_t(m, beta, waste_find(m, tname("%smodel.layers.%d.self_attn.b_proj.weight", c->prefix, L)), in, H, hid);
    for (int h = 0; h < H; h++) beta[h] = 1.0f / (1.0f + expf(-beta[h]));

    {
        /* Heads own disjoint K x V slices of the recurrent state, so the
         * decode step splits cleanly across the pool. Splitting here rather
         * than inside kda.c keeps the backend dispatch (NEON) in play, and
         * whole-head ranges keep the result bit-identical to the serial
         * version. Worth doing: K3 spends 19% of a decode step in this call,
         * 69 layers x 96 heads, and it was running on one core. */
        kda_par a = { D, D, q, k, v, g, beta, m->S[L], o, m->att };
        waste_parallel_for(H, 1, kda_step_range, &a);
    }

    if (c->full_rank_gate) {
        matvec_t(m, gate, waste_find(m, tname("%smodel.layers.%d.self_attn.g_proj.weight",
                                              c->prefix, L)), in, C, hid);
    } else {
        matvec_t(m, lo, waste_find(m, tname("%smodel.layers.%d.self_attn.g_a_proj.weight",
                                            c->prefix, L)), in, D, hid);
        matvec_t(m, gate, waste_find(m, tname("%smodel.layers.%d.self_attn.g_b_proj.weight",
                                              c->prefix, L)), lo, C, D);
    }
    const float *onw = T(m, "%smodel.layers.%d.self_attn.o_norm.weight", c->prefix, L);
    for (int h = 0; h < H; h++)
        waste_k.rmsnorm_gated(D, o + h * D, gate + h * D, onw, c->eps, o + h * D);

    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L)), o, hid, C);
}

/* MLA with kv_b_proj absorbed into the query and the output.
 *
 * The cache holds only what MLA is designed to cache: the kv_lora-wide
 * latent plus the rope dims, 2.25 KB per token per layer against the
 * 120 KB the expanded per-head K and V used to take. The scores are the
 * same numbers because
 *     q_nope . (W_kb c) == (W_kb^T q_nope) . c
 * and the values because
 *     sum_s a_s (W_vb c_s) == W_vb (sum_s a_s c_s),
 * so both halves of kv_b_proj move off the per-cached-token path and onto
 * the per-step one. Arithmetic goes up (the dots are 576 and 512 wide
 * instead of 192 and 128) and memory traffic goes down by the same 53x
 * the cache shrinks by.
 *
 * Heads are independent all the way through, so one parallel_for covers
 * the absorption, the scores, the softmax and the output projection —
 * the old expanded path ran that loop on one core.
 */
/* Rotate one qk_rope-wide slice in place at `pos`.
 *
 * GPT-J / interleaved: pair j is (x[2j], x[2j+1]). Upstream reaches the same
 * arithmetic by de-interleaving before a half-split rotate,
 *     q = q.view(b, h, s, d/2, 2).transpose(4, 3).reshape(b, h, s, d)
 * so pairing dim j with dim j + qk_rope/2 instead — the LLaMA layout — rotates
 * the wrong partners and still yields finite, weight-shaped output.
 *
 * The angles depend only on (pos, j), not on the head, so the caller builds
 * the tables once per token per layer and every head reuses them. */
static void rope_tables(const waste_config *c, int pos, float *cs, float *sn)
{
    for (int j = 0; j < c->qk_rope / 2; j++) {
        const float a = (float)pos * c->rope_inv_freq[j];
        cs[j] = cosf(a);
        sn[j] = sinf(a);
    }
}

static void rope_apply(int half, float *x, const float *cs, const float *sn)
{
    for (int j = 0; j < half; j++) {
        const float e = x[2 * j], o = x[2 * j + 1];
        x[2 * j]     = e * cs[j] - o * sn[j];
        x[2 * j + 1] = e * sn[j] + o * cs[j];
    }
}

/* ---- DeepSeek Sparse Attention, k-pool flavour (GLM-5.3-Flash) ---------
 *
 * An MLA layer here does not attend over the whole context. A small
 * indexer, with projections of its own, scores *pools* of index_kpool
 * adjacent cached tokens and keeps the best index_topk / index_kpool of
 * them; the query then attends over those pools' tokens plus the tail of
 * the context that has not filled a pool yet.
 *
 * Two things make this cheap to keep resident. A pool's compressed key is
 * one index_dim vector per index_kpool tokens — 32x less than the raw keys
 * the scores would otherwise need — and it is computed once, when its last
 * token arrives, never recomputed. So the per-step state is a rolling
 * buffer of the (key, gate) pairs of the pool being filled, and an
 * append-only array of finished pool keys.
 *
 * Below index_topk tokens of context this is exactly dense attention: every
 * complete pool is selected and the tail covers the rest. dsa_select says
 * so by returning -1, and the head loop takes its ordinary path. That is
 * not an approximation for short prompts — it is what the arithmetic
 * reduces to, and it is why the selection cost only appears where the
 * saving does.
 */
static void layernorm(float *o, const float *x, const float *w,
                      const float *b, int n, float eps)
{
    float mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    float var = 0;
    for (int i = 0; i < n; i++) { const float d = x[i] - mean; var += d * d; }
    const float inv = 1.0f / sqrtf(var / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = (x[i] - mean) * inv * w[i] + b[i];
}

/* Fold this token into the pool being filled, and close the pool when it is
 * the last token of one. `in` is the attention input, i.e. the layer's
 * post-input_layernorm hidden state, which is what the indexer reads. */
static void dsa_record(waste_model *m, int L, const float *in, int pos)
{
    const waste_config *c = &m->cfg;
    const int D = c->index_dim, P = c->index_kpool, hid = c->hidden;
    float *slot = m->idxbuf[L] + (size_t)(pos % P) * 2 * D;

    matvec_t(m, slot, waste_find(m, tname(
                 "%smodel.layers.%d.self_attn.indexer.wk.weight", c->prefix, L)),
             in, D, hid);
    /* A LayerNorm, not an RMSNorm: it subtracts the mean and it has a bias.
     * Everything else in this model normalizes the other way, which is
     * exactly why it is spelled out here. */
    layernorm(slot, slot, T(m, "%smodel.layers.%d.self_attn.indexer.k_norm.weight", c->prefix, L),
              T(m, "%smodel.layers.%d.self_attn.indexer.k_norm.bias", c->prefix, L),
              D, 1e-6f);
    matvec_t(m, slot + D, waste_find(m, tname(
                 "%smodel.layers.%d.self_attn.indexer.index_kpool_compress_gate",
                 c->prefix, L)), in, D, hid);

    if (pos % P != P - 1) return;                 /* pool still incomplete */
    if (m->n_pool[L] >= m->pool_cap) return;      /* bounded by kv_cap */

    /* The pooled key is a per-channel weighted average over the pool's
     * tokens: one softmax of (gate + a learned positional bias) per
     * channel, across index_kpool tokens. Not one softmax per token. */
    const float *ape = T(m, "%smodel.layers.%d.self_attn.indexer.index_kpool_compress_ape",
                         c->prefix, L);
    float *dst = m->idxpool[L] + (size_t)m->n_pool[L] * D;
    for (int d = 0; d < D; d++) {
        float mx = -1e30f;
        for (int j = 0; j < P; j++) {
            const float v = m->idxbuf[L][(size_t)j * 2 * D + D + d] + ape[j * D + d];
            if (v > mx) mx = v;
        }
        float sum = 0, acc = 0;
        for (int j = 0; j < P; j++) {
            const float e = expf(m->idxbuf[L][(size_t)j * 2 * D + D + d] +
                                 ape[j * D + d] - mx);
            sum += e;
            acc += e * m->idxbuf[L][(size_t)j * 2 * D + d];
        }
        dst[d] = acc / sum;
    }
    m->n_pool[L]++;
}

typedef struct {
    const float *pool, *q, *w;
    float *score;
    int D, heads;
    float scale;
} dsa_par;

static void dsa_score_range(int lo, int hi, void *ap)
{
    const dsa_par *a = (const dsa_par *)ap;
    for (int p = lo; p < hi; p++) {
        const float *k = a->pool + (size_t)p * a->D;
        float acc = 0;
        for (int h = 0; h < a->heads; h++) {
            float d = dotf(a->q + (size_t)h * a->D, k, a->D) * a->scale;
            if (d > 0.0f) acc += a->w[h] * d;      /* relu, then weighted */
        }
        a->score[p] = acc;
    }
}

/* Keep the `k` highest scores with a size-k min-heap over pool ids. */
static void dsa_heap_push(int *heap, int *n, int k, const float *score, int p)
{
    if (*n < k) {
        int i = (*n)++;
        heap[i] = p;
        while (i) {
            const int par = (i - 1) / 2;
            if (score[heap[par]] <= score[heap[i]]) break;
            const int t = heap[par]; heap[par] = heap[i]; heap[i] = t;
            i = par;
        }
        return;
    }
    if (score[p] <= score[heap[0]]) return;
    heap[0] = p;
    for (int i = 0;;) {
        const int l = 2 * i + 1, r = l + 1;
        int sm = i;
        if (l < k && score[heap[l]] < score[heap[sm]]) sm = l;
        if (r < k && score[heap[r]] < score[heap[sm]]) sm = r;
        if (sm == i) break;
        const int t = heap[sm]; heap[sm] = heap[i]; heap[i] = t;
        i = sm;
    }
}

static int cmp_int(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* Which cached positions this query attends over. Writes them ascending
 * into m->idxsel and returns how many; -1 means "all of them", which is
 * both the short-context case and the cheapest one. */
static int dsa_select(waste_model *m, int L, const float *in,
                      const float *q_resid, int pos)
{
    const waste_config *c = &m->cfg;
    const int D = c->index_dim, P = c->index_kpool, hid = c->hidden;
    const int kv_len = pos + 1, npool = m->n_pool[L];
    int keep = c->index_topk / P;
    if (keep > npool) keep = npool;
    if (keep >= npool) return -1;               /* every pool kept: dense */

    float *q = m->idxq;
    float *w = q + (size_t)c->index_heads * D;
    matvec_t(m, q, waste_find(m, tname(
                 "%smodel.layers.%d.self_attn.indexer.wq_b.weight", c->prefix, L)),
             q_resid, c->index_heads * D, c->q_lora);
    matvec_t(m, w, waste_find(m, tname(
                 "%smodel.layers.%d.self_attn.indexer.weights_proj.weight",
                 c->prefix, L)), in, c->index_heads, hid);
    const float hs = 1.0f / sqrtf((float)c->index_heads);
    for (int h = 0; h < c->index_heads; h++) w[h] *= hs;

    {
        dsa_par a = { m->idxpool[L], q, w, m->idxscore, D, c->index_heads,
                      1.0f / sqrtf((float)D) };
        waste_parallel_for(npool, 64, dsa_score_range, &a);
    }

    /* Ties are possible and mean the same thing: a pool whose every head
     * scored negative lands at exactly 0 after the relu, and so do all its
     * neighbours. Which of those the heap keeps is unspecified here and
     * ordered by index upstream; they contribute the same nothing either
     * way. */
    int n = 0;
    for (int p = 0; p < npool; p++)
        dsa_heap_push(m->idxrank, &n, keep, m->idxscore, p);
    qsort(m->idxrank, (size_t)n, sizeof(int), cmp_int);

    int out = 0;
    for (int i = 0; i < n; i++) {
        const int base = m->idxrank[i] * P;
        for (int j = 0; j < P; j++) m->idxsel[out++] = base + j;
    }
    /* WASTE_DUMP_DSA=path records the selection itself, one line per
     * (layer, position):
     *
     *     L pos npool keep  p0,p1,... : score0 score1 ...
     *
     * the pools that won and the scores they won on. A logit diff can say
     * that two implementations disagree; only this can say whether they
     * disagree about the *ranking* or about a tie, and that is the
     * difference between a bug and float noise. */
    if (dump_dsa) {
        FILE *df = fopen(dump_dsa, "a");
        if (df) {
            fprintf(df, "%d %d %d %d ", L, pos, npool, n);
            for (int i = 0; i < n; i++) fprintf(df, "%d,", m->idxrank[i]);
            fprintf(df, " :");
            for (int p2 = 0; p2 < npool; p2++)
                fprintf(df, " %.9g", m->idxscore[p2]);
            fputc('\n', df);
            fclose(df);
        }
    }
    /* The tail: the tokens past the last complete pool are always visible,
     * which is what keeps the most recent context from waiting for a pool
     * to fill before it can be attended to. */
    if (c->index_tail)
        for (int t = npool * P; t < kv_len; t++) m->idxsel[out++] = t;
    return out;
}

typedef struct {
    waste_model *m;
    const waste_tensor *kvb;
    const float *q, *lat;
    float *o;
    int S, qd, qk_nope, qk_rope, vh, kv_lora, latd;
    const int *sel;                  /* NULL = every cached position       */
    int nsel;
    float scale;
} mla_par;

static void mla_head_range(int lo, int hi, void *ap)
{
    const mla_par *a = (const mla_par *)ap;
    const int kl = a->kv_lora, latd = a->latd, S = a->S;
    const int nope = a->qk_nope, rope = a->qk_rope, vh = a->vh;
    const int wrows = nope + vh;                 /* kv_b_proj rows per head */

    for (int h = lo; h < hi; h++) {
        const float *qh = a->q + (size_t)h * a->qd;
        float *qa  = a->m->qabs + (size_t)h * kl;
        float *ca  = a->m->cacc + (size_t)h * kl;
        float *rw  = a->m->mrow + (size_t)h * kl;
        float *sc  = a->m->att  + (size_t)h * S;

        /* q~ = W_kb^T q_nope */
        memset(qa, 0, (size_t)kl * sizeof(float));
        for (int i = 0; i < nope; i++) {
            const float qi = qh[i];
            if (qi == 0.0f) continue;
            waste_deq_row(a->kvb, (long)h * wrows + i, kl, rw);
            for (int j = 0; j < kl; j++) qa[j] += qi * rw[j];
        }

        /* Dense unless the indexer narrowed it: `sel` lists the positions
         * this query may see, ascending, and everything downstream works in
         * terms of how many there are rather than of the context length. */
        const int *sel = a->sel;
        const int n = sel ? a->nsel : S;
        for (int t = 0; t < n; t++) {
            const float *cs = a->lat + (size_t)(sel ? sel[t] : t) * latd;
            float acc = dotf(qa, cs, kl);
            acc += dotf(qh + nope, cs + kl, rope);
            sc[t] = acc * a->scale;
        }
        softmax(sc, n);

        memset(ca, 0, (size_t)kl * sizeof(float));
        for (int t = 0; t < n; t++) {
            const float w = sc[t];
            const float *cs = a->lat + (size_t)(sel ? sel[t] : t) * latd;
            for (int j = 0; j < kl; j++) ca[j] += w * cs[j];
        }

        /* o = W_vb c~ */
        float *oh = a->o + (size_t)h * vh;
        for (int i = 0; i < vh; i++) {
            waste_deq_row(a->kvb, (long)h * wrows + nope + i, kl, rw);
            oh[i] = dotf(rw, ca, kl);
        }
    }
}

static void mla_layer(waste_model *m, int L, const float *in, float *out, int pos)
{
    const waste_config *c = &m->cfg;
    const int nh = c->n_heads, qd = c->qk_nope + c->qk_rope, vh = c->v_head;
    const int hid = c->hidden;
    const int latd = c->kv_lora + c->qk_rope;
    float *q = m->tmp, *ckv = q + nh * qd, *o = ckv + latd;

    float *q_resid = NULL;
    if (c->q_lora) {
        /* K3 LoRAs the query too: q_a -> RMSNorm -> q_b */
        float *qa = o + (size_t)nh * vh;
        q_resid = qa;
        matvec_t(m, qa, waste_find(m, tname("%smodel.layers.%d.self_attn.q_a_proj.weight",
                                            c->prefix, L)), in, c->q_lora, hid);
        waste_rmsnorm(qa, qa, waste_find(m, tname("%smodel.layers.%d.self_attn.q_a_layernorm.weight",
                                            c->prefix, L))->data, c->q_lora, c->eps);
        matvec_t(m, q, waste_find(m, tname("%smodel.layers.%d.self_attn.q_b_proj.weight",
                                           c->prefix, L)), qa, nh * qd, c->q_lora);
    } else {
        matvec_t(m, q, waste_find(m, tname("%smodel.layers.%d.self_attn.q_proj.weight",
                                           c->prefix, L)), in, nh * qd, hid);
    }
    matvec_t(m, ckv, waste_find(m, tname("%smodel.layers.%d.self_attn.kv_a_proj_with_mqa.weight", c->prefix, L)),
             in, c->kv_lora + c->qk_rope, hid);
    waste_rmsnorm(ckv, ckv, T(m, "%smodel.layers.%d.self_attn.kv_a_layernorm.weight", c->prefix, L),
            c->kv_lora, c->eps);
    /* Rotate before caching, not after: the cached entry is reused by every
     * later query and carries this token's position, while the query carries
     * the querying token's. Rotating on read would need the pair of positions
     * and would redo the work once per (query, key). */
    if (!c->mla_nope) {
        float cs[WASTE_MAX_ROPE_HALF], sn[WASTE_MAX_ROPE_HALF];
        rope_tables(c, pos, cs, sn);
        const int half = c->qk_rope / 2;
        for (int h = 0; h < nh; h++)
            rope_apply(half, q + (size_t)h * qd + c->qk_nope, cs, sn);
        rope_apply(half, ckv + c->kv_lora, cs, sn);
    }
    /* Cache the latent — normalized kpass followed by the rope dims, rotated
     * unless the model is NoPE. kv_b_proj is not applied here at all; it is
     * absorbed below. */
    memcpy(m->latcache[L] + (size_t)pos * latd, ckv, (size_t)latd * sizeof(float));
    /* WASTE_DUMP_LATENT=path appends the cached latent and the absorbed
     * query, the two things a KV-cache quantizer has to keep faithful. */
    const char *dump_latent = getenv("WASTE_DUMP_LATENT");
    if (dump_latent) {
        FILE *df = fopen(dump_latent, "ab");
        if (df) { fwrite(ckv, sizeof(float), (size_t)latd, df); fclose(df); }
    }
    m->n_kv[L] = pos + 1;

    /* The indexer runs on the same input the projections did, after this
     * token is in the cache — so a query always sees itself — and before
     * the heads, which need its answer. */
    int nsel = -1;
    if (c->index_topk) {
        dsa_record(m, L, in, pos);
        nsel = dsa_select(m, L, in, q_resid, pos);
    }

    {
        mla_par a;
        a.m = m;
        a.sel = nsel >= 0 ? m->idxsel : NULL;
        a.nsel = nsel >= 0 ? nsel : 0;
        a.kvb = waste_find(m, tname("%smodel.layers.%d.self_attn.kv_b_proj.weight",
                                    c->prefix, L));
        a.q = q; a.lat = m->latcache[L]; a.o = o;
        a.S = m->n_kv[L]; a.qd = qd;
        a.qk_nope = c->qk_nope; a.qk_rope = c->qk_rope;
        a.vh = vh; a.kv_lora = c->kv_lora; a.latd = latd;
        /* YaRN raises the attention scale by mscale_all_dim^2 when the config
         * sets it; att_mul is 1 otherwise, including on every NoPE model. */
        a.scale = c->att_mul / sqrtf((float)qd);
        waste_parallel_for(nh, 1, mla_head_range, &a);
    }
    if (c->mla_output_gate) {
        /* sigmoid gate on the attention output, before o_proj */
        float *g = o + (size_t)nh * vh + (c->q_lora ? c->q_lora : 0);
        matvec_t(m, g, waste_find(m, tname("%smodel.layers.%d.self_attn.g_proj.weight",
                                           c->prefix, L)), in, nh * vh, hid);
        for (int i = 0; i < nh * vh; i++) o[i] *= 1.0f / (1.0f + expf(-g[i]));
    }
    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.self_attn.o_proj.weight", c->prefix, L)), o, hid, nh * vh);
}

static void ffn(waste_model *m, const waste_tensor *W1, const waste_tensor *W3,
                const waste_tensor *W2, const float *in, float *out,
                int inter, int hid, float w, int accum)
{
    float *a = m->ff, *b = a + inter;
    matvec_t(m, a, W1, in, inter, hid);
    matvec_t(m, b, W3, in, inter, hid);
    waste_act_pair_range(&m->cfg, a, b, inter);
    float *dst = accum ? m->h : out;
    matvec_t(m, dst, W2, a, hid, inter);
    if (accum) for (int i = 0; i < hid; i++) out[i] += w * dst[i];
}

/* What layer L+1's router says about layer L's hidden state.
 *
 * The router of L+1 will really see the state after L's MoE and L+1's
 * attention; this asks it early, one residual short. Measured at 59.0%
 * recall over the true top-16 and steeply ranked — 92.2% at rank 1, 27.9%
 * at rank 16 — which is what makes a narrow prefetch pay and a wide one
 * not. LEARNED.md §34.
 *
 * Writes at most `n` ids and returns how many. 0 when L+1 is not a MoE
 * layer, which is also what happens on the last one. Scratch is the router
 * area of m->att, dead by the time this is called.
 */
static int predict_next_moe(waste_model *m, int L, const float *in, int *out, int n)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, hid = c->hidden;
    if (n <= 0 || E <= 0 || L + 1 >= c->n_layers) return 0;
    const waste_tensor *g = waste_find(m, tname(
        "%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L + 1));
    if (!g) return 0;
    if (n > E) n = E;

    float *sc = m->att + WASTE_ATT_ROUTER_OFF, *p = sc + E;
    matvec_t(m, sc, g, in, E, hid);
    const float *bias = T(m, "%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias",
                          c->prefix, L + 1);
    /* Selection order only, so the sigmoid is monotone and could be skipped
     * — kept because the bias is added in probability space, as the real
     * router does it, and a different order here would be a different
     * prediction rather than a faster one. */
    for (int e = 0; e < E; e++)
        p[e] = 1.0f / (1.0f + expf(-sc[e])) + (bias ? bias[e] : 0.0f);
    for (int j = 0; j < n; j++) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int q = 0; q < j; q++) if (out[q] == e) { taken = 1; break; }
            if (!taken && p[e] > bv) { bv = p[e]; best = e; }
        }
        out[j] = best;
    }
    return n;
}

static void moe_layer(waste_model *m, int L, const float *in, float *out, int *routed)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden;
    /* K3's Stable LatentMoE: experts run on a narrower projection of the
     * hidden state. `in` still drives the router and the shared experts. */
    const int lat = c->latent_dim ? c->latent_dim : hid;
    float *sc = m->att + WASTE_ATT_ROUTER_OFF;
    matvec_t(m, sc, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L)), in, E, hid);
    const float *bias = T(m, "%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias", c->prefix, L);
    float *score = sc + E;
    for (int e = 0; e < E; e++) score[e] = 1.0f / (1.0f + expf(-sc[e]));

    /* Cache-conditional routing (arXiv:2412.00099), off unless WASTE_CCR_LAMBDA
     * asks for it. Experts already in the cache get a bonus **for ranking
     * only** — `w[j]` below still takes the untouched `score[best]`, so the
     * gating weights are exactly what the model computed and only the choice
     * of which K to run moves.
     *
     * The bonus is scaled by this layer's own logit range so one lambda means
     * the same thing in a layer whose scores span 0.9 and one whose scores
     * span 0.02. The paper uses a running mean of that range; this uses the
     * range of the token in hand, which needs no per-layer state — and static
     * mutable state would be wrong here anyway, since waste.h promises several
     * models can be open in one process.
     *
     * This changes what the model outputs. LEARNED §54 is the reason it can
     * never become a default: "what routing buys is not which experts exist,
     * nor how they are weighted on average — it is the per-token exclusion",
     * and this edits precisely that. It is an instrument for measuring the
     * hit-rate/KL trade, and it stays one until a KL curve says otherwise. */
    uint8_t resident[1024];
    float ccr_bump = 0.0f;
    if (ccr_lambda > 0.0f && E <= (int)sizeof resident) {
        waste_ecache_resident_mask(&m->cache, L, E, resident);
        float lo = 1e30f, hi = -1e30f;
        for (int e = 0; e < E; e++) {
            const float v = score[e] + (bias ? bias[e] : 0.0f);
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        ccr_bump = ccr_lambda * (hi - lo);
    }

    int idx[64];
    float w[64];
    for (int j = 0; j < K; j++) {
        int best = -1;
        float bv = -1e30f;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int p = 0; p < j; p++) if (idx[p] == e) { taken = 1; break; }
            if (taken) continue;
            float v = score[e] + (bias ? bias[e] : 0.0f);
            if (ccr_bump > 0.0f && resident[e]) v += ccr_bump;
            if (v > bv) { bv = v; best = e; }
        }
        idx[j] = best;
        w[j] = score[best];          /* unmodified: ranking moved, gating did not */
    }
    if (c->renorm && K > 1) {
        float s = 0;
        for (int j = 0; j < K; j++) s += w[j];
        for (int j = 0; j < K; j++) w[j] /= (s + 1e-20f);
    }
    for (int j = 0; j < K; j++) w[j] *= c->routed_scale;
    if (routed) for (int j = 0; j < K; j++) routed[j] = idx[j];

    /* WASTE_DUMP_SCORES=path appends one line per (token, layer):
     *
     *     pos  L  v0 v1 .. v(E-1)
     *
     * the *selection* value for every expert, `score[e] + bias[e]` — the
     * quantity the loop above ranks on, not the weight it later applies.
     *
     * WASTE_DUMP_ROUTE records the ids that won. Any question about a
     * *different* ranking needs the ones that lost too: a residency prior
     * (arXiv:2412.00099) promotes an expert the real router placed outside
     * the top-K, and a trace of winners cannot say which one or by how much.
     * So this dumps the whole vector, and the question "would a cache-aware
     * ranking hit more often, and how far does the distribution move" is
     * answerable offline — which is the same reasoning, and the same
     * economics, as the comment below. */
    if (dump_scores) {
        FILE *sf = fopen(dump_scores, "a");
        if (sf) {
            fprintf(sf, "%d %d", dump_pos0, L);
            /* %.9g, not %.6g: this dump exists to answer how close a
             * ranking decision was, and six digits cannot resolve the ties
             * that decide one. Two engines that agree on every score to
             * float precision printed as *identical* at six digits while
             * selecting differently, which reads as a selection bug and is
             * a near-tie the format could not show. Nine digits round-trip
             * a float. */
            for (int e = 0; e < E; e++)
                fprintf(sf, " %.9g", score[e] + (bias ? bias[e] : 0.0f));
            fputc('\n', sf);
            fclose(sf);
        }
    }

    /* WASTE_DUMP_ROUTE=path appends one line per (token, layer):
     *
     *     L  id0..idK-1  w0..wK-1
     *
     * the layer, the top-K expert ids in selection order, then their
     * renormalized weights. The weights gated lever C of
     * docs/EFFICIENCY.md; the ids gate the cross-layer prefetcher whose
     * next_layer_top field docs/FORMAT.md has reserved since the skeleton.
     * Both questions are "is the signal there", and both are cheaper to
     * answer from a trace than from a build. */
    /* Route capture, for tests/sweep.c.
     *
     * On K3 at top-8 two kernels that agree with the f32 path to 4e-5 on
     * Kimi-Linear differ from *each other* by logit rel L2 0.13 — because
     * any perturbation, however small, eventually flips one expert out of
     * eight at some layer, and a flipped expert is a discrete change with
     * a size of its own. So a logit norm cannot separate "the arithmetic
     * moved" from "the selection moved", and on this model it is mostly
     * measuring the second. This records what was actually selected, which
     * is the interpretable gate §61 and §62 used on the CUDA work. */
    if (waste_route_cap && waste_route_n + K <= waste_route_cap_n) {
        for (int j = 0; j < K; j++) waste_route_cap[waste_route_n + j] = idx[j];
        waste_route_n += K;
    }

    if (dump_route) {
        /* Third group on the line: what the *next* layer's router says about
         * *this* layer's hidden state. That is the predictor deltafin calls
         * "router lookahead", and it is a different and much stronger one
         * than the co-occurrence over expert ids that LEARNED §29 refuted —
         * it asks the real router rather than a statistic about its past
         * answers. Whether the hidden state has moved too far between here
         * and there is exactly what wants measuring. -1 when there is no
         * next MoE layer to ask. */
        int look[64];
        const int nlook = predict_next_moe(m, L, in, look, K);
        FILE *df = fopen(dump_route, "a");
        if (df) {
            fprintf(df, "%d %d", dump_pos0, L);
            for (int j = 0; j < K; j++) fprintf(df, " %d", idx[j]);
            for (int j = 0; j < K; j++) fprintf(df, " %.6g", w[j]);
            for (int j = 0; j < K; j++) fprintf(df, " %d", j < nlook ? look[j] : -1);
            fputc('\n', df);
            fclose(df);
        }
    }

    /* Every id this layer will read is known here, before the first read.
     * Handing them over lets the cache keep reads in flight while the
     * matmuls below run; without read-ahead it does nothing. */
    waste_ecache_hint(&m->cache, L, idx, K);

    const int inter = c->moe_inter;
    float *ga = m->ff, *ub = ga + inter, *acc = m->e_gate;
    const float *xin = in;
    float *lat_in = m->ares + hid, *lat_out = lat_in + lat;
    if (c->latent_dim) {
        matvec_t(m, lat_in, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
                     c->prefix, L)), in, lat, hid);
        xin = lat_in;
    }
    float *ysum = c->latent_dim ? lat_out : out;
    memset(ysum, 0, (size_t)lat * sizeof(float));
    const int lut_sz = ((hid > lat ? hid : lat) / m->vec_dim) * m->stages * m->cb_entries;
    float *lut_gate = m->lut, *lut_up = lut_gate + lut_sz, *lut_down = lut_up + lut_sz;
    /* The int8 shadow mirrors m->lut region for region — same element
     * count, one byte each — so the three offsets are the same lut_sz. */
    int8_t *q_gate = NULL, *q_up = NULL, *q_down = NULL;
    float *qs_gate = NULL, *qs_up = NULL, *qs_down = NULL;
    if (m->lut8) {
        const int nsc = lut_sz / (m->stages * m->cb_entries)
                        / WASTE_VQ_LUT_BLK + 2;
        q_gate = m->lut8; q_up = q_gate + lut_sz; q_down = q_up + lut_sz;
        qs_gate = m->lut8_scale; qs_up = qs_gate + nsc; qs_down = qs_up + nsc;
    }
#if defined(WASTE_ENABLE_METAL)
    /* ---- the routed experts, in batches, applied on the device ---------
     *
     * Two phases per batch. Phase 1 is the batch's gate and up in one
     * command buffer — they share the layer's two tables, because every
     * routed expert sees the same input. Phase 2 is the batch's down, each
     * with its own table, because each sees its own activated
     * intermediate.
     *
     * Batches rather than the whole layer, and the reason is measured:
     * holding all K records before any arithmetic starts is the barrier
     * docs/LEARNED.md §44 describes, and with the applies on the device it
     * costs more than they save — expert I/O went 2.69 -> 8.19 s over
     * eight steps while the apply went 5.78 -> 4.31. The hint above has
     * already queued every read; a batch only has to wait for its own.
     *
     * Between the phases the CPU does the SiTU and builds the batch's down
     * tables, which is where §61's "build the LUT on the device" would go
     * next — on that vehicle it was the difference between 3.8 and 9.1
     * tok/s.
     */
    if (metal_moe && m->xga && m->index_bits != 6 && K > 1 &&
        K <= WASTE_PF_MAX && m->cache.n_slots >= 4 * K) {
        const uint8_t *recs[WASTE_PF_MAX];
        waste_vq_job jobs[2 * WASTE_PF_MAX];
        const size_t rec_bytes = m->bank[L].rec_bytes;
        int ok = 1, lut_done = 0;
        for (int j0 = 0; j0 < K && ok; j0 += xpar_batch) {
            int j1 = j0 + xpar_batch;
            if (j1 > K) j1 = K;
            PROF_START(P_EDEQ);
            int n = j0;
            for (; n < j1; n++) {
                recs[n] = waste_ecache_hold(&m->cache, L, idx[n], bank_fetch, m);
                if (!recs[n]) break;
            }
            PROF_END(P_EDEQ);
            if (n < j1) { ok = 0; break; }

            if (!lut_done) {
                const waste_expert_hdr *h0 = (const waste_expert_hdr *)recs[j0];
                vq_build_lut(m, lut_gate, h0->codebook_id + 0 * m->stages,
                             xin, lat, m->stages, m->cb_entries, m->vec_dim,
                             NULL, NULL);
                vq_build_lut(m, lut_up, h0->codebook_id + 1 * m->stages,
                             xin, lat, m->stages, m->cb_entries, m->vec_dim,
                             NULL, NULL);
                lut_done = 1;
            }

            PROF_START(P_LUTA);
            int nj = 0;
            for (int j = j0; j < j1; j++) {
                const waste_expert_hdr *h = (const waste_expert_hdr *)recs[j];
                waste_vq_job *g1 = &jobs[nj++], *u1 = &jobs[nj++];
                g1->y = m->xga + (size_t)j * inter;
                g1->rec = recs[j]; g1->rec_bytes = rec_bytes;
                g1->idx_off = (uint32_t)h->gate_off;
                g1->sc_off = (uint32_t)h->chan_corr_off;
                g1->lut = m->lut; g1->lut_bytes = m->lut_bytes;
                g1->lut_off = 0;
                g1->m = inter; g1->nv = lat / m->vec_dim;
                g1->stages = m->stages; g1->entries = m->cb_entries;
                *u1 = *g1;
                u1->y = m->xub + (size_t)j * inter;
                u1->idx_off = (uint32_t)h->up_off;
                u1->sc_off = (uint32_t)(h->chan_corr_off + (size_t)inter * 2);
                u1->lut_off = (uint32_t)lut_sz;
            }
            ok = waste_metal_vq3r(jobs, nj) == 0;
            PROF_END(P_LUTA);
            if (!ok) break;

            PROF_START(P_EMM);
            for (int j = j0; j < j1; j++) {
                float *g2 = m->xga + (size_t)j * inter;
                const float *u2 = m->xub + (size_t)j * inter;
                waste_act_pair_range(c, g2, u2, inter);
            }
            PROF_END(P_EMM);

            PROF_START(P_LUTB);
            for (int j = j0; j < j1; j++) {
                const waste_expert_hdr *h = (const waste_expert_hdr *)recs[j];
                lutb_arg a = { m->xlut + (size_t)j * m->xlut_sz, m->codebooksT,
                               m->xga + (size_t)j * inter,
                               h->codebook_id + 2 * m->stages, m->stages,
                               m->cb_entries, m->vec_dim };
                waste_parallel_for(inter / m->vec_dim, 16, waste_k.lutb_range, &a);
            }
            PROF_END(P_LUTB);

            {
            PROF_START(P_LUTA);
            nj = 0;
            for (int j = j0; j < j1; j++) {
                const waste_expert_hdr *h = (const waste_expert_hdr *)recs[j];
                waste_vq_job *d = &jobs[nj++];
                d->y = m->xacc + (size_t)j * lat;
                d->rec = recs[j]; d->rec_bytes = rec_bytes;
                d->idx_off = (uint32_t)h->down_off;
                d->sc_off = (uint32_t)(h->chan_corr_off + (size_t)2 * inter * 2);
                d->lut = m->xlut;
                d->lut_bytes = (size_t)K * m->xlut_sz * sizeof(float);
                d->lut_off = (uint32_t)((size_t)j * m->xlut_sz);
                d->m = lat; d->nv = inter / m->vec_dim;
                d->stages = m->stages; d->entries = m->cb_entries;
            }
            ok = waste_metal_vq3r(jobs, nj) == 0;
            PROF_END(P_LUTA);
            }
            waste_ecache_release(&m->cache);
        }
        if (ok) {
            PROF_START(P_EMM);
            for (int j = 0; j < K; j++) {
                const float *accj = m->xacc + (size_t)j * lat;
                const float wj = w[j];
                for (int i = 0; i < lat; i++) ysum[i] += wj * accj[i];
            }
            PROF_END(P_EMM);
            goto moe_done;
        }
        /* Anything that did not run on the device falls through to the
         * serial loop, which re-reads and reports the reason. */
        waste_ecache_release(&m->cache);
    }
#endif

    /* Expert-parallel path. Needs the per-expert scratch, and needs the
     * cache to be able to hold all K records at once — the held set is
     * unevictable, so a cache that is not comfortably larger than K would
     * be asked to find a victim among slots that are all pinned.
     *
     * And, by default, it needs this layer's K experts to be in the cache
     * already. That is the whole of §44's finding read the other way round:
     * one task per expert is a barrier against the read-ahead, worth 1.18x
     * when there is nothing to read and a regression when there is. The
     * env var used to be the only way to say which, and the right value
     * inverted between models — because what it was really standing in for
     * was whether the experts were resident, and that is a question the
     * cache can answer in K hash lookups.
     *
     * So it is asked here instead, per layer and per token. A run whose
     * budget holds the whole bank takes this path on every layer after the
     * first pass; a K3 run that streams takes it on the layers that happen
     * to be warm and leaves the reads overlapping everywhere else. Neither
     * needs a flag, and the answer follows the cache rather than the
     * container. WASTE_XPAR=0/1 still forces it either way. */
    const int xpar_here = xpar_on >= 0
        ? xpar_on
        : waste_ecache_resident_all(&m->cache, L, idx, K);
    if (xpar_here && m->xga && K > 1 && K <= WASTE_PF_MAX &&
        m->cache.n_slots >= 4 * K) {
        /* In batches, not all K at once. Holding every record before doing
         * any arithmetic is a barrier against the read-ahead: the hint has
         * already queued all K reads, and waiting for the last one before
         * starting the first expert stops the reads overlapping the
         * multiplies they were issued to hide behind. On Kimi-Linear, where
         * expert I/O is under 1% of a step, it cost nothing; on K3 it moved
         * the I/O bucket from 3.11s to 14.54s and made the whole change a
         * regression. A batch is a barrier only across itself. */
        const uint8_t *recs[WASTE_PF_MAX];
        int lut_done = 0, ok = 1;
        for (int j0 = 0; j0 < K; j0 += xpar_batch) {
            int j1 = j0 + xpar_batch;
            if (j1 > K) j1 = K;
            PROF_START(P_EDEQ);
            int n = j0;
            for (; n < j1; n++) {
                recs[n] = waste_ecache_hold(&m->cache, L, idx[n],
                                            bank_fetch, m);
                if (!recs[n]) break;
            }
            PROF_END(P_EDEQ);
            if (n < j1) { ok = 0; break; }
            PROF_START(P_EMM);
            if (!lut_done) {
                const waste_expert_hdr *h0 = (const waste_expert_hdr *)recs[0];
                vq_build_lut(m, lut_gate, h0->codebook_id + 0 * m->stages,
                             xin, lat, m->stages, m->cb_entries, m->vec_dim,
                             q_gate, qs_gate);
                vq_build_lut(m, lut_up, h0->codebook_id + 1 * m->stages,
                             xin, lat, m->stages, m->cb_entries, m->vec_dim,
                             q_up, qs_up);
                lut_done = 1;
            }
            xpar_arg pa = { m, c, recs, w, j0, inter, lat, lut_gate, lut_up,
                            q_gate, q_up, qs_gate, qs_up };
            waste_parallel_for(j1 - j0, 1, moe_expert_range, &pa);
            PROF_END(P_EMM);
            waste_ecache_release(&m->cache);
        }
        if (ok) {
            PROF_START(P_EMM);
            /* Summed here, in j order, so the total does not depend on the
             * thread count or the batch size — the same guarantee the row
             * split gives. */
            for (int j = 0; j < K; j++) {
                const float *accj = m->xacc + (size_t)j * lat;
                const float wj = w[j];
                for (int i = 0; i < lat; i++) ysum[i] += wj * accj[i];
            }
            PROF_END(P_EMM);
            goto moe_done;
        }
        /* Something did not read. Let go of what was held and fall through
         * to the serial loop, which re-reads and reports the reason. */
        waste_ecache_release(&m->cache);
    }

    int lut_ready = 0;
    for (int j = 0; j < K; j++) {
        PROF_START(P_EDEQ);
        const uint8_t *rec = read_expert(m, L, idx[j]);
        PROF_END(P_EDEQ);
        /* This token is already wrong — the expert it needed is not
         * readable. Stop rather than sum the ones that were, and let the
         * step report it; m->read_error is what carries the reason out. */
        if (!rec) break;
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *sc = (const uint16_t *)(rec + h->chan_corr_off);

        PROF_START(P_EMM);
        /* gate/up see the same input and the same per-layer codebooks for
         * every routed expert, so their tables are built once per token. */
        if (!lut_ready) {
            vq_build_lut(m, lut_gate, h->codebook_id + 0 * m->stages,
                         xin, lat, m->stages, m->cb_entries, m->vec_dim,
                         q_gate, qs_gate);
            vq_build_lut(m, lut_up, h->codebook_id + 1 * m->stages,
                         xin, lat, m->stages, m->cb_entries, m->vec_dim,
                         q_up, qs_up);
            lut_ready = 1;
        }
        vq_apply(m, ga, rec + h->gate_off, sc, inter, lat, lut_gate,
                 q_gate, qs_gate);
        vq_apply(m, ub, rec + h->up_off, sc + inter, inter, lat, lut_up,
                 q_up, qs_up);
        waste_act_pair_range(c, ga, ub, inter);
        vq_matvec(m, acc, rec + h->down_off, sc + 2 * inter, ga, lat, inter,
                  h->codebook_id + 2 * m->stages, lut_down, q_down, qs_down);
        const float wj = w[j];
        for (int i = 0; i < lat; i++) ysum[i] += wj * acc[i];
        PROF_END(P_EMM);
    }
moe_done:
    if (c->latent_dim) {
        if (c->latent_norm)
            waste_rmsnorm(ysum, ysum, waste_find(m, tname(
                        "%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight",
                        c->prefix, L))->data, lat, c->eps);
        matvec_t(m, out, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
                     c->prefix, L)), ysum, hid, lat);
    }

    /* The guess goes out last, when this layer's sixteen reads have all
     * been consumed and the disk is about to go idle through the next
     * layer's attention. Issuing it earlier would put speculative reads in
     * front of demand ones on a queue that is the bottleneck. */
    if (lookahead_n) {
        int nxt[64];
        const int nn = predict_next_moe(m, L, in, nxt, lookahead_n);
        if (nn) waste_ecache_prefetch(&m->cache, L + 1, nxt, nn);
    }

    /* shared expert — on the original hidden state, not the latent */
    float *tmp = m->h;
    ffn(m, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight", c->prefix, L)),
        in, tmp, c->moe_inter * (c->n_shared ? c->n_shared : 1), hid, 1.0f, 0);
    for (int i = 0; i < hid; i++) out[i] += tmp[i];
}

/* ---- Manifold-Constrained Hyper-Connections (GLM-5.3-Flash) ------------
 *
 * mHC replaces the single residual stream with `hc_mult` parallel ones.
 * Before each sublayer a learned mapping reads all of them at once and
 * produces three things:
 *
 *   pre  [H]      collapse weights — the one vector the sublayer runs on
 *   post [H]      where the sublayer's output lands, per stream
 *   comb [H][H]   how the streams mix into each other
 *
 * comb is then projected onto the doubly-stochastic manifold by
 * Sinkhorn-Knopp — alternating row and column normalization — which is the
 * "manifold-constrained" half of the name and what bounds the stream norms
 * across 45 layers. Twenty iterations on a 4x4 matrix, twice a layer:
 * nothing beside the 24 x 16384 projection that produced its logits.
 *
 * This is the same *kind* of mechanism as K3's Attention Residuals above —
 * both enrich what a layer sees beyond the immediately preceding one — and
 * the two are mutually exclusive in practice. Where AttnRes keeps a history
 * and attends over it, mHC keeps H streams and mixes them.
 *
 * The mapping runs in f32 because upstream casts the streams, the weights
 * and all three outputs to float before touching them; the engine is f32
 * throughout, so that matches by construction rather than by conversion.
 */
static void hc_norm_rows(float *comb, int H, float eps)
{
    for (int i = 0; i < H; i++) {
        float sum = 0;
        for (int j = 0; j < H; j++) sum += comb[i * H + j];
        const float inv = 1.0f / (sum + eps);
        for (int j = 0; j < H; j++) comb[i * H + j] *= inv;
    }
}

static void hc_norm_cols(float *comb, int H, float eps)
{
    for (int j = 0; j < H; j++) {
        float sum = 0;
        for (int i = 0; i < H; i++) sum += comb[i * H + j];
        const float inv = 1.0f / (sum + eps);
        for (int i = 0; i < H; i++) comb[i * H + j] *= inv;
    }
}

/* One mHC site. `site` is "attn" or "ffn"; the two differ only in which
 * (fn, base, scale) triple they read. Fills post/comb for hc_scatter and
 * writes the collapsed stream the sublayer runs on. */
static void hc_collapse(waste_model *m, int L, const char *site, const float *x,
                        float *post, float *comb, float *collapsed)
{
    const waste_config *c = &m->cfg;
    const int H = c->hc_mult, hid = c->hidden;
    const size_t HD = (size_t)H * hid;
    const int nmix = (2 + H) * H;

    /* Unweighted RMSNorm over the flattened streams — no learned gain, so
     * this is not waste_rmsnorm with a vector of ones: it is the whole
     * H * hidden vector normalized as one. */
    float ss = 0;
    for (size_t i = 0; i < HD; i++) ss += x[i] * x[i];
    const float inv = 1.0f / sqrtf(ss / (float)HD + c->eps);
    for (size_t i = 0; i < HD; i++) m->hcflat[i] = x[i] * inv;

    float *w = m->hcmix;
    matvec_t(m, w, waste_find(m, tname("%smodel.layers.%d.hc_%s_fn",
                                       c->prefix, L, site)),
             m->hcflat, nmix, (int)HD);
    const float *base = T(m, "%smodel.layers.%d.hc_%s_base", c->prefix, L, site);
    const float *sc   = T(m, "%smodel.layers.%d.hc_%s_scale", c->prefix, L, site);
    float *pre = w + nmix;

    for (int i = 0; i < H; i++) {
        pre[i]  = 1.0f / (1.0f + expf(-(w[i] * sc[0] + base[i]))) + c->hc_eps;
        post[i] = 2.0f / (1.0f + expf(-(w[H + i] * sc[1] + base[H + i])));
    }
    for (int i = 0; i < H; i++) {
        float *row = comb + (size_t)i * H;
        for (int j = 0; j < H; j++)
            row[j] = w[2 * H + i * H + j] * sc[2] + base[2 * H + i * H + j];
        softmax(row, H);
        for (int j = 0; j < H; j++) row[j] += c->hc_eps;
    }
    /* Columns first, then hc_iters-1 rounds of (rows, columns): the order
     * upstream uses, and the count is off by one from the obvious reading
     * of "iterations" because the first column pass happens before the
     * loop. Sinkhorn does not converge to the same matrix from the other
     * order in a finite number of steps. */
    hc_norm_cols(comb, H, c->hc_eps);
    for (int it = 1; it < c->hc_iters; it++) {
        hc_norm_rows(comb, H, c->hc_eps);
        hc_norm_cols(comb, H, c->hc_eps);
    }

    for (int d = 0; d < hid; d++) {
        float acc = 0;
        for (int i = 0; i < H; i++) acc += pre[i] * x[(size_t)i * hid + d];
        collapsed[d] = acc;
    }
}

/* x[i] <- post[i] * y + sum_k comb[k][i] * x[k], the scatter half of a
 * site. Upstream writes it as matmul(comb.transpose(-1, -2), residual),
 * which is this indexing read the other way round. */
static void hc_scatter(waste_model *m, float *x, const float *post,
                       const float *comb, const float *y)
{
    const int H = m->cfg.hc_mult, hid = m->cfg.hidden;
    float *tmp = m->hcflat;                  /* free again by this point */
    for (int i = 0; i < H; i++) {
        float *dst = tmp + (size_t)i * hid;
        const float pi = post[i];
        for (int d = 0; d < hid; d++) dst[d] = pi * y[d];
        for (int k = 0; k < H; k++) {
            const float ck = comb[(size_t)k * H + i];
            const float *src = x + (size_t)k * hid;
            for (int d = 0; d < hid; d++) dst[d] += ck * src[d];
        }
    }
    memcpy(x, tmp, (size_t)H * hid * sizeof(float));
}

/* The final collapse: GLM's hc_head is an unweighted mean over the streams,
 * not another learned mapping. */
static void hc_head(const waste_config *c, float *out, const float *x)
{
    const int H = c->hc_mult, hid = c->hidden;
    const float inv = 1.0f / (float)H;
    for (int d = 0; d < hid; d++) {
        float acc = 0;
        for (int i = 0; i < H; i++) acc += x[(size_t)i * hid + d];
        out[d] = acc * inv;
    }
}

/* ---- Attention Residuals (K3) ------------------------------------------
 * Every layer mixes its running sum with a history of block residuals via a
 * learned softmax attention over that history; every attn_res_block_size
 * layers the current sum is appended to the history.
 *
 *   v      = [block_residual..., prefix_sum]        (nb+1) x hidden
 *   k      = waste_rmsnorm(v)                             (no weight yet)
 *   scores = sum(k * (norm.weight * proj.weight))   (nb+1)
 *   out    = softmax(scores) . v
 */
void waste_apply_attn_res(waste_model *m, const float *blockres, int nb,
                          const float *prefix_sum, const float *norm_w,
                          const float *proj_w, float *out)
{
    const int hid = m->cfg.hidden;
    float *sc = m->att;
    for (int i = 0; i <= nb; i++) {
        const float *v = (i < nb) ? blockres + (size_t)i * hid : prefix_sum;
        float ss = 0;
        for (int j = 0; j < hid; j++) ss += v[j] * v[j];
        const float r = 1.0f / sqrtf(ss / (float)hid + m->cfg.eps);
        float acc = 0;
        for (int j = 0; j < hid; j++) acc += v[j] * r * norm_w[j] * proj_w[j];
        sc[i] = acc;
    }
    softmax(sc, nb + 1);
    /* The output aggregation passes the same buffer as prefix_sum and out,
     * so the prefix_sum term has to be written first, element by element —
     * a memset here would zero the input before reading it. */
    const float pl = sc[nb];
    for (int j = 0; j < hid; j++) out[j] = pl * prefix_sum[j];
    for (int i = 0; i < nb; i++) {
        const float *v = blockres + (size_t)i * hid;
        const float p = sc[i];
        for (int j = 0; j < hid; j++) out[j] += p * v[j];
    }
}

int waste_model_warm_cache(waste_model *m, const char *path)
{
    if (m->cache.n_slots <= 0) return 0;
    const int n = waste_ecache_warm(&m->cache, path, bank_fetch, m);
    /* Warming is best-effort: a usage file naming experts this container
     * does not have is stale, not broken, and must not be reported as a
     * read failure by the first generation that follows. */
    waste_model_clear_read_error(m);
    return n;
}

int waste_model_save_usage(const waste_model *m, const char *path)
{
    if (m->cache.n_slots <= 0) return 0;
    return waste_ecache_save_usage(&m->cache, path, m->cache.clock);
}

/* ---- session state ------------------------------------------------------
 * Written with the shapes it depends on, so a state file that no longer
 * matches the model is rejected rather than silently producing nonsense.
 */

typedef struct {
    uint32_t magic, version;
    int32_t  n_layers, hidden, kda_heads, kda_dim, conv_k, n_heads;
    int32_t  qk_nope, qk_rope, v_head, attn_res_block;
    int32_t  pos, n_blockres;
    /* hc_mult widens the residual stream this file ends with; index_dim
     * decides whether it carries indexer pools at all. Both were reserved
     * words, and both have to be compared: a file written by one and read
     * by the other is the right length in neither direction. */
    int32_t  hc_mult, index_dim;
} waste_state_hdr;

static void state_fill(const waste_model *m, waste_state_hdr *h, int pos)
{
    const waste_config *c = &m->cfg;
    memset(h, 0, sizeof *h);
    h->magic = WASTE_MAGIC_KDASTATE;
    h->n_layers = c->n_layers;
    h->hidden = c->hidden;
    h->pos = pos;
    if (c->arch_qwen) {
        /* Version 2: GDN/QSA/HC/PLE state. The fields keep their Kimi
         * names and carry Qwen's shapes, so the struct stays one size and
         * every shape this file's length depends on is still compared.
         * `hc_mult` and `index_dim` are free to reuse here because Qwen
         * has neither mHC nor the DSA indexer — the version guards the
         * two readings apart. */
        h->version = 2;
        h->kda_heads = c->gdn_v_heads;
        h->kda_dim = c->gdn_k_dim;
        h->conv_k = c->conv_k;
        h->n_heads = c->qsa_n_kv;
        h->qk_nope = c->hc_count;
        h->qk_rope = c->qsa_head_dim;
        h->v_head = c->idx_head_dim;
        h->attn_res_block = c->idx_compress;
        h->hc_mult = c->gdn_v_dim;
        h->index_dim = c->gdn_k_heads;
        h->n_blockres = 0;
        return;
    }
    h->version = 1;
    h->n_layers = c->n_layers; h->hidden = c->hidden;
    h->kda_heads = c->kda_heads; h->kda_dim = c->kda_dim; h->conv_k = c->conv_k;
    h->n_heads = c->n_heads; h->qk_nope = c->qk_nope; h->qk_rope = c->qk_rope;
    h->v_head = c->v_head; h->attn_res_block = c->attn_res_block;
    h->hc_mult = c->hc_mult; h->index_dim = c->index_topk ? c->index_dim : 0;
    h->pos = pos; h->n_blockres = m->n_blockres;
}

static int qwen_state_hdr_ok(const waste_config *c, const waste_state_hdr *h)
{
    return h->version == 2 && h->n_layers == c->n_layers &&
           h->hidden == c->hidden && h->kda_heads == c->gdn_v_heads &&
           h->kda_dim == c->gdn_k_dim && h->conv_k == c->conv_k &&
           h->n_heads == c->qsa_n_kv && h->qk_nope == c->hc_count &&
           h->qk_rope == c->qsa_head_dim && h->v_head == c->idx_head_dim &&
           h->attn_res_block == c->idx_compress &&
           h->hc_mult == c->gdn_v_dim &&
           h->index_dim == c->gdn_k_heads && h->n_blockres == 0;
}

static uint64_t qwen_layer_state_bytes(const waste_config *c, int L, int T)
{
    if (c->qwen_full[L]) {
        const int Hkv = c->qsa_n_kv, D = c->qsa_head_dim, Dk = c->idx_head_dim;
        return 12ULL + (uint64_t)T * (uint64_t)Dk * 4ULL +
               (uint64_t)T * (uint64_t)Hkv * (uint64_t)D * 4ULL;
    }
    const int Hv = c->gdn_v_heads, Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
    const int Hk = c->gdn_k_heads;
    const int qkv = 2 * Hk * Dk + Hv * Dv;
    const int ck = c->conv_k > 0 ? c->conv_k - 1 : 0;
    return (uint64_t)Hv * (uint64_t)Dk * (uint64_t)Dv * 4ULL +
           (uint64_t)qkv * (uint64_t)ck * 4ULL;
}

static uint64_t qwen_state_tail_bytes(const waste_config *c)
{
    const int R = (c->ple_conv_k > 1 && c->ngram_size > 0)
                ? (c->ple_conv_k - 1) * c->ngram_size : 0;
    return (uint64_t)c->hc_count * (uint64_t)c->hidden * 4ULL +
           (uint64_t)c->hc_count * (uint64_t)c->hidden *
           (uint64_t)(R > 0 ? R : 1) * 4ULL + 8ULL * 4ULL;
}

/* Every buffer a session accumulates into, back to the state of a fresh
 * open. Lived in waste.c reaching into the model's fields; it is here so
 * there is one copy, and so a measurement harness that drives the model
 * directly can start each arm from the same place the last one did. */
void waste_model_reset(waste_model *m)
{
    const waste_config *c = &m->cfg;
    for (int L = 0; L < c->n_layers; L++) {
        if (c->arch_qwen) {
            const int Hv = c->gdn_v_heads, Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
            const int qkv = 2 * c->gdn_k_heads * Dk + Hv * Dv;
            if (m->S[L])
                memset(m->S[L], 0, (size_t)Hv * Dk * Dv * sizeof(float));
            if (m->conv[L])
                memset(m->conv[L], 0,
                       (size_t)qkv * (c->conv_k > 0 ? c->conv_k - 1 : 0) * sizeof(float));
            m->n_qsa_blk[L] = 0;
            m->n_qsa_tail[L] = 0;
            if (m->qsa_rawk[L])
                memset(m->qsa_rawk[L], 0,
                       (size_t)m->kv_cap * c->idx_head_dim * sizeof(float));
        } else {
            if (m->S[L])
                memset(m->S[L], 0, (size_t)c->kda_heads * c->kda_dim * c->kda_dim * sizeof(float));
            if (m->conv[L])
                memset(m->conv[L], 0,
                       (size_t)3 * c->kda_heads * c->kda_dim * (c->conv_k - 1) * sizeof(float));
        }
        m->n_kv[L] = 0;
    }
    m->n_blockres = 0;
    if (m->x) memset(m->x, 0, (size_t)(c->hc_mult ? c->hc_mult : 1) *
                              c->hidden * sizeof(float));
    for (int L = 0; L < c->n_layers; L++) m->n_pool[L] = 0;
    if (c->arch_qwen) {
        if (m->hcx)
            memset(m->hcx, 0, (size_t)c->hc_count * c->hidden * sizeof(float));
        if (m->ple_ring) {
            const int R = (c->ple_conv_k > 1 && c->ngram_size > 0)
                ? (c->ple_conv_k - 1) * c->ngram_size : 0;
            memset(m->ple_ring, 0, (size_t)c->hc_count * c->hidden * (R > 0 ? R : 1) * sizeof(float));
        }
        for (int i = 0; i < 8; i++) m->ple_prev[i] = c->eos_token_id;
    }
    if (m->blockres && c->attn_res_block) {
        const int nb = c->n_layers / c->attn_res_block + 2;
        memset(m->blockres, 0, (size_t)nb * c->hidden * sizeof(float));
    }
}

/* The one tuning knob a sweep varies between arms in a single process.
 * WASTE_LOOKAHEAD still sets the default; this changes it after load, which
 * is the whole point — a new process costs 48 seconds of model load on K3
 * and, worse, a different machine state. */
void waste_model_set_lookahead(int n) { lookahead_n = n < 0 ? 0 : n; }

/* For tests/sweep.c: the SDOT trunk path is chosen once from the
 * environment, and an arm has to be able to flip it inside one process —
 * two arms in two processes are two computers (docs/LEARNED.md §33). */
void waste_model_set_sdot4(int mode, int sg)
{
    const uint32_t f = waste_cpu_features();
    if ((mode == TK_SDOT || mode == TK_I8MM) && !(f & WASTE_CPU_DOTPROD)) mode = TK_F32;
    if (mode == TK_I8MM && !(f & WASTE_CPU_I8MM)) mode = TK_SMLAL;
    if (mode < 0 || mode > TK_SMLAL) mode = TK_F32;
    trunk_kern = mode;
    if (sg == 32 || sg == 64 || sg == 128) sdot4_sg = sg;
}
/* For tests/sweep.c: the size above which a matvec goes to the device.
 * 0 sends everything, a very large value sends nothing — which is how one
 * process measures both arms of "is the GPU worth it here". */
/* For tests/sweep.c: the routed experts' applies on the device, or not. */
void waste_model_set_metal_moe(int on) { metal_moe = on; }

/* For tests/sweep.c. Only effective when the int8 shadow was
 * allocated at load, i.e. when WASTE_VQ8 was set in the environment
 * — the table is a load-time allocation and an arm cannot conjure it. */
void waste_model_set_vq8(int on) { vq8_on = on; }

void waste_model_set_wide(int mask) { wide_mask = mask < 0 ? 0 : mask; }
int  waste_model_fast_threads(void) { return waste_pool_fast(); }
int  waste_pool_threads_public(void) { return waste_pool_threads(); }

void waste_model_set_device_min_kb(long kb)
{
    waste_k.device_min_bytes = kb < 0 ? (size_t)-1 : (size_t)kb << 10;
}

int  waste_model_get_lookahead(void)  { return lookahead_n; }

/* Read-ahead. The internal SSD reaches 12.89 GB/s at queue depth 2 against
 * 10.73 at depth 1, so two readers is the whole of the bandwidth story; the
 * rest of the win is that a pread no longer blocks the matmuls.
 * WASTE_IO_THREADS=0 restores the synchronous path exactly, which is what
 * the cache-vs-no-cache checks compare against.
 *
 * A function rather than a block because waste_model_resize_cache has to
 * start them again on the same terms, and two copies of a policy is one
 * copy too many. */
static void start_readers(waste_model *m)
{
    int nio = 2, depth = 2;
    const char *e = getenv("WASTE_IO_THREADS");
    if (e) nio = atoi(e);
    e = getenv("WASTE_IO_DEPTH");
    if (e) depth = atoi(e);
    if (depth < nio) depth = nio;
    waste_ecache_io_start(&m->cache, bank_fetch, m, nio, depth);
}

/* ---- background fill ----------------------------------------------------
 *
 * When the resolved cache can hold every record the container has, the
 * demand stream still discovers them one miss at a time — 200 tokens of
 * Kimi-Linear read 13.2 GB of a 16.5 GB bank as 3443 separate misses, and
 * until a record has been asked for once it is not there. This reads the
 * rest, in bank order, on one thread, while the model runs.
 *
 * It is not a preload: nothing waits for it. A run that generates four
 * tokens gets whatever landed in the meantime and pays for the reads it
 * would have paid for anyway, spread differently; a run that generates
 * hundreds finds every layer resident well before it would have. The
 * second-order effect is the larger one — a fully resident layer is what
 * lets moe_layer take the expert-parallel path (see xpar_here), which is
 * worth 1.17x on Kimi-Linear and cannot be taken while records are still
 * arriving.
 *
 * Only when everything fits. Below that, "put anything in an empty slot"
 * competes with the demand stream for the slots it is about to need, and
 * LFRU is a better judge of what belongs there than file order is.
 *
 * WASTE_PRELOAD=0 turns it off; it is not otherwise configurable, because
 * the condition that gates it is not a preference.
 */
static void *fill_worker(void *p)
{
    waste_model *m = (waste_model *)p;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        if (m->bank[L].fd < 0) continue;
        for (int e = 0; e < m->bank[L].n_experts; e++) {
            if (atomic_load(&m->fill_stop)) return NULL;
            /* 0 means "already here, or no empty slot left". The second is
             * the end of the sweep and the first is not, so it cannot stop
             * on it — but the cursor makes a full cache cheap to discover,
             * one lock and no read. */
            if (waste_ecache_admit(&m->cache, L, e, bank_fetch, m) < 0)
                return NULL;            /* a bad read; the demand path reports it */
        }
    }
    return NULL;
}

static void start_fill(waste_model *m)
{
    m->fill_records = 0;
    for (int L = 0; L < m->cfg.n_layers; L++)
        if (m->bank[L].fd >= 0)
            m->fill_records += (uint64_t)m->bank[L].n_experts;
    if (!m->fill_records || !m->cache.io) return;
    if ((uint64_t)m->cache.n_slots < m->fill_records) return;
    { const char *e = getenv("WASTE_PRELOAD"); if (e && *e == '0') return; }
    atomic_store(&m->fill_stop, 0);
    if (pthread_create(&m->fill_th, NULL, fill_worker, m) == 0)
        m->fill_running = 1;
}

static void stop_fill(waste_model *m)
{
    if (!m->fill_running) return;
    atomic_store(&m->fill_stop, 1);
    pthread_join(m->fill_th, NULL);
    m->fill_running = 0;
}

/* Give the cache a different size without touching the trunk.
 *
 * A budget sweep used to need one process per budget, because the cache is
 * sized at open — and that is what made docs/LEARNED.md §32 and §33 come out
 * wrong, since each process met a machine the previous one had changed. The
 * trunk is the expensive part of a load, not the cache; re-making only the
 * cache keeps the 27 GB where it is and leaves the footprint at exactly what
 * the budget would have made it. */
int waste_model_resize_cache(waste_model *m, size_t cache_bytes)
{
    const int policy = m->cache.policy;
    int64_t rec = 0;
    for (int L = 0; L < m->cfg.n_layers; L++)
        if (m->bank[L].rec_bytes > rec) rec = m->bank[L].rec_bytes;
    if (rec <= 0) return -1;
    stop_fill(m);                          /* it holds slots of this cache */
    waste_ecache_free(&m->cache);          /* stops the readers first */
    if (waste_ecache_init(&m->cache, cache_bytes, (size_t)rec, policy)) return -1;
    start_readers(m);
    start_fill(m);
    return 0;
}

int waste_model_state_save(const waste_model *m, const char *path, int pos)
{
    const waste_config *c = &m->cfg;
    static _Atomic unsigned save_seq;
    const size_t pn = strlen(path);
    char *tmp = (char *)malloc(pn + 64);
    if (!tmp) return -1;
    snprintf(tmp, pn + 64, "%s.tmp.%ld.%u", path, (long)getpid(),
             atomic_fetch_add(&save_seq, 1));
    const int tfd = open(tmp, O_WRONLY | O_CREAT | O_EXCL | WASTE_O_BINARY, 0600);
    if (tfd < 0) { free(tmp); return -1; }
    FILE *f = fdopen(tfd, "wb");
    if (!f) { close(tfd); remove(tmp); free(tmp); return -1; }
    waste_state_hdr h;
    state_fill(m, &h, pos);
    int rc = fwrite(&h, sizeof h, 1, f) == 1 ? 0 : -1;

    if (c->arch_qwen) {
        for (int L = 0; L < c->n_layers && !rc; L++) {
            if (c->qwen_full[L]) {
                const int32_t T = m->n_kv[L];
                const int Hkv = c->qsa_n_kv, D = c->qsa_head_dim, Dk = c->idx_head_dim;
                const int32_t blk = m->n_qsa_blk[L], tail = m->n_qsa_tail[L];
                if (fwrite(&T, sizeof T, 1, f) != 1 ||
                    fwrite(&blk, sizeof blk, 1, f) != 1 ||
                    fwrite(&tail, sizeof tail, 1, f) != 1) {
                    rc = -1;
                    break;
                }
                if (T > 0 && m->qsa_rawk[L] &&
                    fwrite(m->qsa_rawk[L], sizeof(float),
                           (size_t)T * (size_t)Dk, f) != (size_t)T * (size_t)Dk)
                    rc = -1;
                const size_t kvbf = (size_t)T * (size_t)Hkv * (size_t)D;
                if (!rc && kvbf && m->qsa_k[L] &&
                    fwrite(m->qsa_k[L], 2, kvbf, f) != kvbf)
                    rc = -1;
                if (!rc && kvbf && m->qsa_v[L] &&
                    fwrite(m->qsa_v[L], 2, kvbf, f) != kvbf)
                    rc = -1;
            } else {
                const int Hv = c->gdn_v_heads, Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
                const int Hk = c->gdn_k_heads;
                const int qkv = 2 * Hk * Dk + Hv * Dv;
                const size_t sn = (size_t)Hv * (size_t)Dk * (size_t)Dv;
                const size_t cn = (size_t)qkv *
                    (size_t)(c->conv_k > 0 ? c->conv_k - 1 : 0);
                if (fwrite(m->S[L], sizeof(float), sn, f) != sn) rc = -1;
                if (!rc && cn && fwrite(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
            }
        }
        const int hcH = c->hc_count * c->hidden;
        if (!rc && hcH && m->hcx &&
            fwrite(m->hcx, sizeof(float), (size_t)hcH, f) != (size_t)hcH)
            rc = -1;
        {
            const int R = (c->ple_conv_k > 1 && c->ngram_size > 0)
                        ? (c->ple_conv_k - 1) * c->ngram_size : 0;
            const size_t pr = (size_t)c->hc_count * (size_t)c->hidden *
                              (size_t)(R > 0 ? R : 1);
            if (!rc && pr && m->ple_ring &&
                fwrite(m->ple_ring, sizeof(float), pr, f) != pr)
                rc = -1;
        }
        if (!rc && fwrite(m->ple_prev, sizeof(int), 8, f) != 8) rc = -1;
    } else {
    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    for (int L = 0; L < c->n_layers && !rc; L++) {
        if (c->kda_layer[L]) {
            if (fwrite(m->S[L], sizeof(float), (size_t)H * D * D, f) != (size_t)H * D * D) rc = -1;
            const size_t cn = (size_t)3 * C * (c->conv_k - 1);
            if (!rc && fwrite(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
        } else {
            const int32_t nkv = m->n_kv[L];
            if (fwrite(&nkv, sizeof nkv, 1, f) != 1) { rc = -1; break; }
            const size_t kn = (size_t)nkv * (c->kv_lora + c->qk_rope);
            if (kn && fwrite(m->latcache[L], sizeof(float), kn, f) != kn) rc = -1;
            /* The indexer's pools are context, not weights: a session
             * restored without them attends over a selection made from an
             * empty history, which is wrong quietly rather than loudly. */
            if (!rc && c->index_topk) {
                const int32_t np = m->n_pool[L];
                const size_t pn = (size_t)np * c->index_dim;
                const size_t bn = (size_t)c->index_kpool * 2 * c->index_dim;
                if (fwrite(&np, sizeof np, 1, f) != 1) rc = -1;
                else if (pn && fwrite(m->idxpool[L], sizeof(float), pn, f) != pn) rc = -1;
                else if (fwrite(m->idxbuf[L], sizeof(float), bn, f) != bn) rc = -1;
            }
        }
    }
    if (!rc && c->attn_res_block && m->n_blockres > 0) {
        const size_t n = (size_t)m->n_blockres * c->hidden;
        if (fwrite(m->blockres, sizeof(float), n, f) != n) rc = -1;
    }
    {
        const size_t xn = (size_t)(c->hc_mult ? c->hc_mult : 1) * c->hidden;
        if (!rc && fwrite(m->x, sizeof(float), xn, f) != xn) rc = -1;
    }
    }   /* end of the non-Qwen state: Qwen's residual is m->hcx, and it
         * has neither blockres nor a widened m->x. */
    if (!rc && waste_sync_file(f)) rc = -1;
    if (fclose(f)) rc = -1;
    if (!rc && waste_replace_file(tmp, path)) rc = -1;
    if (rc) remove(tmp);
    free(tmp);
    return rc;
}

int waste_model_state_load(waste_model *m, const char *path, int *pos)
{
    const waste_config *c = &m->cfg;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    waste_state_hdr h, want;
    state_fill(m, &want, 0);
    if (fread(&h, sizeof h, 1, f) != 1) { fclose(f); return -1; }
    if (h.magic != want.magic) { fclose(f); return -2; }
    if (c->arch_qwen) {
        if (!qwen_state_hdr_ok(c, &h)) { fclose(f); return -2; }
    } else if (h.version != want.version ||
        h.n_layers != want.n_layers || h.hidden != want.hidden ||
        h.kda_heads != want.kda_heads || h.kda_dim != want.kda_dim ||
        h.conv_k != want.conv_k || h.n_heads != want.n_heads ||
        h.qk_nope != want.qk_nope || h.qk_rope != want.qk_rope ||
        h.v_head != want.v_head || h.attn_res_block != want.attn_res_block ||
        h.hc_mult != want.hc_mult || h.index_dim != want.index_dim) {
        fclose(f);
        return -2;
    }

    const int H = c->kda_heads, D = c->kda_dim, C = H * D;
    const int nb_max = c->attn_res_block
                     ? c->n_layers / c->attn_res_block + 2 : 0;
    if (h.pos < 0 || h.pos == INT32_MAX ||
        (c->arch_qwen ? h.pos > m->kv_cap :
         (m->has_mla && h.pos > m->kv_cap)) ||
        (!c->arch_qwen && (h.n_blockres < 0 || h.n_blockres > nb_max))) {
        fclose(f);
        return -2;
    }

    const int64_t fsize_i = waste_file_size(fileno(f));
    uint64_t off = sizeof h;
    if (fsize_i < 0) { fclose(f); return -1; }
    const uint64_t fsize = (uint64_t)fsize_i;
    if (c->arch_qwen) {
        for (int L = 0; L < c->n_layers; L++) {
            uint64_t bytes = 0;
            if (c->qwen_full[L]) {
                int32_t T = 0, blk = 0, tail = 0;
                if (off > fsize || fsize - off < 12 ||
                    waste_pread(fileno(f), &T, 4, (int64_t)off) != 4 ||
                    waste_pread(fileno(f), &blk, 4, (int64_t)(off + 4)) != 4 ||
                    waste_pread(fileno(f), &tail, 4, (int64_t)(off + 8)) != 4) {
                    fclose(f); return -2;
                }
                if (T < 0 || T > m->kv_cap || T != h.pos) {
                    fclose(f); return -2;
                }
                bytes = qwen_layer_state_bytes(c, L, T);
            } else {
                bytes = qwen_layer_state_bytes(c, L, 0);
            }
            if (off > fsize || bytes > fsize - off) { fclose(f); return -2; }
            off += bytes;
        }
        const uint64_t tail = qwen_state_tail_bytes(c);
        if (off > fsize || tail > fsize - off || off + tail != fsize) {
            fclose(f); return -2;
        }
    } else {
    for (int L = 0; L < c->n_layers; L++) {
        uint64_t bytes = 0;
        if (c->kda_layer[L]) {
            bytes = ((uint64_t)H * D * D + (uint64_t)3 * C * (c->conv_k - 1)) * 4;
        } else {
            int32_t nkv = 0;
            if (off > fsize || fsize - off < sizeof nkv ||
                waste_pread(fileno(f), &nkv, sizeof nkv, (int64_t)off) != sizeof nkv) {
                fclose(f); return -2;
            }
            if (nkv < 0 || nkv > m->kv_cap || nkv != h.pos) {
                fclose(f); return -2;
            }
            bytes = sizeof nkv + (uint64_t)nkv * (c->kv_lora + c->qk_rope) * 4;
            if (c->index_topk) {
                int32_t np = 0;
                const uint64_t at = off + bytes;
                if (at > fsize || fsize - at < sizeof np ||
                    waste_pread(fileno(f), &np, sizeof np, (int64_t)at) != sizeof np) {
                    fclose(f); return -2;
                }
                if (np < 0 || np > m->pool_cap ||
                    np != (c->index_kpool ? h.pos / c->index_kpool : 0)) {
                    fclose(f); return -2;
                }
                bytes += sizeof np + ((uint64_t)np * c->index_dim +
                                      (uint64_t)c->index_kpool * 2 * c->index_dim) * 4;
            }
        }
        if (off > fsize || bytes > fsize - off) { fclose(f); return -2; }
        off += bytes;
    }
    {
        const uint64_t tail = (uint64_t)h.n_blockres * c->hidden * 4 +
                              (uint64_t)(c->hc_mult ? c->hc_mult : 1) *
                              c->hidden * 4;
        if (off > fsize || tail > fsize - off || off + tail != fsize) {
            fclose(f); return -2;
        }
    }
    }
    if (fseek(f, (long)sizeof h, SEEK_SET)) { fclose(f); return -1; }

    int rc = 0;
    if (c->arch_qwen) {
        for (int L = 0; L < c->n_layers && !rc; L++) {
            if (c->qwen_full[L]) {
                int32_t T = 0, blk = 0, tail = 0;
                const int Hkv = c->qsa_n_kv, Dq = c->qsa_head_dim, Dk = c->idx_head_dim;
                if (fread(&T, sizeof T, 1, f) != 1 ||
                    fread(&blk, sizeof blk, 1, f) != 1 ||
                    fread(&tail, sizeof tail, 1, f) != 1) {
                    rc = -1; break;
                }
                if (T > 0 && m->qsa_rawk[L] &&
                    fread(m->qsa_rawk[L], sizeof(float),
                          (size_t)T * (size_t)Dk, f) != (size_t)T * (size_t)Dk)
                    rc = -1;
                const size_t kvbf = (size_t)T * (size_t)Hkv * (size_t)Dq;
                if (!rc && kvbf && m->qsa_k[L] &&
                    fread(m->qsa_k[L], 2, kvbf, f) != kvbf)
                    rc = -1;
                if (!rc && kvbf && m->qsa_v[L] &&
                    fread(m->qsa_v[L], 2, kvbf, f) != kvbf)
                    rc = -1;
                m->n_kv[L] = T;
                m->n_qsa_blk[L] = blk;
                m->n_qsa_tail[L] = tail;
            } else {
                const int Hv = c->gdn_v_heads, Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
                const int Hk = c->gdn_k_heads;
                const int qkv = 2 * Hk * Dk + Hv * Dv;
                const size_t sn = (size_t)Hv * (size_t)Dk * (size_t)Dv;
                const size_t cn = (size_t)qkv *
                    (size_t)(c->conv_k > 0 ? c->conv_k - 1 : 0);
                if (fread(m->S[L], sizeof(float), sn, f) != sn) rc = -1;
                if (!rc && cn && fread(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
            }
        }
        const int hcH = c->hc_count * c->hidden;
        if (!rc && hcH && m->hcx &&
            fread(m->hcx, sizeof(float), (size_t)hcH, f) != (size_t)hcH)
            rc = -1;
        {
            const int R = (c->ple_conv_k > 1 && c->ngram_size > 0)
                        ? (c->ple_conv_k - 1) * c->ngram_size : 0;
            const size_t pr = (size_t)c->hc_count * (size_t)c->hidden *
                              (size_t)(R > 0 ? R : 1);
            if (!rc && pr && m->ple_ring &&
                fread(m->ple_ring, sizeof(float), pr, f) != pr)
                rc = -1;
        }
        if (!rc && fread(m->ple_prev, sizeof(int), 8, f) != 8) rc = -1;
    } else {
    for (int L = 0; L < c->n_layers && !rc; L++) {
        if (c->kda_layer[L]) {
            if (fread(m->S[L], sizeof(float), (size_t)H * D * D, f) != (size_t)H * D * D) rc = -1;
            const size_t cn = (size_t)3 * C * (c->conv_k - 1);
            if (!rc && fread(m->conv[L], sizeof(float), cn, f) != cn) rc = -1;
        } else {
            int32_t nkv = 0;
            if (fread(&nkv, sizeof nkv, 1, f) != 1) { rc = -1; break; }
            if (nkv < 0 || nkv > m->kv_cap) { rc = -1; break; }
            const size_t kn = (size_t)nkv * (c->kv_lora + c->qk_rope);
            if (kn && fread(m->latcache[L], sizeof(float), kn, f) != kn) rc = -1;
            m->n_kv[L] = nkv;
            if (!rc && c->index_topk) {
                int32_t np = 0;
                const size_t bn = (size_t)c->index_kpool * 2 * c->index_dim;
                if (fread(&np, sizeof np, 1, f) != 1) { rc = -1; break; }
                if (np < 0 || np > m->pool_cap) { rc = -1; break; }
                const size_t pn = (size_t)np * c->index_dim;
                if (pn && fread(m->idxpool[L], sizeof(float), pn, f) != pn) rc = -1;
                else if (fread(m->idxbuf[L], sizeof(float), bn, f) != bn) rc = -1;
                m->n_pool[L] = np;
            }
        }
    }
    m->n_blockres = h.n_blockres;
    if (!rc && c->attn_res_block && h.n_blockres > 0) {
        const size_t n = (size_t)h.n_blockres * c->hidden;
        if (fread(m->blockres, sizeof(float), n, f) != n) rc = -1;
    }
    {
        const size_t xn = (size_t)(c->hc_mult ? c->hc_mult : 1) * c->hidden;
        if (!rc && fread(m->x, sizeof(float), xn, f) != xn) rc = -1;
    }
    }   /* end of the non-Qwen state: Qwen's residual is m->hcx, and it
         * has neither blockres nor a widened m->x. */
    fclose(f);
    if (!rc && pos) *pos = h.pos;
    /* -3 means the file changed or the device failed after the successful
     * preflight and live state may have been touched.  The public wrapper
     * resets the context in that exceptional case rather than leaving a
     * mixture of two conversations. */
    return rc ? -3 : 0;
}

/* ---- chunked prefill ---------------------------------------------------
 * Decode and prefill want opposite strategies for the experts.
 *
 * Decoding one token, an expert's weights are used for a single vector, so
 * expanding them is pure waste and the LUT wins: never dequantize.
 *
 * Prefilling a chunk, the same expert serves many tokens, so it pays to
 * expand it once and run a real GEMM — dense FMA runs ~50x faster than the
 * gathers the LUT needs, and the expansion amortizes over the chunk.
 *
 * The bigger win is upstream of that: tokens in a chunk route to
 * overlapping expert sets, so the union is far smaller than n * top_k, and
 * each distinct expert is read from disk exactly once.
 */

int waste_model_chunk_max(const waste_model *m) { (void)m; return WASTE_CHUNK_MAX; }

/* Y[T][out] = X[T][in] . W^T, parallel over output rows. */
typedef struct {
    float *Y; const float *W, *X; int in, out, T;
} mm_arg;

static void mm_rows(int b, int e, void *p)
{
    mm_arg *a = (mm_arg *)p;
    for (int o = b; o < e; o++) {
        const float *row = a->W + (size_t)o * a->in;
        for (int t = 0; t < a->T; t++)
            a->Y[(size_t)t * a->out + o] = dotf(row, a->X + (size_t)t * a->in, a->in);
    }
}

static void matmul_f32(float *Y, const float *W, const float *X,
                       int out, int in, int T)
{
    mm_arg a = { Y, W, X, in, out, T };
    waste_parallel_for(out, 32, mm_rows, &a);
}

/* Tensor-aware batched matmul; dequantizes a Q8G row on the fly. */
typedef struct {
    float *Y; const int8_t *W; const uint16_t *ws; const float *X;
    int in, out, T, ng, group, bits;
    size_t rowbytes;
    const int8_t *xq;        /* activations, int8 per group (i8mm path)    */
    const float  *xs;        /* their scales, [T][ng]                      */
} mmq_arg;

#if defined(__ARM_FEATURE_MATMUL_INT8)
/* Unpack one weight row to int8, whatever it is stored as. */
static void unpack_row_i8(const mmq_arg *a, int o, int8_t *dst)
{
    const int8_t *q = a->W + (size_t)o * a->rowbytes;
    if (a->bits == 8) { memcpy(dst, q, (size_t)a->in); return; }
    if (a->bits == 4) {
        const uint8_t *p4 = (const uint8_t *)q;
        for (int i = 0; i < a->in; i++) {
            const uint8_t byte = p4[i >> 1];
            dst[i] = (int8_t)((i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8);
        }
        return;
    }
    for (int i = 0; i < a->in; i++) dst[i] = (int8_t)q3_at((const uint8_t *)q, i);
}

/* SMMLA: one instruction multiplies a 2x8 int8 tile by an 8x2 int8 tile
 * into a 2x2 int32 accumulator — 32 MACs, against 4 for an fp32 FMA. The
 * tiles here are two weight rows by two tokens, accumulated per
 * quantization group so each group's pair of scales applies once.
 *
 * Only reachable when the activations have been quantized to int8, which
 * the f32 path deliberately avoids, so it stays behind waste_cpu_features
 * and a measured decision rather than being the default. */
static void mmq_rows_i8mm(int b, int e, void *p)
{
    mmq_arg *a = (mmq_arg *)p;
    const int g = a->group, ng = a->ng, T = a->T, in = a->in;
    int8_t *w0 = (int8_t *)malloc((size_t)in * 2);
    if (!w0) return;
    int8_t *w1 = w0 + in;

    for (int o = b; o < e; o += 2) {
        const int have2 = (o + 1 < e);
        unpack_row_i8(a, o, w0);
        if (have2) unpack_row_i8(a, o + 1, w1); else memset(w1, 0, (size_t)in);
        const uint16_t *s0 = a->ws + (size_t)o * ng;
        const uint16_t *s1 = have2 ? a->ws + (size_t)(o + 1) * ng : s0;

        for (int t = 0; t < T; t += 2) {
            const int havet2 = (t + 1 < T);
            const int8_t *x0 = a->xq + (size_t)t * in;
            const int8_t *x1 = havet2 ? x0 + in : x0;
            const float *sx0 = a->xs + (size_t)t * ng;
            const float *sx1 = havet2 ? sx0 + ng : sx0;
            float acc00 = 0, acc01 = 0, acc10 = 0, acc11 = 0;

            for (int k = 0; k < ng; k++) {
                const int base = k * g;
                const int lim = (base + g <= in) ? g : in - base;
                int32x4_t r = vdupq_n_s32(0);
                int i = 0;
                for (; i + 8 <= lim; i += 8) {
                    const int8x16_t wt = vcombine_s8(vld1_s8(w0 + base + i),
                                                     vld1_s8(w1 + base + i));
                    const int8x16_t xt = vcombine_s8(vld1_s8(x0 + base + i),
                                                     vld1_s8(x1 + base + i));
                    r = vmmlaq_s32(r, wt, xt);
                }
                int32_t p00 = vgetq_lane_s32(r, 0), p01 = vgetq_lane_s32(r, 1);
                int32_t p10 = vgetq_lane_s32(r, 2), p11 = vgetq_lane_s32(r, 3);
                for (; i < lim; i++) {           /* tail, groups need not be x8 */
                    p00 += (int32_t)w0[base + i] * x0[base + i];
                    p01 += (int32_t)w0[base + i] * x1[base + i];
                    p10 += (int32_t)w1[base + i] * x0[base + i];
                    p11 += (int32_t)w1[base + i] * x1[base + i];
                }
                const float f0 = f16_to_f32(s0[k]), f1 = f16_to_f32(s1[k]);
                acc00 += (float)p00 * f0 * sx0[k];
                acc01 += (float)p01 * f0 * sx1[k];
                acc10 += (float)p10 * f1 * sx0[k];
                acc11 += (float)p11 * f1 * sx1[k];
            }
            a->Y[(size_t)t * a->out + o] = acc00;
            if (have2)  a->Y[(size_t)t * a->out + o + 1] = acc10;
            if (havet2) {
                a->Y[(size_t)(t + 1) * a->out + o] = acc01;
                if (have2) a->Y[(size_t)(t + 1) * a->out + o + 1] = acc11;
            }
        }
    }
    free(w0);
}
#endif

static void mmq_rows(int b, int e, void *p)
{
    mmq_arg *a = (mmq_arg *)p;
    const int g = a->group, ng = a->ng;
    float *row = (float *)malloc((size_t)a->in * sizeof(float));
    if (!row) return;
    for (int o = b; o < e; o++) {
        const int8_t *q = a->W + (size_t)o * a->rowbytes;
        const uint16_t *sc = a->ws + (size_t)o * ng;
        for (int k = 0; k < ng; k++) {
            const float s = f16_to_f32(sc[k]);
            const int lim = (k * g + g <= a->in) ? g : a->in - k * g;
            if (a->bits == 3) {
                const uint8_t *p3 = (const uint8_t *)q;
                for (int i = 0; i < lim; i++)
                    row[k * g + i] = (float)q3_at(p3, (long)k * g + i) * s;
            } else if (a->bits == 4) {
                const uint8_t *p4 = (const uint8_t *)q + (size_t)k * g / 2;
                for (int i = 0; i < lim; i++) {
                    const uint8_t byte = p4[i / 2];
                    const int v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
                    row[k * g + i] = (float)v * s;
                }
            } else {
                for (int i = 0; i < lim; i++)
                    row[k * g + i] = (float)q[(size_t)k * g + i] * s;
            }
        }
        for (int t = 0; t < a->T; t++)
            a->Y[(size_t)t * a->out + o] = dotf(row, a->X + (size_t)t * a->in, a->in);
    }
    free(row);
}

/* Batched trunk matmul. Instrumented separately because it hides inside
 * the MoE bucket during prefill — the latent projections and the shared
 * experts go through here, the routed experts do not. */
void waste_matmul_t(waste_model *m, float *Y, const waste_tensor *t,
                     const float *X, int out, int in, int T)
{
    (void)m;
    if (!t || (!t->q && !t->data)) {
        memset(Y, 0, (size_t)T * out * sizeof(float));
        return;
    }
    PROF_START(P_MM);
    if (!t->q) { matmul_f32(Y, t->data, X, out, in, T); PROF_END(P_MM); return; }
    const int g = t->group, ng = (in + g - 1) / g;
    mmq_arg a = { Y, t->q, t->qs, X, in, out, T, ng, g, t->bits, t->rowbytes,
                  NULL, NULL };
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (i8mm_on && T >= 2 && m->mmxq &&
        (size_t)T * in <= m->mmx_cap && (size_t)T * ng <= m->mms_cap) {
        for (int t2 = 0; t2 < T; t2++)
            quant_act(X + (size_t)t2 * in, in, g,
                      m->mmxq + (size_t)t2 * in, m->mmxs + (size_t)t2 * ng);
        a.xq = m->mmxq; a.xs = m->mmxs;
        waste_parallel_for(out, 32, mmq_rows_i8mm, &a);
        PROF_END(P_MM);
        return;
    }
#endif
    waste_parallel_for(out, 32, mmq_rows, &a);
    PROF_END(P_MM);
}

/* ---- forward ----------------------------------------------------------- */

static int prefill_alloc(waste_model *m, int T)
{
    if (m->chunk_cap >= T) return 0;
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, lat = c->latent_dim ? c->latent_dim : hid;
    const int nb = c->attn_res_block ? c->n_layers / c->attn_res_block + 2 : 1;

    free(m->cx); free(m->cnorm); free(m->cresid); free(m->cq); free(m->ckv);
    free(m->clat); free(m->cff); free(m->cexp); free(m->cblockres);
    free(m->cprefix); free(m->croute); free(m->crw); free(m->cused);
    /* Re-entered whenever a longer chunk is asked for, so these have to be
     * cleared as well as freed: the VQ4P branch below only assigns them
     * when it runs, and a stale pointer here is a double free. */
    free(m->cq8); free(m->cq8_scale);
    m->cq8 = NULL; m->cq8_scale = NULL;

    m->cx     = (float *)calloc((size_t)T * hid, sizeof(float));
    m->cnorm  = (float *)calloc((size_t)T * hid, sizeof(float));
    m->cresid = (float *)calloc((size_t)T * hid, sizeof(float));
    {   /* cq holds 2 LUTs per token plus one for `down`. Same story as
         * m->lut: the geometry is the container's, not 8/3/256. */
        int nmax = hid > lat ? hid : lat;
        if (c->moe_inter > nmax) nmax = c->moe_inter;
        const size_t lut_sz = (size_t)(nmax / m->vec_dim + 1) *
                              (size_t)m->stages * (size_t)m->cb_entries;
        m->cq = (float *)calloc((size_t)(2 * T + 1) * lut_sz + 64, sizeof(float));
        if (m->index_bits == 6) {
            const size_t nsc = lut_sz / ((size_t)m->stages * m->cb_entries)
                               / WASTE_VQ_LUT_BLK + 2;
            m->cq8 = (int8_t *)calloc((size_t)(2 * T + 1) * lut_sz + 64, 1);
            m->cq8_scale = (float *)calloc((size_t)(2 * T + 1) * nsc + 64,
                                           sizeof(float));
            if (!m->cq8 || !m->cq8_scale) return -1;
        }
    }
    {   /* ckv holds the shared-expert staging: gate, up, out */
        const int si = c->moe_inter * (c->n_shared ? c->n_shared : 1);
        m->ckv = (float *)calloc((size_t)T * (size_t)(2 * si + hid) + 64, sizeof(float));
    }
    m->clat   = (float *)calloc((size_t)T * (size_t)(2 * lat + 2 * hid), sizeof(float));
    {   /* cff serves two callers: the per-token MoE staging (2*moe_inter +
         * lat) and the batched dense FFN (2 * T * dense_inter). Size for
         * whichever is larger — getting this wrong overflows silently. */
        const size_t moe_need = (size_t)2 * c->moe_inter + (size_t)(hid > lat ? hid : lat);
        const size_t dense_need = (size_t)2 * T * c->dense_inter;
        m->cff = (float *)calloc((moe_need > dense_need ? moe_need : dense_need) + 64,
                                 sizeof(float));
    }
    /* one expanded expert: gate/up [inter][lat] and down [hid][inter] */
    m->cexp   = (float *)calloc((size_t)2 * c->moe_inter * lat + (size_t)lat * c->moe_inter,
                                sizeof(float));
    m->cblockres = (float *)calloc((size_t)T * nb * hid, sizeof(float));
    m->cprefix   = (float *)calloc((size_t)T * hid, sizeof(float));
    m->croute = (int *)calloc((size_t)T * 64, sizeof(int));
    m->crw    = (float *)calloc((size_t)T * 64, sizeof(float));
    /* The distinct experts of a chunk are at most one per (token, slot),
     * so croute's shape bounds this one too. */
    m->cused  = (int *)calloc((size_t)T * 64, sizeof(int));
    m->chunk_cap = T;
    return (m->cx && m->cnorm && m->cresid && m->cq && m->ckv && m->clat &&
            m->cff && m->cexp && m->cblockres && m->cprefix && m->croute &&
            m->crw && m->cused) ? 0 : -1;
}

/* MoE over a whole chunk.
 *
 * The first attempt expanded each expert once and ran GEMMs, on the theory
 * that the cost amortizes over the chunk. Measured, it does not: a chunk of
 * 16 tokens spreads over ~1200 distinct experts, i.e. under 3 tokens each,
 * so expanding 7 M weights to serve 3 vectors is far worse than the LUT.
 *
 * What the chunk *does* buy is I/O: those 3328 (token, expert) pairs are
 * only 1210 distinct experts, so the disk sees 2.75x fewer reads. So keep
 * the decode-style LUT maths and reorganize purely to read each expert
 * once. gate/up tables depend on the token but not the expert (codebooks
 * are per layer), so they are built once per token and reused across every
 * expert that token routes to.
 */
static void moe_chunk(waste_model *m, int L, const float *in, float *out, int nT)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden;
    const int lat = c->latent_dim ? c->latent_dim : hid, inter = c->moe_inter;
    int *route = m->croute;
    float *rw = m->crw;

    float *sc = m->att + WASTE_ATT_ROUTER_OFF, *score = sc + E;
    const float *bias = T(m, "%smodel.layers.%d.block_sparse_moe.gate.e_score_correction_bias",
                          c->prefix, L);
    for (int t = 0; t < nT; t++) {
        matvec_t(m, sc, waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight",
                                            c->prefix, L)), in + (size_t)t * hid, E, hid);
        for (int e = 0; e < E; e++) score[e] = 1.0f / (1.0f + expf(-sc[e]));
        int *idx = route + (size_t)t * K;
        float *w = rw + (size_t)t * K;
        for (int j = 0; j < K; j++) {
            int best = -1; float bv = -1e30f;
            for (int e = 0; e < E; e++) {
                int taken = 0;
                for (int p = 0; p < j; p++) if (idx[p] == e) { taken = 1; break; }
                if (taken) continue;
                const float v = score[e] + (bias ? bias[e] : 0.0f);
                if (v > bv) { bv = v; best = e; }
            }
            idx[j] = best; w[j] = score[best];
        }
        if (c->renorm && K > 1) {
            float s = 0;
            for (int j = 0; j < K; j++) s += w[j];
            for (int j = 0; j < K; j++) w[j] /= (s + 1e-20f);
        }
        for (int j = 0; j < K; j++) w[j] *= c->routed_scale;
        /* Same trace as moe_layer's, from the path that routes a whole
         * chunk at once — a prefill produces in one pass what decode would
         * take hundreds of seconds to emit. */
        if (dump_route) {
            FILE *df = fopen(dump_route, "a");
            if (df) {
                fprintf(df, "%d %d", dump_pos0 + t, L);
                for (int j = 0; j < K; j++) fprintf(df, " %d", idx[j]);
                for (int j = 0; j < K; j++) fprintf(df, " %.6g", w[j]);
                /* No lookahead on this path (LEARNED §36), but the column
                 * stays so every row of a trace has one shape. */
                for (int j = 0; j < K; j++) fprintf(df, " -1");
                fputc('\n', df);
                fclose(df);
            }
        }
    }

    const float *xin = in;
    float *lat_in = m->clat, *ysum = lat_in + (size_t)nT * lat;
    if (c->latent_dim) {
        waste_matmul_t(m, lat_in, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_down_proj.weight",
                     c->prefix, L)), in, lat, hid, nT);
        xin = lat_in;
    }
    memset(ysum, 0, (size_t)nT * lat * sizeof(float));

    /* gate/up tables: one per token, shared by every expert it routes to */
    const int lut_sz = ((hid > lat ? hid : lat) / m->vec_dim) * m->stages * m->cb_entries;
    float *lut_gu = m->cq;                    /* [nT][2][lut_sz] */
    float *lut_down = lut_gu + (size_t)nT * 2 * lut_sz;
    /* Same mirroring as moe_layer, over m->cq's 2*nT+1 regions. */
    int8_t *q_gu = m->cq8;
    int8_t *q_down = q_gu ? q_gu + (size_t)nT * 2 * lut_sz : NULL;
    const int nsc = lut_sz / (m->stages * m->cb_entries)
                    / WASTE_VQ_LUT_BLK + 2;
    float *qs_gu = m->cq8_scale;
    float *qs_down = qs_gu ? qs_gu + (size_t)nT * 2 * nsc : NULL;
    int lut_ready = 0;

    float *ga = m->cff, *ub = ga + inter, *acc = m->cff + 2 * inter;

    /* Collect the distinct experts first, so their reads can be handed to
     * the cache ahead of the arithmetic. The order is the same ascending
     * one the loop below consumes, which is what lets the read-ahead be a
     * queue rather than a guess. A chunk can name more experts than one
     * hint holds, so it is fed in windows; the pipeline drains once per
     * window, which costs one read's latency per WASTE_PF_MAX experts. */
    int *used_ids = m->cused;
    int n_used = 0;
    for (int e = 0; e < E; e++)
        for (int i = 0; i < nT * K; i++)
            if (route[i] == e) { used_ids[n_used++] = e; break; }

    for (int w = 0; w < n_used; w += WASTE_PF_MAX) {
    const int wn = n_used - w < WASTE_PF_MAX ? n_used - w : WASTE_PF_MAX;
    waste_ecache_hint(&m->cache, L, used_ids + w, wn);
    for (int u = w; u < w + wn; u++) {
        const int e = used_ids[u];

        PROF_START(P_EDEQ);
        const uint8_t *rec = read_expert(m, L, e);
        PROF_END(P_EDEQ);
        if (!rec) goto chunk_lost;       /* see moe_layer: the chunk is lost */
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *s16 = (const uint16_t *)(rec + h->chan_corr_off);

        PROF_START(P_EMM);
        if (!lut_ready) {
            for (int t = 0; t < nT; t++) {
                vq_build_lut(m, lut_gu + (size_t)(2 * t) * lut_sz,
                             h->codebook_id + 0 * m->stages,
                             xin + (size_t)t * lat, lat, m->stages,
                             m->cb_entries, m->vec_dim,
                             q_gu ? q_gu + (size_t)(2 * t) * lut_sz : NULL,
                             qs_gu ? qs_gu + (size_t)(2 * t) * nsc : NULL);
                vq_build_lut(m, lut_gu + (size_t)(2 * t + 1) * lut_sz,
                             h->codebook_id + 1 * m->stages,
                             xin + (size_t)t * lat, lat, m->stages,
                             m->cb_entries, m->vec_dim,
                             q_gu ? q_gu + (size_t)(2 * t + 1) * lut_sz : NULL,
                             qs_gu ? qs_gu + (size_t)(2 * t + 1) * nsc : NULL);
            }
            lut_ready = 1;
        }
        for (int t = 0; t < nT; t++) {
            float wj = 0;
            for (int j = 0; j < K; j++)
                if (route[(size_t)t * K + j] == e) { wj = rw[(size_t)t * K + j]; break; }
            if (wj == 0.0f) continue;

            vq_apply(m, ga, rec + h->gate_off, s16, inter, lat,
                     lut_gu + (size_t)(2 * t) * lut_sz,
                     q_gu ? q_gu + (size_t)(2 * t) * lut_sz : NULL,
                     qs_gu ? qs_gu + (size_t)(2 * t) * nsc : NULL);
            vq_apply(m, ub, rec + h->up_off, s16 + inter, inter, lat,
                     lut_gu + (size_t)(2 * t + 1) * lut_sz,
                     q_gu ? q_gu + (size_t)(2 * t + 1) * lut_sz : NULL,
                     qs_gu ? qs_gu + (size_t)(2 * t + 1) * nsc : NULL);
            waste_act_pair_range(c, ga, ub, inter);
            vq_matvec(m, acc, rec + h->down_off, s16 + 2 * inter, ga, lat, inter,
                      h->codebook_id + 2 * m->stages, lut_down,
                      q_down, qs_down);
            float *dst = ysum + (size_t)t * lat;
            for (int i = 0; i < lat; i++) dst[i] += wj * acc[i];
        }
        PROF_END(P_EMM);
    }
    }
chunk_lost:

    if (c->latent_dim) {
        if (c->latent_norm) {
            const float *nw = waste_find(m, tname(
                "%smodel.layers.%d.block_sparse_moe.routed_expert_norm.weight",
                c->prefix, L))->data;
            for (int t = 0; t < nT; t++)
                waste_rmsnorm(ysum + (size_t)t * lat, ysum + (size_t)t * lat, nw, lat, c->eps);
        }
        waste_matmul_t(m, out, waste_find(m, tname(
                     "%smodel.layers.%d.block_sparse_moe.routed_expert_up_proj.weight",
                     c->prefix, L)), ysum, hid, lat, nT);
    } else {
        memcpy(out, ysum, (size_t)nT * hid * sizeof(float));
    }

    /* shared experts, on the full hidden state */
    const int si = inter * (c->n_shared ? c->n_shared : 1);
    float *sa = m->ckv, *sb = sa + (size_t)nT * si, *sh = sb + (size_t)nT * si;
    waste_matmul_t(m, sa, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.gate_proj.weight",
                 c->prefix, L)), in, si, hid, nT);
    waste_matmul_t(m, sb, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.up_proj.weight",
                 c->prefix, L)), in, si, hid, nT);
    waste_act_pair_range(c, sa, sb, nT * si);
    waste_matmul_t(m, sh, waste_find(m, tname(
                 "%smodel.layers.%d.block_sparse_moe.shared_experts.down_proj.weight",
                 c->prefix, L)), sa, hid, si, nT);
    for (int i = 0; i < nT * hid; i++) out[i] += sh[i];
}

/* Prefill a chunk. KDA and MLA still walk the tokens in order — the
 * recurrence and the causal mask demand it, and both are cheap — but every
 * projection is a GEMM and the MoE sees the whole chunk at once. */
/* The public API rejects out-of-range ids; this is the backstop for the
 * internal entry points, because the id indexes the embedding table
 * directly and a miss is an out-of-bounds read rather than a wrong
 * answer. Clamp and say so once, instead of reading past the table. */
static int clamp_token(const waste_model *m, int token)
{
    if (token >= 0 && token < m->cfg.vocab) return token;
    static int warned = 0;
    if (!warned) {
        warned = 1;
        fprintf(stderr, "waste: token id %d outside vocabulary of %d, clamped\n",
                token, m->cfg.vocab);
    }
    return 0;
}

const float *waste_model_prefill(waste_model *m, const int *tokens, int n,
                                 int pos0)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    if (n <= 0) return m->logits;
    if (c->arch_qwen) {
        const float *lg = NULL;
        for (int t = 0; t < n; t++) {
            lg = waste_model_step(m, tokens[t], pos0 + t, NULL);
            if (!lg) return NULL;
        }
        return lg;
    }
    if (n == 1) return waste_model_step(m, tokens[0], pos0, NULL);
    /* The chunked path carries one residual per token and one dense
     * attention per layer. mHC's parallel streams and the DSA indexer's
     * per-token pool bookkeeping are neither, and a chunk that quietly ran
     * without them would differ from the same prompt decoded one token at a
     * time — the exact failure WASTE_CHUNK exists to be checked against. So
     * a container with either is prefilled the way it decodes, one token
     * per call, until the chunked path grows both. */
    if (c->hc_mult || c->index_topk) {
        const float *out = NULL;
        for (int t = 0; t < n; t++) {
            out = waste_model_step(m, tokens[t], pos0 + t, NULL);
            if (!out) return NULL;
        }
        return out;
    }
    dump_pos0 = pos0;
    if (n > WASTE_CHUNK_MAX) n = WASTE_CHUNK_MAX;
    /* mla_layer writes one latent per position with no bound of its own,
     * so the bound is here. The public API refuses an over-long prompt
     * before it reaches this; a direct model.h caller gets a NULL. */
    {
        const int cm = waste_model_ctx_max(m);
        if (cm && (pos0 < 0 || pos0 > cm - n)) { m->ctx_full = 1; return NULL; }
    }
    if (prefill_alloc(m, n)) return NULL;

    for (int t = 0; t < n; t++) {
        float *dst = m->cx + (size_t)t * hid;
        /* An image token has no embedding of its own: the vision tower
         * supplies one per merged patch, and the caller expanded the
         * placeholder into that many positions before getting here. Rows
         * are consumed in order, which is what makes several images in one
         * prompt work without tracking which is which. */
        if (m->media && m->media_used < m->media_n &&
            tokens[t] == m->cfg_media_token) {
            memcpy(dst, m->media + (size_t)m->media_used * hid,
                   (size_t)hid * sizeof(float));
            m->media_used++;
            continue;
        }
        waste_embed_row(m, tokens[t], dst);
    }

    const int ares_on = c->attn_res_block > 0;
    int nb = 0;
    int ps_live = 0;

    for (int L = 0; L < c->n_layers; L++) {
        /* A record already failed: stop instead of streaming the rest of
         * the layers from a container that has been shown to be wrong.
         * On K3 that is gigabytes of pointless reads per token. */
        if (m->read_error) break;
        if (ares_on) {
            memcpy(m->cprefix, m->cx, (size_t)n * hid * sizeof(float));
            ps_live = 1;
            if (nb > 0) {
                const float *nw = waste_find(m, tname("%smodel.layers.%d.self_attention_res_norm.weight", c->prefix, L))->data;
                const float *pw = waste_find(m, tname("%smodel.layers.%d.self_attention_res_proj.weight", c->prefix, L))->data;
                const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
                for (int t = 0; t < n; t++)
                    waste_apply_attn_res(m, m->cblockres + (size_t)t * stride, nb,
                                         m->cprefix + (size_t)t * hid, nw, pw,
                                         m->cx + (size_t)t * hid);
            }
            if (L % c->attn_res_block == 0) {
                const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
                for (int t = 0; t < n; t++)
                    memcpy(m->cblockres + (size_t)t * stride + (size_t)nb * hid,
                           m->cprefix + (size_t)t * hid, (size_t)hid * sizeof(float));
                nb++;
                ps_live = 0;
            }
        }

        const float *iln = waste_find(m, tname("%smodel.layers.%d.input_layernorm.weight", c->prefix, L))->data;
        for (int t = 0; t < n; t++)
            waste_rmsnorm(m->cnorm + (size_t)t * hid, m->cx + (size_t)t * hid, iln, hid, c->eps);

        /* attention: per token, but on the batched norm buffer */
        for (int t = 0; t < n; t++) {
            if (c->kda_layer[L]) kda_layer(m, L, m->cnorm + (size_t)t * hid,
                                           m->cresid + (size_t)t * hid);
            else mla_layer(m, L, m->cnorm + (size_t)t * hid,
                           m->cresid + (size_t)t * hid, pos0 + t);
        }

        if (ares_on) {
            if (ps_live) for (int i = 0; i < n * hid; i++) m->cprefix[i] += m->cresid[i];
            else { memcpy(m->cprefix, m->cresid, (size_t)n * hid * sizeof(float)); ps_live = 1; }
            const float *nw = waste_find(m, tname("%smodel.layers.%d.mlp_res_norm.weight", c->prefix, L))->data;
            const float *pw = waste_find(m, tname("%smodel.layers.%d.mlp_res_proj.weight", c->prefix, L))->data;
            const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
            for (int t = 0; t < n; t++)
                waste_apply_attn_res(m, m->cblockres + (size_t)t * stride, nb,
                                     m->cprefix + (size_t)t * hid, nw, pw,
                                     m->cx + (size_t)t * hid);
        } else {
            for (int i = 0; i < n * hid; i++) m->cx[i] += m->cresid[i];
        }

        const float *pln = waste_find(m, tname("%smodel.layers.%d.post_attention_layernorm.weight", c->prefix, L))->data;
        for (int t = 0; t < n; t++)
            waste_rmsnorm(m->cnorm + (size_t)t * hid, m->cx + (size_t)t * hid, pln, hid, c->eps);

        if (waste_find(m, tname("%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L))) {
            PROF_START(P_ROUTE);
            moe_chunk(m, L, m->cnorm, m->cresid, n);
            PROF_END(P_ROUTE);
        } else {
            const int inter = c->dense_inter;
            float *a = m->cff, *b = a + (size_t)n * inter;
            waste_matmul_t(m, a, waste_find(m, tname("%smodel.layers.%d.mlp.gate_proj.weight", c->prefix, L)), m->cnorm, inter, hid, n);
            waste_matmul_t(m, b, waste_find(m, tname("%smodel.layers.%d.mlp.up_proj.weight", c->prefix, L)), m->cnorm, inter, hid, n);
            waste_act_pair_range(c, a, b, n * inter);
            waste_matmul_t(m, m->cresid, waste_find(m, tname("%smodel.layers.%d.mlp.down_proj.weight", c->prefix, L)), a, hid, inter, n);
        }

        if (ares_on) {
            for (int i = 0; i < n * hid; i++) m->cprefix[i] += m->cresid[i];
            memcpy(m->cx, m->cprefix, (size_t)n * hid * sizeof(float));
        } else {
            for (int i = 0; i < n * hid; i++) m->cx[i] += m->cresid[i];
        }
    }

    /* hand the chunk's block-residual history to the per-token path: decode
     * continues from the last token's row */
    if (ares_on) {
        const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
        memcpy(m->blockres, m->cblockres + (size_t)(n - 1) * stride,
               (size_t)nb * hid * sizeof(float));
    }
    m->n_blockres = nb;
    if (ares_on && nb > 0) {
        const float *onw = waste_find(m, tname("%smodel.output_attn_res_norm.weight", c->prefix))->data;
        const float *opw = waste_find(m, tname("%smodel.output_attn_res_proj.weight", c->prefix))->data;
        const int stride = (c->n_layers / c->attn_res_block + 2) * hid;
        for (int t = 0; t < n; t++)
            waste_apply_attn_res(m, m->cblockres + (size_t)t * stride, nb,
                                 m->cx + (size_t)t * hid, onw, opw,
                                 m->cx + (size_t)t * hid);
    }
    const float *fnw = waste_find(m, tname("%smodel.norm.weight", c->prefix))->data;
    float *last = m->cnorm;
    waste_rmsnorm(last, m->cx + (size_t)(n - 1) * hid, fnw, hid, c->eps);
    matvec_t(m, m->logits, waste_find(m, tname("%slm_head.weight", c->prefix)), last,
             c->vocab, hid);
    memcpy(m->x, m->cx + (size_t)(n - 1) * hid, (size_t)hid * sizeof(float));
    return m->read_error ? NULL : m->logits;
}

/* ---- Qwen3.8-Flash-Next forward (not KDA/MLA/AttnRes) ---------------- */

static uint16_t f32_to_bf16(float x)
{
    union { float f; uint32_t u; } a;
    a.f = x;
    const uint32_t u = a.u;
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t b)
{
    union { float f; uint32_t u; } a;
    a.u = (uint32_t)b << 16;
    return a.f;
}

static void qwen_row(waste_model *m, const waste_tensor *t, long row, float *dst)
{
    const int cols = t->shape[t->ndim - 1];
    if (!t->on_disk && t->data) {
        memcpy(dst, t->data + (size_t)row * (size_t)cols, (size_t)cols * sizeof(float));
        return;
    }
    if (!t->on_disk && t->q) {
        waste_deq_row(t, row, cols, dst);
        return;
    }
    const int g = t->group, ng = (cols + g - 1) / g;
    const int8_t *q; const uint16_t *sc;
    trunk_row(m, t, row, &q, &sc);
    for (int k = 0; k < ng; k++) {
        const float sv = f16_to_f32(sc[k]);
        for (int i = 0; i < g && k * g + i < cols; i++) {
            int v;
            if (t->bits == 4) {
                const uint8_t byte = ((const uint8_t *)q)[(k * g + i) / 2];
                v = (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8;
            } else {
                v = q[k * g + i];
            }
            dst[k * g + i] = (float)v * sv;
        }
    }
}

static void qwen_rope_cs(const waste_config *c, int pos, float *cos, float *sin)
{
    const int half = c->rotary_dim / 2;
    float ft[WASTE_MAX_ROPE_HALF], fh[WASTE_MAX_ROPE_HALF];
    float fw[WASTE_MAX_ROPE_HALF], freqs[WASTE_MAX_ROPE_HALF];
    if (half <= 0 || half > WASTE_MAX_ROPE_HALF) return;
    for (int j = 0; j < half; j++) {
        const float a = (float)pos * c->rope_inv_freq[j];
        ft[j] = fh[j] = fw[j] = a;
    }
    waste_qwen_mrope_interleave(ft, fh, fw, c->mrope_section, half, freqs);
    for (int j = 0; j < half; j++) {
        const float cj = cosf(freqs[j]), sj = sinf(freqs[j]);
        cos[j] = cj; sin[j] = sj;
        cos[j + half] = cj; sin[j + half] = sj;
    }
}

static void qwen_hc_mix_t(waste_model *m, const float *hyper,
                          const waste_tensor *nw, const waste_tensor *down,
                          const waste_tensor *up, const waste_tensor *inject,
                          int use_inj, float *mixed, float *inj_w)
{
    const waste_config *c = &m->cfg;
    const int hc = c->hc_count, hid = c->hidden, rank = c->hc_lowrank;
    const int H = hc * hid;
    if (!nw || !nw->data || !down || !up) {
        memset(mixed, 0, (size_t)hid * sizeof(float));
        if (inj_w) memset(inj_w, 0, (size_t)hc * sizeof(float));
        return;
    }
    float *normed = m->tmp;
    float *lo = normed + H;
    float *gate = lo + rank;
    waste_qwen_rmsnorm(normed, hyper, nw->data, H, hid, c->eps);
    matvec_t(m, lo, down, normed, rank, H);
    for (int i = 0; i < rank; i++) lo[i] = silu(lo[i] / (float)hc);
    matvec_t(m, gate, up, lo, H, rank);
    for (int i = 0; i < H; i++) gate[i] = 1.0f / (1.0f + expf(-gate[i]));
    for (int d = 0; d < hid; d++) {
        float s = 0.0f;
        for (int b = 0; b < hc; b++) s += gate[b * hid + d] * normed[b * hid + d];
        mixed[d] = s / (float)hc;
    }
    if (use_inj && inject && inj_w) {
        float tmpi[16];
        matvec_t(m, tmpi, inject, normed, hc, H);
        for (int b = 0; b < hc; b++)
            inj_w[b] = 2.0f / (1.0f + expf(-tmpi[b] / (float)hc));
    }
}

static void qwen_dilated_conv_step(int C, int KS, int dil, const float *w,
                                   float *ring, const float *x, float *y)
{
    const int R = (KS - 1) * dil;
    for (int c = 0; c < C; c++) {
        const float *wc = w + (size_t)c * KS;
        float *rc = ring + (size_t)c * R;
        float acc = x[c] * wc[KS - 1];
        for (int k = 0; k < KS - 1; k++)
            acc += rc[k * dil] * wc[k];
        for (int j = 0; j + 1 < R; j++) rc[j] = rc[j + 1];
        if (R > 0) rc[R - 1] = x[c];
        y[c] = silu(acc);
    }
}

static void qwen_ple_inject(waste_model *m, int token)
{
    const waste_config *c = &m->cfg;
    const int L = c->ple_layer;
    if (L < 0) return;
    const int hid = c->hidden, hc = c->hc_count, H = hc * hid;
    const int pe = c->ple_embed ? c->ple_embed : hid;
    const int ngram = c->ngram_size > 0 ? c->ngram_size : 3;
    const int heads = (ngram - 1) * (c->heads_per_ngram ? c->heads_per_ngram : 8);
    const int ctxn = ngram - 1;
    int ids[8];
    for (int i = 0; i < ctxn && i < 8; i++) ids[i] = m->ple_prev[i];
    ids[ctxn] = token;
    const int n = ctxn + 1;
    int local[WASTE_QWEN_PLE_HEADS];
    waste_qwen_ple_row_ids(ids, n, ctxn, c->eos_token_id, ngram,
                           c->heads_per_ngram ? c->heads_per_ngram : 8,
                           c->ple_mult, c->ple_sz, local);
    float *emb = m->ple_emb;
    if (!emb) return;
    memset(emb, 0, (size_t)pe * sizeof(float));
    int off = 0;
    for (int h = 0; h < heads && h < WASTE_QWEN_PLE_HEADS; h++) {
        if (c->ple_sz[h] <= 0) return;
        const waste_tensor *ht = waste_find(m, tname(
            "%smodel.layers.%d.ple.ple_embedding.ngram_head.%d.weight",
            c->prefix, L, h));
        if (!ht) continue;
        const int width = ht->shape[ht->ndim - 1];
        if (off + width > pe) break;
        qwen_row(m, ht, local[h], emb + off);
        m->ple_reads++;
        off += width;
    }
    float *key = m->tmp, *val = key + H, *qnorm = val + hid;
    matvec_t(m, key, waste_find(m, tname("%smodel.layers.%d.ple.key_proj.weight",
                                         c->prefix, L)), emb, H, pe);
    matvec_t(m, val, waste_find(m, tname("%smodel.layers.%d.ple.value_proj.weight",
                                         c->prefix, L)), emb, hid, pe);
    const waste_tensor *tnk = waste_find(m, tname("%smodel.layers.%d.ple.norm_key.weight",
                                                  c->prefix, L));
    const waste_tensor *tnq = waste_find(m, tname("%smodel.layers.%d.ple.norm_query.weight",
                                                  c->prefix, L));
    const waste_tensor *tnc = waste_find(m, tname("%smodel.layers.%d.ple.norm_conv.weight",
                                                  c->prefix, L));
    if (!tnk || !tnk->data || !tnq || !tnq->data || !tnc || !tnc->data) return;
    const float *nk = tnk->data, *nq = tnq->data, *nc = tnc->data;
    waste_qwen_rmsnorm(key, key, nk, H, hid, c->eps);
    waste_qwen_rmsnorm(qnorm, m->hcx, nq, H, hid, c->eps);
    float *gated = qnorm + H;
    const float inv = 1.0f / sqrtf((float)hid);
    for (int b = 0; b < hc; b++) {
        float g = 0.0f;
        for (int d = 0; d < hid; d++)
            g += key[b * hid + d] * qnorm[b * hid + d];
        g *= inv;
        const float mag = sqrtf(fabsf(g) < 1e-6f ? 1e-6f : fabsf(g));
        g = copysignf(mag, g);
        const float sg = 1.0f / (1.0f + expf(-g));
        for (int d = 0; d < hid; d++) gated[b * hid + d] = sg * val[d];
    }
    float *gnorm = gated + H;
    waste_qwen_rmsnorm(gnorm, gated, nc, H, hid, c->eps);
    const waste_tensor *cw = waste_find(m, tname("%smodel.layers.%d.ple.conv1d.weight",
                                                 c->prefix, L));
    const int KS = c->ple_conv_k > 0 ? c->ple_conv_k : 4;
    float *conv_y = gnorm + H;
    if (cw && cw->data)
        qwen_dilated_conv_step(H, KS, ngram, cw->data, m->ple_ring, gnorm, conv_y);
    else
        memcpy(conv_y, gnorm, (size_t)H * sizeof(float));
    for (int i = 0; i < H; i++) m->hcx[i] += gated[i] + conv_y[i];
    for (int i = 0; i < ctxn - 1 && i < 7; i++) m->ple_prev[i] = m->ple_prev[i + 1];
    if (ctxn > 0) m->ple_prev[ctxn - 1] = token;
}

static void qwen_gdn_layer(waste_model *m, int L, const float *in, float *out)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, Hk = c->gdn_k_heads, Hv = c->gdn_v_heads;
    const int Dk = c->gdn_k_dim, Dv = c->gdn_v_dim;
    const int qkv = 2 * Hk * Dk + Hv * Dv;
    memset(out, 0, (size_t)hid * sizeof(float));
    if (!m->gdn_g) return;
    float *mixed = m->tmp;
    float *conv_y = mixed + qkv;
    float *z = conv_y + qkv;
    float *a = z + Hv * Dv;
    float *b = a + Hv;
    float *core = b + Hv;
    matvec_t(m, mixed, waste_find(m, tname("%smodel.layers.%d.linear_attn.in_proj_qkv.weight",
                                           c->prefix, L)), in, qkv, hid);
    const waste_tensor *cw = waste_find(m, tname("%smodel.layers.%d.linear_attn.conv1d.weight",
                                                 c->prefix, L));
    if (cw && cw->data)
        waste_k.short_conv_step(qkv, c->conv_k, cw->data, NULL, m->conv[L], mixed, conv_y);
    else
        memcpy(conv_y, mixed, (size_t)qkv * sizeof(float));
    matvec_t(m, z, waste_find(m, tname("%smodel.layers.%d.linear_attn.in_proj_z.weight",
                                       c->prefix, L)), in, Hv * Dv, hid);
    matvec_t(m, a, waste_find(m, tname("%smodel.layers.%d.linear_attn.in_proj_a.weight",
                                       c->prefix, L)), in, Hv, hid);
    matvec_t(m, b, waste_find(m, tname("%smodel.layers.%d.linear_attn.in_proj_b.weight",
                                       c->prefix, L)), in, Hv, hid);
    for (int h = 0; h < Hv; h++) b[h] = 1.0f / (1.0f + expf(-b[h]));
    const waste_tensor *tA = waste_find(m, tname("%smodel.layers.%d.linear_attn.A_log",
                                                 c->prefix, L));
    const waste_tensor *tdt = waste_find(m, tname("%smodel.layers.%d.linear_attn.dt_bias",
                                                  c->prefix, L));
    if (!tA || !tA->data || !tdt || !tdt->data) return;
    waste_qwen_gdn_decay(a, tA->data, tdt->data, Hv, m->gdn_g);
    const float *q = conv_y;
    const float *k = conv_y + Hk * Dk;
    const float *v = conv_y + 2 * Hk * Dk;
    waste_qwen_gdn_step(Hk, Hv, Dk, Dv, q, k, v, m->gdn_g, b, m->S[L], core, m->att);
    const waste_tensor *tnw = waste_find(m, tname("%smodel.layers.%d.linear_attn.norm.weight",
                                                  c->prefix, L));
    if (!tnw || !tnw->data) return;
    float *normed = mixed;
    for (int h = 0; h < Hv; h++)
        waste_k.rmsnorm_gated(Dv, core + (size_t)h * Dv, z + (size_t)h * Dv,
                              tnw->data, c->eps, normed + (size_t)h * Dv);
    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.linear_attn.out_proj.weight",
                                         c->prefix, L)), normed, hid, Hv * Dv);
}

static void qwen_qsa_layer(waste_model *m, int L, const float *in, float *out, int pos)
{
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, Hq = c->n_heads, Hkv = c->qsa_n_kv, D = c->qsa_head_dim;
    const int Dk = c->idx_head_dim, compress = c->idx_compress > 0 ? c->idx_compress : 4;
    const int qd = Hq * D, kvd = Hkv * D;
    const int idxd = (c->idx_n_heads + c->idx_kv_heads) * Dk;
    const int rot = c->rotary_dim;
    memset(out, 0, (size_t)hid * sizeof(float));
    if (!m->qsa_q || !m->qsa_gate || !m->qsa_attn || !m->qsa_sel ||
        !m->qsa_kf || !m->qsa_vf || !m->qsa_rawk[L])
        return;
    float *qgate = m->tmp;
    float *k = qgate + qd * 2;
    float *v = k + kvd;
    float *idx = v + kvd;
    matvec_t(m, qgate, waste_find(m, tname("%smodel.layers.%d.self_attn.q_proj.weight",
                                           c->prefix, L)), in, qd * 2, hid);
    matvec_t(m, k, waste_find(m, tname("%smodel.layers.%d.self_attn.k_proj.weight",
                                       c->prefix, L)), in, kvd, hid);
    matvec_t(m, v, waste_find(m, tname("%smodel.layers.%d.self_attn.v_proj.weight",
                                       c->prefix, L)), in, kvd, hid);
    matvec_t(m, idx, waste_find(m, tname(
        "%smodel.layers.%d.self_attn.indexer.index_qk_proj.weight",
        c->prefix, L)), in, idxd, hid);
    float *q = m->qsa_q, *gate = m->qsa_gate;
    for (int h = 0; h < Hq; h++) {
        memcpy(q + (size_t)h * D, qgate + (size_t)h * 2 * D, (size_t)D * sizeof(float));
        memcpy(gate + (size_t)h * D, qgate + (size_t)h * 2 * D + D, (size_t)D * sizeof(float));
    }
    const waste_tensor *tqn = waste_find(m, tname("%smodel.layers.%d.self_attn.q_norm.weight",
                                                  c->prefix, L));
    const waste_tensor *tkn = waste_find(m, tname("%smodel.layers.%d.self_attn.k_norm.weight",
                                                  c->prefix, L));
    if (!tqn || !tqn->data || !tkn || !tkn->data) return;
    for (int h = 0; h < Hq; h++)
        waste_qwen_rmsnorm(q + (size_t)h * D, q + (size_t)h * D, tqn->data, D, D, c->eps);
    for (int h = 0; h < Hkv; h++)
        waste_qwen_rmsnorm(k + (size_t)h * D, k + (size_t)h * D, tkn->data, D, D, c->eps);
    float cos[256], sin[256];
    qwen_rope_cs(c, pos, cos, sin);
    for (int h = 0; h < Hq; h++)
        if (waste_qwen_rope_apply(q + (size_t)h * D, D, cos, sin, rot) != 0) return;
    for (int h = 0; h < Hkv; h++)
        if (waste_qwen_rope_apply(k + (size_t)h * D, D, cos, sin, rot) != 0) return;

    if (pos >= 0 && pos < m->kv_cap) {
        uint16_t *kb = m->qsa_k[L] + (size_t)pos * Hkv * D;
        uint16_t *vb = m->qsa_v[L] + (size_t)pos * Hkv * D;
        for (int i = 0; i < kvd; i++) {
            kb[i] = f32_to_bf16(k[i]);
            vb[i] = f32_to_bf16(v[i]);
        }
        m->n_kv[L] = pos + 1;
    }
    const int iq = c->idx_n_heads * Dk;
    float *q_idx = idx;
    float *raw_k = idx + iq;
    const waste_tensor *tqln = waste_find(m, tname(
        "%smodel.layers.%d.self_attn.indexer.q_layernorm.weight", c->prefix, L));
    const waste_tensor *tkln = waste_find(m, tname(
        "%smodel.layers.%d.self_attn.indexer.k_layernorm.weight", c->prefix, L));
    if (!tqln || !tqln->data || !tkln || !tkln->data) return;
    for (int h = 0; h < c->idx_n_heads; h++) {
        waste_qwen_rmsnorm(q_idx + (size_t)h * Dk, q_idx + (size_t)h * Dk,
                           tqln->data, Dk, Dk, c->eps);
        if (waste_qwen_rope_apply(q_idx + (size_t)h * Dk, Dk, cos, sin, rot) != 0)
            return;
    }
    if (pos >= 0 && pos < m->kv_cap && m->qsa_rawk[L])
        memcpy(m->qsa_rawk[L] + (size_t)pos * Dk, raw_k, (size_t)Dk * sizeof(float));
    const int T = m->n_kv[L];
    m->n_qsa_blk[L] = compress > 0 ? T / compress : 0;
    m->n_qsa_tail[L] = compress > 0 ? T % compress : 0;

    const int block_topk = c->idx_budget / compress;
    float *full_cos = m->qsa_cs;
    float *full_sin = m->qsa_cs + (size_t)m->kv_cap * (rot > 0 ? rot : 1);
    if (full_cos && rot > 0) {
        for (int t = 0; t < T; t++)
            qwen_rope_cs(c, t, full_cos + (size_t)t * rot, full_sin + (size_t)t * rot);
    }
    int nsel = waste_qwen_qsa_select(q_idx, c->idx_n_heads, Dk, m->qsa_rawk[L], T, pos,
                                     full_cos, full_sin, rot, tkln->data, c->eps,
                                     compress, block_topk, m->qsa_sel, m->qsa_work,
                                     m->qsa_taken);
    float *kf = m->qsa_kf, *vf = m->qsa_vf, *attn = m->qsa_attn, *scr = m->qsa_scr;
    for (int i = 0; i < nsel; i++) {
        const int t = m->qsa_sel[i];
        m->qsa_sel[i] = i;
        const uint16_t *kb = m->qsa_k[L] + (size_t)t * Hkv * D;
        const uint16_t *vb = m->qsa_v[L] + (size_t)t * Hkv * D;
        for (int j = 0; j < kvd; j++) {
            kf[(size_t)i * kvd + j] = (t >= 0 && t < T) ? bf16_to_f32(kb[j]) : 0.0f;
            vf[(size_t)i * kvd + j] = (t >= 0 && t < T) ? bf16_to_f32(vb[j]) : 0.0f;
        }
    }
    if (nsel > 0)
        waste_qwen_qsa_attn(q, Hq, D, kf, vf, Hkv, nsel, m->qsa_sel, nsel,
                            1.0f / sqrtf((float)D), attn, scr);
    else
        memset(attn, 0, (size_t)qd * sizeof(float));
    for (int i = 0; i < qd; i++)
        attn[i] *= 1.0f / (1.0f + expf(-gate[i]));
    matvec_t(m, out, waste_find(m, tname("%smodel.layers.%d.self_attn.o_proj.weight",
                                         c->prefix, L)), attn, hid, qd);
}

static void qwen_moe_layer(waste_model *m, int L, const float *in, float *out, int *routed)
{
    const waste_config *c = &m->cfg;
    const int E = c->n_experts, K = c->top_k, hid = c->hidden, inter = c->moe_inter;
    float *sc = m->att + WASTE_ATT_ROUTER_OFF;
    matvec_t(m, sc, waste_find(m, tname("%smodel.layers.%d.mlp.gate.weight",
                                        c->prefix, L)), in, E, hid);
    int idx[64];
    float w[64];
    if (waste_qwen_moe_route(sc, E, K, c->renorm, idx, w, m->moe_prob, m->moe_used) != 0) {
        memset(out, 0, (size_t)hid * sizeof(float));
        return;
    }
    if (routed) for (int j = 0; j < K; j++) routed[j] = idx[j];
    if (dump_route) {
        FILE *df = fopen(dump_route, (dump_pos0 || L) ? "a" : "wb");
        if (df) {
            fprintf(df, "%d %d", dump_pos0, L);
            for (int j = 0; j < K; j++) fprintf(df, " %d", idx[j]);
            for (int j = 0; j < K; j++) fprintf(df, " %.6g", w[j]);
            for (int j = 0; j < K; j++) fprintf(df, " -1");
            fputc('\n', df);
            fclose(df);
        }
    }
    waste_ecache_hint(&m->cache, L, idx, K);
    memset(out, 0, (size_t)hid * sizeof(float));
    float *ga = m->ff, *ub = ga + inter, *acc = m->e_gate;
    const int lut_sz = (hid / m->vec_dim) * m->stages * m->cb_entries;
    float *lut_gate = m->lut, *lut_up = lut_gate + lut_sz, *lut_down = lut_up + lut_sz;

    /* Same expert-parallel path the Kimi/GLM router takes, and the same
     * question decides it: one task per expert wins when the records are
     * already resident, and loses when holding a batch barriers the
     * read-ahead. Asked per layer and per token from the cache, not from
     * the architecture — see moe_layer, which explains what it decides on.
     * WASTE_XPAR=0/1 still forces it either way. */
    const int xpar_here = xpar_on >= 0
        ? xpar_on
        : waste_ecache_resident_all(&m->cache, L, idx, K);
    if (xpar_here && m->xga && K > 1 && K <= WASTE_PF_MAX &&
        m->cache.n_slots >= 4 * K) {
        const uint8_t *recs[WASTE_PF_MAX];
        int lut_done = 0, ok = 1;
        for (int j0 = 0; j0 < K; j0 += xpar_batch) {
            int j1 = j0 + xpar_batch;
            if (j1 > K) j1 = K;
            PROF_START(P_EDEQ);
            int n = j0;
            for (; n < j1; n++) {
                recs[n] = waste_ecache_hold(&m->cache, L, idx[n], bank_fetch, m);
                if (!recs[n]) break;
            }
            PROF_END(P_EDEQ);
            if (n < j1) { ok = 0; break; }
            PROF_START(P_EMM);
            if (!lut_done) {
                const waste_expert_hdr *h0 = (const waste_expert_hdr *)recs[0];
                vq_build_lut(m, lut_gate, h0->codebook_id + 0 * m->stages,
                             in, hid, m->stages, m->cb_entries, m->vec_dim,
                             NULL, NULL);
                vq_build_lut(m, lut_up, h0->codebook_id + 1 * m->stages,
                             in, hid, m->stages, m->cb_entries, m->vec_dim,
                             NULL, NULL);
                lut_done = 1;
            }
            /* Qwen's experts read the hidden state directly: there is no
             * latent projection, so `lat` is the hidden size. */
            xpar_arg pa = { m, c, recs, w, j0, inter, hid,
                            lut_gate, lut_up, NULL, NULL, NULL, NULL };
            waste_parallel_for(j1 - j0, 1, moe_expert_range, &pa);
            PROF_END(P_EMM);
            waste_ecache_release(&m->cache);
        }
        if (ok) {
            /* Summed in j order, so the total does not depend on the
             * thread count or the batch size. */
            for (int j = 0; j < K; j++) {
                const float *accj = m->xacc + (size_t)j * hid;
                const float wj = w[j];
                for (int i = 0; i < hid; i++) out[i] += wj * accj[i];
            }
            goto qwen_moe_done;
        }
        /* Something did not read: let go of what was held and fall through
         * to the serial loop, which re-reads and reports the reason. */
        waste_ecache_release(&m->cache);
    }

    int lut_ready = 0;
    for (int j = 0; j < K; j++) {
        const uint8_t *rec = read_expert(m, L, idx[j]);
        if (!rec) break;
        const waste_expert_hdr *h = (const waste_expert_hdr *)rec;
        const uint16_t *corr = (const uint16_t *)(rec + h->chan_corr_off);
        if (!lut_ready) {
            vq_build_lut(m, lut_gate, h->codebook_id + 0 * m->stages,
                         in, hid, m->stages, m->cb_entries, m->vec_dim, NULL, NULL);
            vq_build_lut(m, lut_up, h->codebook_id + 1 * m->stages,
                         in, hid, m->stages, m->cb_entries, m->vec_dim, NULL, NULL);
            lut_ready = 1;
        }
        vq_apply(m, ga, rec + h->gate_off, corr, inter, hid, lut_gate, NULL, NULL);
        vq_apply(m, ub, rec + h->up_off, corr + inter, inter, hid, lut_up, NULL, NULL);
        for (int i = 0; i < inter; i++) ga[i] = silu(ga[i]) * ub[i];
        vq_matvec(m, acc, rec + h->down_off, corr + 2 * inter, ga, hid, inter,
                  h->codebook_id + 2 * m->stages, lut_down, NULL, NULL);
        const float wj = w[j];
        for (int i = 0; i < hid; i++) out[i] += wj * acc[i];
    }
qwen_moe_done: ;
    const int shared = c->shared_inter ? c->shared_inter : inter;
    ffn(m,
        waste_find(m, tname("%smodel.layers.%d.mlp.shared_expert.gate_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.mlp.shared_expert.up_proj.weight", c->prefix, L)),
        waste_find(m, tname("%smodel.layers.%d.mlp.shared_expert.down_proj.weight", c->prefix, L)),
        /* Not m->h: qwen_step passes it as `out`, and the routed sum is
         * already in there waiting for HyperConnection to consume it. The
         * expert accumulator is dead once the loop above has finished and
         * is sized for hid floats, so the shared expert lands there. */
        in, acc, shared, hid, 1.0f, 0);
    float sg;
    matvec_t(m, &sg, waste_find(m, tname("%smodel.layers.%d.mlp.shared_expert_gate.weight",
                                         c->prefix, L)), in, 1, hid);
    sg = 1.0f / (1.0f + expf(-sg));
    for (int i = 0; i < hid; i++) out[i] += sg * acc[i];
}

static const float *qwen_step(waste_model *m, int token, int pos, int *routed)
{
    dump_pos0 = pos;
    const waste_config *c = &m->cfg;
    const int hid = c->hidden, hc = c->hc_count;
    {
        const int cm = waste_model_ctx_max(m);
        if (cm && (pos < 0 || pos >= cm)) { m->ctx_full = 1; return NULL; }
    }
    waste_embed_row(m, token, m->x);
    for (int b = 0; b < hc; b++)
        memcpy(m->hcx + (size_t)b * hid, m->x, (size_t)hid * sizeof(float));

    float *block = m->h;
    for (int L = 0; L < c->n_layers; L++) {
        if (m->read_error) break;
        if (L == c->ple_layer) qwen_ple_inject(m, token);
        float inj[16];
        qwen_hc_mix_t(m, m->hcx,
            waste_find(m, tname("%smodel.layers.%d.attn_hyper_connection.hc_norm.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.attn_hyper_connection.input_mix_weight_down.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.attn_hyper_connection.input_mix_weight_up.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.attn_hyper_connection.block_inject_weight.weight", c->prefix, L)),
            1, m->x, inj);
        if (!c->qwen_full[L]) qwen_gdn_layer(m, L, m->x, block);
        else qwen_qsa_layer(m, L, m->x, block, pos);
        waste_qwen_hc_combine(m->hcx, block, inj, hc, hid, m->hcx);

        qwen_hc_mix_t(m, m->hcx,
            waste_find(m, tname("%smodel.layers.%d.mlp_hyper_connection.hc_norm.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.mlp_hyper_connection.input_mix_weight_down.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.mlp_hyper_connection.input_mix_weight_up.weight", c->prefix, L)),
            waste_find(m, tname("%smodel.layers.%d.mlp_hyper_connection.block_inject_weight.weight", c->prefix, L)),
            1, m->x, inj);
        qwen_moe_layer(m, L, m->x, block, routed ? routed + (size_t)L * c->top_k : NULL);
        waste_qwen_hc_combine(m->hcx, block, inj, hc, hid, m->hcx);
        /* Same role as the Kimi dump in waste_model_step: one residual
         * stream after every layer. Qwen's stream is the hc hyper-state. */
        const char *dump_hidden = getenv("WASTE_DUMP_HIDDEN");
        if (dump_hidden) {
            FILE *df = fopen(dump_hidden, (L || pos) ? "ab" : "wb");
            if (df) {
                fwrite(m->hcx, sizeof(float), (size_t)hc * (size_t)hid, df);
                fclose(df);
            }
        }
    }
    qwen_hc_mix_t(m, m->hcx,
        waste_find(m, tname("%smodel.hyper_connection_mixer.hc_norm.weight", c->prefix)),
        waste_find(m, tname("%smodel.hyper_connection_mixer.input_mix_weight_down.weight", c->prefix)),
        waste_find(m, tname("%smodel.hyper_connection_mixer.input_mix_weight_up.weight", c->prefix)),
        NULL, 0, m->x, NULL);
    matvec_t(m, m->logits, waste_find(m, tname("%slm_head.weight", c->prefix)), m->x,
             c->vocab, hid);
    return m->read_error ? NULL : m->logits;
}

const float *waste_model_step(waste_model *m, int token, int pos, int *routed)
{
    if (m->cfg.arch_qwen) return qwen_step(m, token, pos, routed);
    dump_pos0 = pos;
    const waste_config *c = &m->cfg;
    const int hid = c->hidden;
    {   /* see waste_model_prefill */
        const int cm = waste_model_ctx_max(m);
        if (cm && (pos < 0 || pos >= cm)) { m->ctx_full = 1; return NULL; }
    }
    /* one embedding row; the table may be kept quantized */
    /* Same splice as the prefill: a chunked prompt whose last chunk is a
     * single token comes through here, and if that token is a media
     * placeholder its embedding is the tower's, not the table's. Without
     * this the bug only appears when an image lands at the very end of a
     * prompt or its token count divides badly — which is to say, rarely
     * and confusingly. */
    if (m->media && m->media_used < m->media_n && token == m->cfg_media_token) {
        memcpy(m->x, m->media + (size_t)m->media_used * hid,
               (size_t)hid * sizeof(float));
        m->media_used++;
    } else {
        waste_embed_row(m, token, m->x);
    }
    /* mHC: every stream starts as a copy of the embedding — upstream
     * expands the embedding along a new axis of hc_mult before layer 0. */
    for (int i = 1; i < c->hc_mult; i++)
        memcpy(m->x + (size_t)i * hid, m->x, (size_t)hid * sizeof(float));

    float *resid = (float *)malloc((size_t)hid * sizeof(float));
    float *norm = (float *)malloc((size_t)hid * sizeof(float));
    if (!resid || !norm) { free(resid); free(norm); return NULL; }
    const int ares_on = c->attn_res_block > 0;
    const int hc_on = c->hc_mult > 0;
    float hc_post[WASTE_MAX_HC], hc_comb[WASTE_MAX_HC * WASTE_MAX_HC];
    float *ps = m->prefix_sum;
    int ps_live = 0;
    m->n_blockres = 0;

    for (int L = 0; L < c->n_layers; L++) {
        if (m->read_error) break;        /* see waste_model_prefill */
        char b[128];
        if (ares_on) {
            memcpy(ps, m->x, (size_t)hid * sizeof(float));
            ps_live = 1;
            if (m->n_blockres > 0) {
                waste_apply_attn_res(m, m->blockres, m->n_blockres, ps,
                    waste_find(m, tname("%smodel.layers.%d.self_attention_res_norm.weight", c->prefix, L))->data,
                    waste_find(m, tname("%smodel.layers.%d.self_attention_res_proj.weight", c->prefix, L))->data,
                    m->x);
            }
            if (L % c->attn_res_block == 0) {
                memcpy(m->blockres + (size_t)m->n_blockres * hid, ps,
                       (size_t)hid * sizeof(float));
                m->n_blockres++;
                ps_live = 0;
            }
        }

        snprintf(b, sizeof b, "%smodel.layers.%d.input_layernorm.weight", c->prefix, L);
        if (hc_on) hc_collapse(m, L, "attn", m->x, hc_post, hc_comb, m->hccol);
        waste_rmsnorm(norm, hc_on ? m->hccol : m->x, waste_find(m, b)->data,
                      hid, c->eps);
        if (c->kda_layer[L]) { PROF_START(P_KDA); kda_layer(m, L, norm, resid); PROF_END(P_KDA); }
        else { PROF_START(P_MLA); mla_layer(m, L, norm, resid, pos); PROF_END(P_MLA); }

        if (hc_on) {
            hc_scatter(m, m->x, hc_post, hc_comb, resid);
        } else if (ares_on) {
            if (ps_live) for (int i = 0; i < hid; i++) ps[i] += resid[i];
            else { memcpy(ps, resid, (size_t)hid * sizeof(float)); ps_live = 1; }
            waste_apply_attn_res(m, m->blockres, m->n_blockres, ps,
                waste_find(m, tname("%smodel.layers.%d.mlp_res_norm.weight", c->prefix, L))->data,
                waste_find(m, tname("%smodel.layers.%d.mlp_res_proj.weight", c->prefix, L))->data,
                m->x);
        } else {
            for (int i = 0; i < hid; i++) m->x[i] += resid[i];
        }

        snprintf(b, sizeof b, "%smodel.layers.%d.post_attention_layernorm.weight", c->prefix, L);
        if (hc_on) hc_collapse(m, L, "ffn", m->x, hc_post, hc_comb, m->hccol);
        waste_rmsnorm(norm, hc_on ? m->hccol : m->x, waste_find(m, b)->data,
                      hid, c->eps);
        snprintf(b, sizeof b, "%smodel.layers.%d.block_sparse_moe.gate.weight", c->prefix, L);
        if (waste_find(m, b)) {
            PROF_START(P_ROUTE);
            moe_layer(m, L, norm, resid, routed ? routed + (size_t)L * c->top_k : NULL);
            PROF_END(P_ROUTE);
        }
        else
            ffn(m, waste_find(m, tname("%smodel.layers.%d.mlp.gate_proj.weight", c->prefix, L)),
                waste_find(m, tname("%smodel.layers.%d.mlp.up_proj.weight", c->prefix, L)),
                waste_find(m, tname("%smodel.layers.%d.mlp.down_proj.weight", c->prefix, L)),
                norm, resid, c->dense_inter, hid, 1.0f, 0);

        if (hc_on) {
            hc_scatter(m, m->x, hc_post, hc_comb, resid);
        } else if (ares_on) {
            for (int i = 0; i < hid; i++) ps[i] += resid[i];
            memcpy(m->x, ps, (size_t)hid * sizeof(float));
        } else {
            for (int i = 0; i < hid; i++) m->x[i] += resid[i];
        }
        /* WASTE_DUMP_HIDDEN=path appends the residual stream after every
         * layer, so a divergence against the PyTorch oracle can be bisected
         * to the layer that introduces it. */
        const char *dump_hidden = getenv("WASTE_DUMP_HIDDEN");
        if (dump_hidden) {
            FILE *df = fopen(dump_hidden, L ? "ab" : "wb");
            if (df) { fwrite(m->x, sizeof(float),
                             (size_t)hid * (hc_on ? c->hc_mult : 1), df);
                      fclose(df); }
        }
    }
    /* One last AttnRes: the output layer attends over every block
     * representation before the final norm (tech report §2.2, "the final
     * output layer then aggregates all N block representations"). */
    if (ares_on && m->n_blockres > 0)
        waste_apply_attn_res(m, m->blockres, m->n_blockres, m->x,
            waste_find(m, tname("%smodel.output_attn_res_norm.weight", c->prefix))->data,
            waste_find(m, tname("%smodel.output_attn_res_proj.weight", c->prefix))->data,
            m->x);
    if (hc_on) hc_head(c, m->hccol, m->x);
    waste_rmsnorm(norm, hc_on ? m->hccol : m->x,
                  waste_find(m, tname("%smodel.norm.weight", c->prefix))->data,
                  hid, c->eps);
    PROF_START(P_HEAD);
    matvec_t(m, m->logits, waste_find(m, tname("%slm_head.weight", c->prefix)), norm, c->vocab, hid);
    PROF_END(P_HEAD);
    free(resid);
    free(norm);
    return m->read_error ? NULL : m->logits;
}
