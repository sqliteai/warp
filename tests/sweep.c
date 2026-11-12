/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * sweep.c — one model load, many configurations, one machine state.
 *
 * A K3 measurement costs about 50 seconds of which 48 are model load and
 * prefill, and that is the smaller problem. The larger one is that every
 * heavy run changes the machine the next one lands on, so two arms measured
 * in two processes are measured on two computers: docs/LEARNED.md §32 and
 * §33 are both records of a conclusion that came out wrong for exactly that
 * reason, and §16's "sweep upward, never downward" exists because of it.
 *
 * So: load once, then run the arms back to back, interleaved, with the
 * expert cache cleared and the session reset between each. What is left
 * varying between two adjacent measurements is the setting and roughly
 * nothing else.
 *
 * The one thing it still cannot vary is the context length, which sizes the
 * KV and KDA state at load.
 *
 *   sweep CONTAINER ids,.. n_gen lookahead=0,6 [repeat]
 *   sweep CONTAINER ids,.. n_gen iodepth=2,4,8 [repeat]
 *   sweep CONTAINER ids,.. n_gen cache=3400,17736,23879 [repeat]
 *
 * `cache` is in MB and re-makes the expert cache in place. The trunk is what
 * a load costs, not the cache, so a budget sweep no longer needs a process
 * per budget — which is what made §32 and §33 come out wrong, each process
 * meeting a machine the one before it had changed. The footprint at each arm
 * is what that budget would have made it: the same trunk plus the cache
 * being asked for.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#include "../src/model.h"

/* Filled by src/model.c under WASTE_PROFILE=1. Reported per arm because
 * the question "does a big cache make the rest of the engine slower" is a
 * rate question, and a rate is only comparable inside one process. */
extern double waste_prof[32];
extern uint64_t waste_prof_n[32];
extern uint64_t waste_tmv_bytes;
extern int *waste_route_cap;
extern int waste_route_n, waste_route_cap_n;
extern double waste_tcheck_num, waste_tcheck_den, waste_tcheck_max;
extern unsigned long long waste_tcheck_n;
#if defined(WASTE_ENABLE_METAL)
extern double waste_metal_t_gpu, waste_metal_t_copy, waste_metal_t_wrap;
extern unsigned long long waste_metal_n_gpu, waste_metal_n_fallback, waste_metal_bytes;
#endif

static double now(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

#define MAX_ARMS 16
#define MAX_IDS 512

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s CONTAINER ids,.. n_gen KEY=v1,v2,.. [repeat]\n"
                "  KEY is lookahead, iodepth, cache (MB), topk, trunk or devkb\n", argv[0]);
        return 2;
    }
    int ids[MAX_IDS], n = 0;
    for (char *p = strtok(argv[2], ","); p && n < MAX_IDS; p = strtok(NULL, ","))
        ids[n++] = atoi(p);
    const int n_gen = atoi(argv[3]);

    char *eq = strchr(argv[4], '=');
    if (!eq) { fprintf(stderr, "expected KEY=v1,v2,..\n"); return 2; }
    *eq = 0;
    const char *key = argv[4];
    int arm[MAX_ARMS], n_arms = 0;
    for (char *p = strtok(eq + 1, ","); p && n_arms < MAX_ARMS; p = strtok(NULL, ","))
        arm[n_arms++] = atoi(p);
    const int repeat = argc > 5 ? atoi(argv[5]) : 3;
    const char *sgs = getenv("WASTE_SDOT4_SG");
    const int sdot4_sg_env = sgs ? atoi(sgs) : 32;

    const int is_look = !strcmp(key, "lookahead");
    const int is_depth = !strcmp(key, "iodepth");
    const int is_cache = !strcmp(key, "cache");
    /* topk may only be *lowered* from what the container declares: the
     * scratch the load allocated is sized from the manifest's top_k, and
     * asking for more experts than that would run the MoE off the end of
     * m->xacc and the per-expert LUTs. Truncation is the experiment;
     * widening is a different container. */
    const int is_topk = !strcmp(key, "topk");
    /* trunk=0,1,2,3 — which kernel the Q4G trunk matvec uses:
     * 0 f32 (the reference), 1 SDOT, 2 i8mm, 3 SMLAL. */
    const int is_sdot4 = !strcmp(key, "trunk");
    /* devkb=N sends every matvec of N KB or more to the accelerator;
     * -1 sends none. No effect in a build without one. */
    const int is_dev = !strcmp(key, "devkb");
    /* gpumoe=0,1 — the routed experts' VQ applies as device batches. */
    const int is_gmoe = !strcmp(key, "gpumoe");
    /* vq8=0,1 — the VQ3R apply through the int8 register table.
     * Needs WASTE_VQ8 set in the environment so the load allocates
     * the shadow table; the arm only chooses which kernel runs. */
    const int is_vq8 = !strcmp(key, "vq8");
    /* wide=0,1,2,3.. — which kernels are cut for the performance
     * cores instead of the whole pool. 1 VQ apply, 2 trunk matvec,
     * 4 LUT build, added together. */
    const int is_wide = !strcmp(key, "wide");
    if (!is_look && !is_depth && !is_cache && !is_topk && !is_sdot4 && !is_dev && !is_gmoe && !is_vq8 && !is_wide) {
        fprintf(stderr, "unknown key %s\n", key);
        return 2;
    }

    waste_model m;
    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    const char *cmb = getenv("WASTE_CACHE_MB");
    lo.cache_bytes = (size_t)(cmb ? atoi(cmb) : 0) << 20;
    lo.direct_io = 1;
    double t0 = now();
    if (waste_model_load(&m, argv[1], 4096, &lo)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const int top_k0 = m.cfg.top_k;
    /* One arm key is not always the whole experiment: WASTE_SWEEP_TOPK
     * pins top_k for every arm, so a kernel sweep can be run at the
     * operating point docs/LEARNED.md §56 recommends rather than at the
     * container's declared top_k. Lowering only, for §56's reason. */
    { const char *tk = getenv("WASTE_SWEEP_TOPK");
      if (tk) { int v = atoi(tk);
                if (v >= 1 && v <= top_k0) m.cfg.top_k = v; } }
    printf("pool: %d threads, %d on the fast cores\n",
           waste_pool_threads_public(), waste_model_fast_threads());
    printf("loaded in %.1fs — cache %d slots; %d arms x %d repeats, "
           "%d prompt + %d generated\n\n",
           now() - t0, m.cache.n_slots, n_arms, repeat, n, n_gen);

    /* Interleaved rather than grouped: if the machine drifts over the run —
     * and it does — a grouped sweep charges the drift to the last arm and
     * an interleaved one spreads it across all of them. */
    const int prof = getenv("WASTE_PROFILE") != NULL;
    /* Teacher-forced quality against the first arm.
     *
     * Two arms that generate greedily diverge as soon as one logit crosses
     * another, and after that they are answering different questions — so
     * comparing their outputs measures the divergence point and nothing
     * else. With WASTE_SWEEP_KL=1 the first arm of the first repeat
     * generates, and every arm after it is fed that same sequence and
     * scored against its logits position by position. KL is the screen;
     * docs/LEARNED.md §56 is the reminder that the continuation is the
     * gate. */
    const int kl_on = getenv("WASTE_SWEEP_KL") != NULL;
    int *ref_routes = NULL;
    const int kl_v = kl_on ? atoi(getenv("WASTE_SWEEP_KL")) : 0;
    float *ref_lg = NULL; int ref_ids[512]; int ref_n = 0;
    if (kl_on) {
        ref_lg = (float *)malloc((size_t)n_gen * m.cfg.vocab * sizeof(float));
        if (!ref_lg) { fprintf(stderr, "no room for the reference logits\n"); return 1; }
        /* One slot per (position, layer, selected expert), generously. */
        waste_route_cap_n = (n + n_gen + 4) * (m.cfg.n_layers + 1) * m.cfg.top_k;
        waste_route_cap = (int *)malloc((size_t)waste_route_cap_n * sizeof(int));
        ref_routes = (int *)malloc((size_t)waste_route_cap_n * sizeof(int));
        if (!waste_route_cap || !ref_routes) { fprintf(stderr, "no room for routes\n"); return 1; }
    }
    printf("%8s %6s %7s %9s %9s %9s%s\n", key, "rep", "slots", "tok/s", "hit",
           "GB read", prof ? "   tmv GB/s   kda s  luta s" : "");
    for (int r = 0; r < repeat; r++) {
        for (int a = 0; a < n_arms; a++) {
            if (is_topk) {
                if (arm[a] < 1 || arm[a] > top_k0) {
                    fprintf(stderr, "topk %d outside 1..%d (the container's)\n",
                            arm[a], top_k0);
                    return 2;
                }
                m.cfg.top_k = arm[a];
            } else if (is_wide) {
                waste_model_set_wide(arm[a]);
            } else if (is_vq8) {
                waste_model_set_vq8(arm[a]);
            } else if (is_gmoe) {
                waste_model_set_metal_moe(arm[a]);
            } else if (is_dev) {
                waste_model_set_device_min_kb(arm[a]);
            } else if (is_sdot4) {
                waste_model_set_kernel(&m, arm[a], sdot4_sg_env);
            } else if (is_look) {
                waste_model_set_lookahead(arm[a]);
            } else if (is_depth) {
                m.cache.depth = arm[a] < 1 ? 1 : arm[a];
            } else if (waste_model_resize_cache(&m, (size_t)arm[a] << 20)) {
                fprintf(stderr, "resize to %d MB failed\n", arm[a]);
                return 1;
            }

            waste_model_reset(&m);
            waste_ecache_clear(&m.cache);

            const float *lg = NULL;
            for (int i = 0; i < n; i++) lg = waste_model_step(&m, ids[i], i, NULL);
            if (!lg) { fprintf(stderr, "prompt failed\n"); return 1; }

            int cur = 0;
            for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            int gen[512]; int ngen = 0;
            const int is_ref = kl_on && r == 0 && a == 0;
            double kl_sum = 0, l2_sum = 0, kl_time = 0; int top10 = 0, argmax_same = 0;
            double pos_l2[512];
            for (int i = 0; i < 512; i++) pos_l2[i] = 0;
            waste_route_n = 0;
            waste_tcheck_num = waste_tcheck_den = waste_tcheck_max = 0;
            waste_tcheck_n = 0;
            memset(waste_prof, 0, sizeof waste_prof);
            memset(waste_prof_n, 0, sizeof waste_prof_n);
            waste_tmv_bytes = 0;
            const double s = now();
            for (int i = 0; i < n_gen; i++) {
                if (kl_on && !is_ref && i < ref_n) cur = ref_ids[i];
                lg = waste_model_step(&m, cur, n + i, NULL);
                if (!lg) { fprintf(stderr, "step failed\n"); return 1; }
                if (ngen < 512) gen[ngen++] = cur;
                if (is_ref) {
                    memcpy(ref_lg + (size_t)i * m.cfg.vocab, lg,
                           (size_t)m.cfg.vocab * sizeof(float));
                    if (ref_n < 512) ref_ids[ref_n++] = cur;
                } else if (kl_on) {
                    const double kt0 = now();
                    const float *R = ref_lg + (size_t)i * m.cfg.vocab;
                    double mr = R[0], mq = lg[0];
                    for (int v = 1; v < m.cfg.vocab; v++) {
                        if (R[v] > mr) mr = R[v];
                        if (lg[v] > mq) mq = lg[v];
                    }
                    double sr = 0, sq = 0, num = 0, den = 0;
                    for (int v = 0; v < m.cfg.vocab; v++) {
                        sr += exp(R[v] - mr); sq += exp(lg[v] - mq);
                        const double d = (double)lg[v] - R[v];
                        num += d * d; den += (double)R[v] * R[v];
                    }
                    double kl = 0;
                    for (int v = 0; v < m.cfg.vocab; v++) {
                        const double pp = exp(R[v] - mr) / sr;
                        if (pp > 1e-12) kl += pp * ((R[v] - mr - log(sr)) - (lg[v] - mq - log(sq)));
                    }
                    kl_sum += kl; l2_sum += sqrt(num / (den > 0 ? den : 1));
                    if (kl_v > 1) pos_l2[i] = sqrt(num / (den > 0 ? den : 1));
                    /* top-10 overlap, by selection sort on both */
                    int ti[10], qi[10];
                    for (int t = 0; t < 10; t++) {
                        int bi = -1; double bv = -1e30;
                        for (int v = 0; v < m.cfg.vocab; v++) {
                            int dup = 0;
                            for (int u = 0; u < t; u++) if (ti[u] == v) dup = 1;
                            if (!dup && R[v] > bv) { bv = R[v]; bi = v; }
                        }
                        ti[t] = bi;
                        bi = -1; bv = -1e30;
                        for (int v = 0; v < m.cfg.vocab; v++) {
                            int dup = 0;
                            for (int u = 0; u < t; u++) if (qi[u] == v) dup = 1;
                            if (!dup && lg[v] > bv) { bv = lg[v]; bi = v; }
                        }
                        qi[t] = bi;
                    }
                    for (int t = 0; t < 10; t++)
                        for (int u = 0; u < 10; u++) if (ti[t] == qi[u]) top10++;
                    if (ti[0] == qi[0]) argmax_same++;
                    kl_time += now() - kt0;
                }
                cur = 0;
                for (int v = 1; v < m.cfg.vocab; v++) if (lg[v] > lg[cur]) cur = v;
            }
            const double dt = now() - s - kl_time;
            const unsigned long long h = m.cache.hits, mi = m.cache.misses;
            printf("%8d %6d %7d %8.3f %8.1f%% %8.1f", arm[a], r + 1,
                   m.cache.n_slots, n_gen / dt,
                   100.0 * (double)h / (double)(h + mi ? h + mi : 1),
                   (double)m.cache.bytes_read / 1073741824.0);
            /* Two different questions, and only the second one matters.
             * The engine renormalizes the selected weights and sums the
             * experts, so *which* expert sits at rank 3 changes nothing —
             * two near-tied scores swapping is invisible in the output.
             * What is visible is an expert entering or leaving the set.
             * Comparing the ordered list conflates the two and reports a
             * harmless rank swap as two mismatches. */
            int route_rank = 0, route_set = 0, route_tot = waste_route_n;
            if (kl_on) {
                if (is_ref) memcpy(ref_routes, waste_route_cap,
                                   (size_t)waste_route_n * sizeof(int));
                else {
                    const int K = m.cfg.top_k;
                    for (int i = 0; i < waste_route_n; i++)
                        if (waste_route_cap[i] == ref_routes[i]) route_rank++;
                    for (int g0 = 0; g0 + K <= waste_route_n; g0 += K)
                        for (int u = 0; u < K; u++)
                            for (int v = 0; v < K; v++)
                                if (ref_routes[g0 + u] == waste_route_cap[g0 + v])
                                    { route_set++; break; }
                }
            }
            if (kl_v > 1 && !is_ref) {
                printf("      relL2 by position:");
                for (int i = 0; i < n_gen && i < 24; i++) printf(" %.1e", pos_l2[i]);
                printf("\n");
            }
            if (kl_on && !is_ref)
                printf("   KL %.3e relL2 %.2e top10 %.1f%% argmax %d/%d"
                       " route set %.2f%% rank %.2f%%",
                       kl_sum / n_gen, l2_sum / n_gen, top10 * 10.0 / n_gen,
                       argmax_same, n_gen,
                       route_tot ? 100.0 * route_set / route_tot : 0.0,
                       route_tot ? 100.0 * route_rank / route_tot : 0.0);
            if (waste_tcheck_n)
                printf("   [vs f32: mean relL2 %.3e over %llu matvecs, max|d| %.3g]",
                       waste_tcheck_num / waste_tcheck_den,
                       (unsigned long long)waste_tcheck_n, waste_tcheck_max);
            if (prof)
                printf("   %9.1f %7.2f %8.2f",
                       waste_prof[9] > 0 ? waste_tmv_bytes / waste_prof[9] / 1e9 : 0.0,
                       waste_prof[1], waste_prof[7]);
#if defined(WASTE_ENABLE_METAL)
            if (getenv("WASTE_METAL_STATS"))
                printf("   [metal %llu disp %.2fs gpu %.2fs copy %.2fs wrap %.1f GB/s"
                       " | %llu on cpu]",
                       waste_metal_n_gpu, waste_metal_t_gpu, waste_metal_t_copy,
                       waste_metal_t_wrap,
                       waste_metal_t_gpu > 0 ? waste_metal_bytes/waste_metal_t_gpu/1e9 : 0.0,
                       waste_metal_n_fallback);
            waste_metal_t_gpu = waste_metal_t_copy = waste_metal_t_wrap = 0;
            waste_metal_n_gpu = waste_metal_n_fallback = waste_metal_bytes = 0;
#endif
            printf("\n");
            if (getenv("WASTE_SWEEP_IDS")) {
                printf("      ids:");
                for (int i = 0; i < ngen; i++) printf(" %d", gen[i]);
                printf("\n");
            }
            fflush(stdout);
        }
    }
    waste_model_free(&m);
    return 0;
}
