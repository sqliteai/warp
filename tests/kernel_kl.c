/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * kernel_kl.c — trunk kernels side by side, scored at every position.
 *
 *   kernel_kl CONTAINER IDS_FILE n_gen [kernel_a=0] [kernels_b=2] [window=512]
 *
 * Kernels are WASTE_TRUNK_KERNEL's numbers: 0 f32, 1 SDOT, 2 i8mm, 3 SMLAL.
 * kernels_b is a comma list, each scored against kernel_a.
 *
 * sweep.c scores a kernel against a stored reference, which keeps one
 * vocab-sized logit vector per position and so stops at 512 of them. The
 * question that limit cannot answer is the one that killed SDOT on K3:
 * whether a small per-matvec error grows as a recurrence carries it, and
 * whether a sparse-attention selection that only starts choosing past a
 * few thousand tokens starts choosing differently. So here the container is
 * loaded once per kernel and every copy steps through the same tokens, with
 * the trunk kernel switched between them, each position scored as it goes:
 * KL(a||b), argmax agreement, top-10 overlap, the logits' relative L2 — and,
 * because a top-K router turns a small arithmetic difference into a
 * discrete one, how many of each layer's routed experts the two agreed on.
 *
 * The prompt is teacher-forced by construction. After it, every copy is fed
 * kernel a's greedy tokens, so generation is scored the same way rather than
 * by where two free-running continuations happen to part.
 *
 * Run it with kernel_b equal to kernel_a first: every column must come out
 * exactly zero, or the copies are sharing state and nothing else it prints
 * means anything.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/model.h"

#define MAX_B 4

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

typedef struct {
    int n, argmax_same, top10, kl_max_pos, n_nll;
    double kl, kl_max, rel;
    /* Next-token NLL of the real text under each kernel. KL says how far
     * two distributions are apart; this says whether the one further from
     * the reference is any worse at the text, which is the trade actually
     * being decided. Prompt positions only — past the prompt, the "real"
     * next token is kernel a's own choice. */
    double nll_a, nll_b;
    long long exp_same, exp_tot, layers_diff, layers_tot;
} window;

static void top10(const float *x, int V, int *idx)
{
    float val[10];
    int k = 0;
    for (int v = 0; v < V; v++) {
        if (k == 10 && x[v] <= val[9]) continue;
        int j = k < 10 ? k++ : 9;
        while (j > 0 && val[j - 1] < x[v]) {
            val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--;
        }
        val[j] = x[v]; idx[j] = v;
    }
}

/* a's argmax, for the next generated token. */
static int argmax(const float *a, int V)
{
    int best = 0;
    for (int v = 1; v < V; v++) if (a[v] > a[best]) best = v;
    return best;
}

static void score(const float *a, const float *b, int V, int pos, int target,
                  window *w)
{
    double ma = a[0], mb = b[0];
    int aa = 0, ab = 0;
    for (int v = 1; v < V; v++) {
        if (a[v] > ma) { ma = a[v]; aa = v; }
        if (b[v] > mb) { mb = b[v]; ab = v; }
    }
    double sa = 0, sb = 0, num = 0, den = 0;
    for (int v = 0; v < V; v++) {
        sa += exp(a[v] - ma);
        sb += exp(b[v] - mb);
        const double d = (double)b[v] - a[v];
        num += d * d;
        den += (double)a[v] * a[v];
    }
    const double la = ma + log(sa), lb = mb + log(sb);
    double kl = 0;
    for (int v = 0; v < V; v++) {
        const double lpa = a[v] - la, pa = exp(lpa);
        if (pa > 1e-12) kl += pa * (lpa - (b[v] - lb));
    }
    if (kl < 0) kl = 0;     /* a sum of ~1e-12 terms can round below zero */
    if (target >= 0 && target < V) {
        w->nll_a += la - a[target];
        w->nll_b += lb - b[target];
        w->n_nll++;
    }

    int ta[10], tb[10], same = 0;
    top10(a, V, ta);
    top10(b, V, tb);
    for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++)
            if (ta[i] == tb[j]) { same++; break; }

    w->n++;
    w->kl += kl;
    if (kl > w->kl_max) { w->kl_max = kl; w->kl_max_pos = pos; }
    w->rel += sqrt(num / (den > 0 ? den : 1));
    w->argmax_same += aa == ab;
    w->top10 += same;
}

static void routes(const int *ra, const int *rb, int L, int K, window *w)
{
    for (int l = 0; l < L; l++) {
        const int *pa = ra + (size_t)l * K, *pb = rb + (size_t)l * K;
        int same = 0;
        for (int u = 0; u < K; u++)
            for (int v = 0; v < K; v++)
                if (pa[u] == pb[v]) { same++; break; }
        w->exp_same += same;
        w->exp_tot += K;
        w->layers_tot++;
        if (same < K) w->layers_diff++;
    }
}

static void add(window *dst, const window *src)
{
    if (src->kl_max > dst->kl_max) { dst->kl_max = src->kl_max; dst->kl_max_pos = src->kl_max_pos; }
    dst->n += src->n; dst->kl += src->kl; dst->rel += src->rel;
    dst->n_nll += src->n_nll; dst->nll_a += src->nll_a; dst->nll_b += src->nll_b;
    dst->argmax_same += src->argmax_same; dst->top10 += src->top10;
    dst->exp_same += src->exp_same; dst->exp_tot += src->exp_tot;
    dst->layers_diff += src->layers_diff; dst->layers_tot += src->layers_tot;
}

static void report(const char *label, int kb, const window *w)
{
    if (!w->n) return;
    printf("%-20s k%d  KL mean %.2e max %.2e @%-5d | argmax %5d/%-5d | top10 %5.1f%% | "
           "relL2 %.2e | experts %6.2f%%, layers differing %lld/%lld",
           label, kb, w->kl / w->n, w->kl_max, w->kl_max_pos, w->argmax_same, w->n,
           10.0 * w->top10 / w->n, w->rel / w->n,
           w->exp_tot ? 100.0 * w->exp_same / w->exp_tot : 0.0,
           w->layers_diff, w->layers_tot);
    if (w->n_nll) {
        const double pa = exp(w->nll_a / w->n_nll), pb = exp(w->nll_b / w->n_nll);
        printf(" | ppl %.3f -> %.3f (%+.2f%%)", pa, pb, 100.0 * (pb / pa - 1.0));
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s CONTAINER IDS_FILE n_gen [kernel_a=0] "
                "[kernels_b=2] [window=512]\n", argv[0]);
        return 2;
    }
    const int n_gen = atoi(argv[3]);
    const int ka = argc > 4 ? atoi(argv[4]) : 0;
    int kb[MAX_B], nb = 0;
    {
        char list[64];
        snprintf(list, sizeof list, "%s", argc > 5 ? argv[5] : "2");
        for (char *p = strtok(list, ","); p && nb < MAX_B; p = strtok(NULL, ","))
            kb[nb++] = atoi(p);
    }
    const int win = argc > 6 && atoi(argv[6]) > 0 ? atoi(argv[6]) : 512;
    const char *sgs = getenv("WASTE_SDOT4_SG");
    const int sg = sgs ? atoi(sgs) : 32;

    FILE *f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    enum { MAXIDS = 32768 };
    static int ids[MAXIDS];
    int n = 0;
    for (int c, v = 0, in = 0; ; ) {
        c = fgetc(f);
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); in = 1; continue; }
        if (in && n < MAXIDS) ids[n++] = v;
        v = 0; in = 0;
        if (c == EOF) break;
    }
    fclose(f);
    if (n < 1 || n_gen < 0 || nb < 1) { fprintf(stderr, "nothing to run\n"); return 1; }

    waste_load_opts lo;
    memset(&lo, 0, sizeof lo);
    const char *cmb = getenv("WASTE_CACHE_MB");
    lo.cache_bytes = (size_t)(cmb ? atoi(cmb) : 0) << 20;
    lo.direct_io = 1;
    /* The same floor test_forward and sweep load with: a container may
     * refuse a context shorter than its own attention geometry needs. */
    int kv = n + n_gen + 16;
    if (kv < 4096) kv = 4096;
    waste_model *ma = (waste_model *)calloc(1, sizeof *ma);
    waste_model *mb[MAX_B];
    double t0 = now();
    if (!ma || waste_model_load(ma, argv[1], kv, &lo)) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    waste_model_set_kernel(ma, ka, sg);
    for (int i = 0; i < nb; i++) {
        mb[i] = (waste_model *)calloc(1, sizeof *mb[i]);
        if (!mb[i] || waste_model_load(mb[i], argv[1], kv, &lo)) {
            fprintf(stderr, "load failed\n");
            return 1;
        }
        waste_model_set_kernel(mb[i], kb[i], sg);
    }
    const int V = ma->cfg.vocab, L = ma->cfg.n_layers, K = ma->cfg.top_k;
    float *A = (float *)malloc((size_t)V * sizeof(float));
    int *ra = (int *)malloc((size_t)L * K * sizeof(int));
    int *rb = (int *)malloc((size_t)L * K * sizeof(int));
    if (!A || !ra || !rb) { fprintf(stderr, "out of memory\n"); return 1; }
    printf("kernel a %d vs", ka);
    for (int i = 0; i < nb; i++) printf(" %d", kb[i]);
    printf(": %d prompt + %d generated, windows of %d; %d loads in %.1fs\n\n",
           n, n_gen, win, nb + 1, now() - t0);
    fflush(stdout);

    window w[MAX_B], prompt[MAX_B], gen[MAX_B];
    memset(w, 0, sizeof w); memset(prompt, 0, sizeof prompt); memset(gen, 0, sizeof gen);
    int cur = 0, w0 = 0;
    t0 = now();
    for (int pos = 0; pos < n + n_gen; pos++) {
        const int tok = pos < n ? ids[pos] : cur;
        const float *la = waste_model_step(ma, tok, pos, ra);
        if (!la) { fprintf(stderr, "kernel a step %d failed\n", pos); return 1; }
        memcpy(A, la, (size_t)V * sizeof(float));
        cur = argmax(A, V);
        for (int i = 0; i < nb; i++) {
            const float *lb = waste_model_step(mb[i], tok, pos, rb);
            if (!lb) { fprintf(stderr, "kernel %d step %d failed\n", kb[i], pos); return 1; }
            score(A, lb, V, pos, pos + 1 < n ? ids[pos + 1] : -1, &w[i]);
            routes(ra, rb, L, K, &w[i]);
        }

        const int end = pos + 1;
        if (end % win == 0 || end == n || end == n + n_gen) {
            char label[64];
            snprintf(label, sizeof label, "%s %5d-%-5d", pos < n ? "prompt" : "gen   ",
                     w0, pos);
            for (int i = 0; i < nb; i++) {
                report(label, kb[i], &w[i]);
                add(pos < n ? &prompt[i] : &gen[i], &w[i]);
                memset(&w[i], 0, sizeof w[i]);
            }
            fflush(stdout);
            w0 = end;
            fprintf(stderr, "  %d/%d positions, %.0fs\n", end, n + n_gen, now() - t0);
        }
    }
    printf("\n");
    for (int i = 0; i < nb; i++) {
        report("prompt, all", kb[i], &prompt[i]);
        report("generated, all", kb[i], &gen[i]);
        add(&prompt[i], &gen[i]);
        report("everything", kb[i], &prompt[i]);
    }
    waste_model_free(ma);
    for (int i = 0; i < nb; i++) waste_model_free(mb[i]);
    return 0;
}
