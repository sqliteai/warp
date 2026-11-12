/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* test_kernel_isolation — the trunk kernel belongs to a model, not to the
 * process (sqliteai/warp#68).
 *
 * Opening a Qwen container used to write a file-static default that every
 * model already open then read from, so a non-Qwen context mid-session
 * switched from the f32 trunk to i8mm and its logits moved underneath it.
 * The fixtures are tiny, so the drift is small; on a real checkpoint it is
 * an unannounced change of arithmetic in a live session.
 *
 * The isolation must not be bought by denying Qwen its kernel: the test
 * checks that the Qwen model picked the same kernel kern_clamp() would
 * give it on this CPU. On a CPU without i8mm (all non-arm64 CI runners,
 * and any arm64 without SMMLA), that is not i8mm — there is nothing to
 * isolate from, and the repo rule is that a missing prerequisite is never
 * a pass. WASTE_TRUNK_KERNEL, if set, changes what "would have selected
 * i8mm" means, so it also routes to SKIP rather than a false PASS/FAIL.
 *
 *   usage: test_kernel_isolation <non-qwen.waste> <qwen.waste>
 *   exit 0 pass, 1 harness failure, 2 skip (no i8mm CPU or kernel pinned), 3 leak observed
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"
#include "../src/waste_backend.h"

static const int PROMPT[] = { 3, 17, 42, 8, 99, 5, 61, 23, 77, 12 };
enum { NTOK = (int)(sizeof PROMPT / sizeof PROMPT[0]) };

/* Reset, replay the same tokens, return a private copy of the final logits. */
static float *replay(waste_model *m, int *vocab)
{
    waste_model_reset(m);
    const float *lg = NULL;
    for (int i = 0; i < NTOK; i++) lg = waste_model_step(m, PROMPT[i], i, NULL);
    if (!lg) return NULL;
    *vocab = m->cfg.vocab;
    float *copy = malloc((size_t)*vocab * sizeof *copy);
    if (copy) memcpy(copy, lg, (size_t)*vocab * sizeof *copy);
    return copy;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <plain.waste> <qwen.waste>\n", argv[0]); return 1; }

    /* Whether a fresh Qwen load on this CPU, with the env unset, would
     * select i8mm at all. If not, there is nothing here to isolate: the
     * check would either report a false FAIL (isolation looks broken
     * because both kernels are already the same one) or, worse, a false
     * PASS bought by comparing a kernel against itself. Same when
     * WASTE_TRUNK_KERNEL is set, since Qwen then doesn't pick i8mm by
     * load-time detection at all. */
    if (getenv("WASTE_TRUNK_KERNEL")) {
        printf("SKIP WASTE_TRUNK_KERNEL is set; Qwen load-time kernel selection is overridden\n");
        return 2;
    }
    const uint32_t cpu = waste_cpu_features();
    if (!(cpu & WASTE_CPU_DOTPROD) || !(cpu & WASTE_CPU_I8MM)) {
        printf("SKIP this CPU has no i8mm (features=0x%x); a Qwen load would not select TK_I8MM here\n", cpu);
        return 2;
    }

    waste_load_opts lo; memset(&lo, 0, sizeof lo); lo.direct_io = 1;
    {
        const char *cmb = getenv("WASTE_CACHE_MB");
        lo.cache_bytes = (size_t)(cmb ? atoi(cmb) : 0) << 20;
    }
    waste_model plain, qwen;
    int rc = 1;
    float *before = NULL, *after = NULL;

    if (waste_model_load(&plain, argv[1], 4096, &lo)) {
        fprintf(stderr, "load %s failed\n", argv[1]); return 1;
    }
    if (plain.cfg.arch_qwen) {
        printf("SKIP fixture %s is a Qwen container; test needs a non-Qwen one\n", argv[1]);
        waste_model_free(&plain); return 2;
    }

    int v1 = 0, v2 = 0;
    const int kern_before = plain.trunk_kern;
    if (!(before = replay(&plain, &v1))) { fprintf(stderr, "first replay failed\n"); goto out1; }

    /* The event under test. */
    if (waste_model_load(&qwen, argv[2], 4096, &lo)) {
        fprintf(stderr, "load %s failed\n", argv[2]); goto out1;
    }
    if (!qwen.cfg.arch_qwen) {
        fprintf(stderr, "fixture %s is not a Qwen container\n", argv[2]); goto out2;
    }

    if (!(after = replay(&plain, &v2))) { fprintf(stderr, "second replay failed\n"); goto out2; }

    if (v1 != v2) { printf("FAIL vocab moved %d -> %d\n", v1, v2); rc = 3; goto out2; }

    if (plain.trunk_kern != kern_before) {
        printf("FAIL trunk_kern of the open model moved %d -> %d when %s was loaded\n",
               kern_before, plain.trunk_kern, argv[2]);
        rc = 3; goto out2;
    }
    if (memcmp(before, after, (size_t)v1 * sizeof *before) != 0) {
        int n = 0;
        for (int i = 0; i < v1; i++) if (memcmp(&before[i], &after[i], sizeof *before)) n++;
        printf("FAIL %d/%d logits changed after loading %s\n", n, v1, argv[2]);
        rc = 3; goto out2;
    }

    /* The isolation must not have been bought by denying Qwen its kernel. */
    if (qwen.trunk_kern != TK_I8MM) {
        printf("FAIL Qwen model did not select i8mm (trunk_kern=%d)\n", qwen.trunk_kern);
        rc = 3; goto out2;
    }

    printf("ok plain.trunk_kern=%d held, qwen.trunk_kern=%d, %d logits bit-identical\n",
           plain.trunk_kern, qwen.trunk_kern, v1);
    rc = 0;

out2:
    waste_model_free(&qwen);
out1:
    waste_model_free(&plain);
    free(before); free(after);
    return rc;
}
