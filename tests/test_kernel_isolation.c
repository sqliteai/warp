/* test_kernel_isolation — the trunk kernel belongs to a model, not to the
 * process (sqliteai/warp#68).
 *
 * Opening a Qwen container used to write a file-static default that every
 * model already open then read from, so a non-Qwen context mid-session
 * switched from the f32 trunk to i8mm and its logits moved underneath it.
 * The fixtures are tiny, so the drift is small; on a real checkpoint it is
 * an unannounced change of arithmetic in a live session.
 *
 *   usage: test_kernel_isolation <non-qwen.waste> <qwen.waste>
 *   exit 0 pass, 1 harness failure, 3 leak observed
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/model.h"

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

    waste_load_opts lo; memset(&lo, 0, sizeof lo); lo.direct_io = 1;
    waste_model plain, qwen;
    int rc = 1;
    float *before = NULL, *after = NULL;

    if (waste_model_load(&plain, argv[1], 4096, &lo)) {
        fprintf(stderr, "load %s failed\n", argv[1]); return 1;
    }
    if (plain.cfg.arch_qwen) {
        fprintf(stderr, "fixture %s is a Qwen container; test needs a non-Qwen one\n", argv[1]);
        waste_model_free(&plain); return 1;
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
    if (qwen.trunk_kern != 2 /* TK_I8MM */) {
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
