/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* test_tokenizer.c — encode/decode round-trip and id dump for the Python diff.
 *
 * Encodes in markup mode, matching tools/tokdiff.py's tiktoken Encoding,
 * which is built with the reserved block registered as special tokens.
 * WASTE_TOK_PLAIN=1 flips it to the mode a user prompt goes through, where
 * `<|open|>` is ordinary text — that difference is the check that a
 * pasted marker cannot forge conversation structure. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/tokenizer.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s container [text...]\n", argv[0]); return 2; }
    waste_tok *t = waste_tok_open(argv[1]);
    if (!t) { fprintf(stderr, "no tokenizer in %s\n", argv[1]); return 1; }
    if (argc == 2) {
        printf("vocab %d  bos %d  eos %d\n", waste_tok_vocab(t),
               waste_tok_bos(t), waste_tok_eos(t));
        return 0;
    }
    const char *plain = getenv("WASTE_TOK_PLAIN");
    const int markup = !(plain && *plain != '0');
    /* The engine takes this from the container's config; waste_tok_open
     * reads a directory, not a manifest, so the harness needs its own way
     * to reach the GLM pattern. See waste_tok_set_han_split. */
    const char *nohan = getenv("WASTE_TOK_NOHAN");
    if (nohan && *nohan != '0') waste_tok_set_han_split(t, 0);
    /* Same for the digit run: 1 is Qwen's `\p{N}`, 3 the default
     * `\p{N}{1,3}`. See waste_tok_set_digit_run. */
    const char *drun = getenv("WASTE_TOK_DIGITS");
    if (drun) waste_tok_set_digit_run(t, atoi(drun));
    for (int a = 2; a < argc; a++) {
        int32_t ids[4096];
        const int n = waste_tok_encode(t, argv[a], ids, 4096, markup);
        printf("%d", n);
        for (int i = 0; i < n; i++) printf(" %d", ids[i]);
        printf("\n");
    }
    waste_tok_free(t);
    return 0;
}
