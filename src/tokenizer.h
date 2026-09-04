/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * tokenizer.h — tiktoken-style BPE for the Kimi family, in C.
 *
 * Loads the `tiktoken.model` shipped with the model (base64 token + rank
 * per line) and reproduces the encoder: split the text with the model's
 * pre-tokenization pattern, then byte-pair-merge each piece by rank.
 *
 * The pattern uses Unicode classes (\p{L}, \p{N}, \p{Han}). Rather than
 * pull in a regex engine, the classes are implemented directly over
 * decoded codepoints — see tokenizer.c for exactly which ranges, and
 * tests/test_tokenizer for the agreement check against Python tiktoken.
 */

#ifndef WASTE_TOKENIZER_H
#define WASTE_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct waste_tok waste_tok;

/* Loads <dir>/tokenizer.model (or tiktoken.model). NULL on failure. */
waste_tok *waste_tok_open(const char *dir);
void       waste_tok_free(waste_tok *t);

int  waste_tok_vocab(const waste_tok *t);
/* Special token ids, or -1 if the model does not define them. */
int  waste_tok_bos(const waste_tok *t);
int  waste_tok_eos(const waste_tok *t);
/* Override the positional guess with the container's own eos_token_id.
 * Ignored when id <= 0, so a container that does not state one keeps the
 * reserved-block default. */
void waste_tok_set_eos(waste_tok *t, int id);
/* 1 (the default) = the model's pattern gives Han its own [\p{Han}]+
 * branch, as both Kimi releases do; 0 = it does not, and Han is an
 * ordinary \p{L} — GLM-5.3-Flash's pattern. The two differ only where a
 * Han run touches a Latin one, and they differ silently. */
void waste_tok_set_han_split(waste_tok *t, int on);

/* Digits per pre-token: 3 (the default) for `\p{N}{1,3}`, as the Kimi and
 * GLM patterns spell it; 1 for `\p{N}`, which is Qwen's and puts every
 * digit in its own piece. Anything else is ignored and the default kept —
 * there is no third spelling in this family. "2026" is one token under 3
 * and four under 1, so a container that gets this wrong reads every
 * number differently from the model that was trained on it. */
void waste_tok_set_digit_run(waste_tok *t, int n);

/* Encodes `text` into `out` (capacity `cap`); returns the count, or -1 if
 * it would not fit. Special tokens in the text are NOT interpreted. */
/* allow_special: 1 = `<|open|>` and friends become their control-token
 * ids (what a chat template is made of); 0 = they are ordinary text, and
 * the model cannot be handed structure by whoever wrote the string. The
 * reference tokenizer draws the same line, and draws it for the same
 * reason: `tokenization_kimi.py` encodes markup with allowed_special="all"
 * and user or tool content with disallowed_special=(). */
int  waste_tok_encode(const waste_tok *t, const char *text, int32_t *out,
                      int cap, int allow_special);

/* Decodes one token into `buf`; returns bytes written (no NUL). */
int  waste_tok_decode_len1(const waste_tok *t, int32_t id);
int  waste_tok_decode1(const waste_tok *t, int32_t id, char *buf, int cap);
int  waste_tok_decode(const waste_tok *t, const int32_t *ids, int n,
                      char *buf, int cap);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_TOKENIZER_H */
