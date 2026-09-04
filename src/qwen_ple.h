/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * qwen_ple.h — Qwen n-gram hashing for Per-Layer Embedding.
 *
 * Equations are from transformers Qwen4ExpTextNGramEmbedding: splitmix
 * multipliers, EOS-bounded shifts, XOR mix, then remainder into 16 head
 * tables. Each head lives on disk as Q8G; this module never asks for the
 * BF16 source table.
 */

#ifndef WASTE_QWEN_PLE_H
#define WASTE_QWEN_PLE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WASTE_QWEN_PLE_HEADS 16

uint64_t waste_qwen_splitmix64(uint64_t value);

/* Official _build_layer_multipliers. out[ngram_size] signed odd integers. */
void waste_qwen_ple_multipliers(int64_t *out, int ngram_size, int ple_layer_index,
                                int64_t seed, int unigram_vocab);

int waste_qwen_ple_is_prime(int64_t value);
int64_t waste_qwen_ple_nth_prime_after(int64_t start, int count);

/* Shift the sequence right by `shift`, filling from EOS and never crossing
 * a previous EOS. Official _shift_right_ignore_eos. */
void waste_qwen_ple_shift_eos(const int *ids, int n, int shift, int eos, int *out);

/* 16 local row ids for the token at `pos` (0-based in `ids[0..n)`).
 * sizes[h] is that head's prime vocab; local row is remainder, not the
 * global offset used by a packed table. */
void waste_qwen_ple_row_ids(const int *ids, int n, int pos, int eos,
                            int ngram_size, int heads_per_ngram,
                            const int64_t *multipliers,
                            const int64_t *head_vocab_sizes,
                            int *local_rows);

#ifdef __cplusplus
}
#endif
#endif /* WASTE_QWEN_PLE_H */
