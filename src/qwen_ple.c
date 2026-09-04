/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* qwen_ple.c — see qwen_ple.h. */

#include "qwen_ple.h"

#include <stddef.h>

#define MASK64 ((uint64_t)-1)
#define SPLITMIX_GAMMA 0x9E3779B97F4A7C15ULL
#define SPLITMIX_M1    0xBF58476D1CE4E5B9ULL
#define SPLITMIX_M2    0x94D049BB133111EBULL
#define PRIME_1        10007

uint64_t waste_qwen_splitmix64(uint64_t value)
{
    value = (value + SPLITMIX_GAMMA) & MASK64;
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64;
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64;
    return (value ^ (value >> 31)) & MASK64;
}

void waste_qwen_ple_multipliers(int64_t *out, int ngram_size, int ple_layer_index,
                                int64_t seed, int unigram_vocab)
{
    const int64_t max_long = (int64_t)(((uint64_t)1 << 63) - 1);
    const int64_t multiplier_max = max_long / (unigram_vocab > 0 ? unigram_vocab : 1);
    int64_t half_bound = multiplier_max / 2;
    if (half_bound < 1) half_bound = 1;
    const uint64_t base_seed = (uint64_t)seed + (uint64_t)PRIME_1 * (uint64_t)ple_layer_index;
    for (int i = 0; i < ngram_size; i++) {
        const uint64_t value = (base_seed + SPLITMIX_GAMMA * (uint64_t)(i + 1)) & MASK64;
        out[i] = 2 * (int64_t)(waste_qwen_splitmix64(value) % (uint64_t)half_bound) + 1;
    }
}

int waste_qwen_ple_is_prime(int64_t value)
{
    if (value < 2) return 0;
    if (value % 2 == 0) return value == 2;
    for (int64_t d = 3; d * d <= value; d += 2)
        if (value % d == 0) return 0;
    return 1;
}

int64_t waste_qwen_ple_nth_prime_after(int64_t start, int count)
{
    int64_t prime = start;
    for (int i = 0; i < count; i++) {
        prime++;
        while (!waste_qwen_ple_is_prime(prime)) prime++;
    }
    return prime;
}

void waste_qwen_ple_shift_eos(const int *ids, int n, int shift, int eos, int *out)
{
    if (shift == 0) {
        for (int i = 0; i < n; i++) out[i] = ids[i];
        return;
    }
    int prev_eos = -1;
    int incl = -1;
    for (int i = 0; i < n; i++) {
        const int segment_start = prev_eos + 1;
        const int pos_in_seg = i - segment_start;
        const int src = i - shift;
        const int valid = (pos_in_seg >= shift) && (src >= 0);
        out[i] = valid ? ids[src] : eos;
        if (ids[i] == eos) incl = i;
        prev_eos = incl;
    }
}

void waste_qwen_ple_row_ids(const int *ids, int n, int pos, int eos,
                            int ngram_size, int heads_per_ngram,
                            const int64_t *multipliers,
                            const int64_t *head_vocab_sizes,
                            int *local_rows)
{
    const int heads = (ngram_size - 1) * heads_per_ngram;
    int h0;
    for (h0 = 0; h0 < heads && h0 < WASTE_QWEN_PLE_HEADS; h0++)
        local_rows[h0] = 0;
    if (!ids || !multipliers || !head_vocab_sizes || !local_rows) return;
    if (ngram_size < 1 || ngram_size > 8 || n < 1 || n > 8 ||
        pos < 0 || pos >= n || heads_per_ngram < 1)
        return;
    int hist[8 * 8];
    for (int s = 0; s < ngram_size; s++)
        waste_qwen_ple_shift_eos(ids, n, s, eos, hist + (size_t)s * n);

    int h = 0;
    for (int ngram = 2; ngram <= ngram_size; ngram++) {
        int64_t mixed = (int64_t)hist[0 * n + pos] * multipliers[0];
        for (int p = 1; p < ngram; p++)
            mixed ^= (int64_t)hist[p * n + pos] * multipliers[p];
        for (int k = 0; k < heads_per_ngram && h < WASTE_QWEN_PLE_HEADS; k++, h++) {
            const int64_t sz = head_vocab_sizes[h];
            if (sz <= 0) { local_rows[h] = 0; continue; }
            int64_t r = mixed % sz;
            if (r < 0) r += sz;
            local_rows[h] = (int)r;
        }
    }
}
