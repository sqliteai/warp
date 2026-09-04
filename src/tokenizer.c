/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/* tokenizer.c — see tokenizer.h. */

#include "tokenizer.h"

#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *p; int len; int rank; } tok_entry;

/* Markup tokens — <|open|>, <|media_begin|>, [EOS] — are not merges and are
 * absent from tiktoken's rank file; they come from specials.json next to
 * it. They have to be matched before the BPE ever sees the text, or the
 * engine splits <|open|> into six ordinary tokens and a chat template
 * turns into gibberish the model was never trained on. */
typedef struct { char *text; int len; int32_t id; } tok_special;

struct waste_tok {
    uint8_t *blob;          /* decoded token bytes, back to back           */
    tok_entry *by_rank;     /* insertion order -> bytes                     */
    int n_tokens, cap_tokens;
    int32_t *hash;          /* open addressing: -> index into by_rank       */
    int hash_mask;
    /* rank -> index into by_rank, for decode. Ranks in a tiktoken file are
     * dense from 0, so a flat array is smaller than a hash and answers in
     * one load. Decoding used to scan all 163840 entries per token, twice
     * over (specials, then ranks), for every token generated. */
    int32_t *by_id;
    int32_t max_id;
    int bos, eos;
    tok_special *special;   /* longest-first, so <|a|> cannot mask <|ab|>  */
    int n_special;
    /* 1 = the pattern has a [\p{Han}]+ branch of its own (the Kimi
     * models), 0 = it does not and Han is matched by \p{L}+ like any other
     * letter (GLM). The difference is only visible on a Han run that runs
     * straight into a Latin one — "中文abc" is two pieces with the branch
     * and one without — which is exactly the kind of text that appears in
     * a Chinese release's own prompts. */
    int han_split;
    /* How many digits one pre-token may take: `\p{N}{1,3}` on the Kimi and
     * GLM patterns, `\p{N}` on Qwen's, which splits every digit into its
     * own piece. Not cosmetic — "2026" is one token under a 3-digit run
     * and four under a 1-digit one, and the difference shows up nowhere as
     * an error, only as a model reading numbers it was never trained on. */
    int digit_run;
};

/* ---- base64 ------------------------------------------------------------ */

static int b64val(int c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static int b64decode(const char *s, int len, uint8_t *out)
{
    /* unsigned: the accumulator shifts past bit 31 on long lines, and signed
     * overflow there is UB the optimizer is entitled to turn into a trap */
    int n = 0, bits = 0;
    uint32_t acc = 0;
    for (int i = 0; i < len; i++) {
        if (s[i] == '=') break;
        const int v = b64val((unsigned char)s[i]);
        if (v < 0) continue;
        acc = (acc << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out[n++] = (uint8_t)((acc >> bits) & 0xff); }
    }
    return n;
}

/* ---- hash over token bytes --------------------------------------------- */

static uint32_t bhash(const uint8_t *p, int len)
{
    uint32_t h = 2166136261u;
    for (int i = 0; i < len; i++) { h ^= p[i]; h *= 16777619u; }
    return h;
}

static void tok_index(waste_tok *t, int i)
{
    uint32_t h = bhash(t->by_rank[i].p, t->by_rank[i].len) & (uint32_t)t->hash_mask;
    while (t->hash[h] >= 0) h = (h + 1) & (uint32_t)t->hash_mask;
    t->hash[h] = i;
}

/* rank of a byte string, or -1 */
static int tok_rank(const waste_tok *t, const uint8_t *p, int len)
{
    uint32_t h = bhash(p, len) & (uint32_t)t->hash_mask;
    for (int probe = 0; probe <= t->hash_mask; probe++) {
        const int32_t i = t->hash[h];
        if (i < 0) return -1;
        if (t->by_rank[i].len == len && memcmp(t->by_rank[i].p, p, (size_t)len) == 0)
            return t->by_rank[i].rank;
        h = (h + 1) & (uint32_t)t->hash_mask;
    }
    return -1;
}

/* ---- special tokens ------------------------------------------------------ */

/* specials.json: [{"id": 163587, "text": "<|open|>"}, ...]. Hand-parsed —
 * the file is written by our own converter and has no nesting — and sorted
 * longest-first so a prefix can never shadow a longer token. */
static int spec_cmp(const void *a, const void *b)
{
    return ((const tok_special *)b)->len - ((const tok_special *)a)->len;
}

static void load_specials(waste_tok *t, const char *dir)
{
    char path[512];
    snprintf(path, sizeof path, "%s/specials.json", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return; }
    const long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return; }
    buf[n] = 0;
    fclose(f);

    int cap = 16;
    t->special = (tok_special *)malloc((size_t)cap * sizeof *t->special);
    if (!t->special) { free(buf); return; }

    const char *p = buf;
    while ((p = strstr(p, "\"id\"")) != NULL) {
        const char *tk = strstr(p, "\"text\"");
        if (!tk) break;
        const char *colon = strchr(p, ':');
        if (!colon || colon > tk) { p += 4; continue; }
        char *id_end = NULL;
        errno = 0;
        const long raw_id = strtol(colon + 1, &id_end, 10);
        if (errno || id_end == colon + 1 || raw_id < 0 || raw_id > INT32_MAX) {
            p = tk + 6;
            continue;
        }
        const int32_t id = (int32_t)raw_id;
        const char *q = strchr(tk + 6, '"');
        if (!q) break;
        q++;
        const char *e = q;
        while (*e && *e != '"') e += (*e == '\\' && e[1]) ? 2 : 1;
        if (!*e) break;
        const int len = (int)(e - q);
        if (len > 0) {
            if (t->n_special == cap) {
                cap *= 2;
                tok_special *g = (tok_special *)realloc(t->special, (size_t)cap * sizeof *g);
                if (!g) break;
                t->special = g;
            }
            char *txt = (char *)malloc((size_t)len + 1);
            if (!txt) break;
            memcpy(txt, q, (size_t)len);
            txt[len] = 0;
            t->special[t->n_special].text = txt;
            t->special[t->n_special].len = len;
            t->special[t->n_special].id = id;
            t->n_special++;
        }
        p = e;
    }
    free(buf);
    qsort(t->special, (size_t)t->n_special, sizeof *t->special, spec_cmp);
}

/* ---- loading ------------------------------------------------------------ */

waste_tok *waste_tok_open(const char *dir)
{
    char path[512];
    FILE *f = NULL;
    const char *names[] = { "tokenizer.model", "tiktoken.model" };
    for (int i = 0; i < 2 && !f; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        f = fopen(path, "rb");
    }
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    char *raw = (char *)malloc((size_t)sz + 1);
    if (!raw || fread(raw, 1, (size_t)sz, f) != (size_t)sz) { free(raw); fclose(f); return NULL; }
    raw[sz] = 0;
    fclose(f);

    waste_tok *t = (waste_tok *)calloc(1, sizeof *t);
    /* The Kimi pattern until a container says otherwise: every model here
     * before GLM has the Han branch, and a default that has to be set to
     * keep working is a default that will be missed. */
    if (t) { t->han_split = 1; t->digit_run = 3; }
    if (!t) { free(raw); return NULL; }
    t->blob = (uint8_t *)malloc((size_t)sz);          /* decoded is smaller */
    t->cap_tokens = 4096;
    t->by_rank = (tok_entry *)malloc(sizeof(tok_entry) * (size_t)t->cap_tokens);
    if (!t->blob || !t->by_rank) { free(raw); waste_tok_free(t); return NULL; }

    size_t used = 0;
    char *line = raw, *end = raw + sz;
    while (line < end) {
        char *nl = memchr(line, '\n', (size_t)(end - line));
        if (!nl) nl = end;
        char *sp = memchr(line, ' ', (size_t)(nl - line));
        if (sp) {
            /* Grown rather than a fixed 300000, which was a bound on a
             * vocabulary size taken from the one model in hand and
             * enforced by nothing — a larger tokenizer.model wrote past
             * the end of the array. */
            if (t->n_tokens == t->cap_tokens) {
                const int nc = t->cap_tokens * 2;
                tok_entry *g = (tok_entry *)realloc(t->by_rank,
                                                    sizeof(tok_entry) * (size_t)nc);
                if (!g) { free(raw); waste_tok_free(t); return NULL; }
                t->by_rank = g;
                t->cap_tokens = nc;
            }
            const int nb = b64decode(line, (int)(sp - line), t->blob + used);
            const int rank = atoi(sp + 1);
            t->by_rank[t->n_tokens].p = t->blob + used;
            t->by_rank[t->n_tokens].len = nb;
            t->by_rank[t->n_tokens].rank = rank;
            t->n_tokens++;
            used += (size_t)nb;
            /* Bounded the same way cfg_sane bounds vocab_size: the index
             * below is a flat array, and a rank field of 2^31-1 in a file
             * we did not write is an 8 GB allocation. */
            if (rank > t->max_id && rank <= (1 << 24)) t->max_id = rank;
        }
        line = nl + 1;
    }
    free(raw);

    {   /* rank -> entry, so decode is a lookup instead of a scan */
        const size_t n = (size_t)t->max_id + 1;
        t->by_id = (int32_t *)malloc(n * sizeof(int32_t));
        if (!t->by_id) { waste_tok_free(t); return NULL; }
        for (size_t i = 0; i < n; i++) t->by_id[i] = -1;
        for (int i = 0; i < t->n_tokens; i++) {
            const int r = t->by_rank[i].rank;
            /* first wins, matching the scan this replaces */
            if (r >= 0 && r <= t->max_id && t->by_id[r] < 0) t->by_id[r] = i;
        }
    }

    int hs = 1;
    while (hs < t->n_tokens * 2) hs <<= 1;
    t->hash_mask = hs - 1;
    t->hash = (int32_t *)malloc((size_t)hs * sizeof(int32_t));
    if (!t->hash) { waste_tok_free(t); return NULL; }
    memset(t->hash, 0xff, (size_t)hs * sizeof(int32_t));
    for (int i = 0; i < t->n_tokens; i++) tok_index(t, i);

    /* Kimi's reserved block starts right after the base vocabulary:
     * [BOS] = base, [EOS] = base+1, <|end_of_msg|> = base+2 — and it is
     * the third of those, not [EOS], that generation_config.json names as
     * eos_token_id on both models. This positional default is a guess
     * that happens to be right here; waste_tok_set_eos replaces it with
     * the container's stated id when there is one. */
    t->bos = t->n_tokens;
    t->eos = t->n_tokens + 2;
    load_specials(t, dir);
    return t;
}

void waste_tok_free(waste_tok *t)
{
    if (!t) return;
    for (int i = 0; i < t->n_special; i++) free(t->special[i].text);
    free(t->special);
    free(t->blob); free(t->by_rank); free(t->hash); free(t->by_id); free(t);
}

int waste_tok_vocab(const waste_tok *t) { return t->n_tokens; }
int waste_tok_bos(const waste_tok *t) { return t->bos; }
int waste_tok_eos(const waste_tok *t) { return t->eos; }

void waste_tok_set_eos(waste_tok *t, int id) { if (t && id > 0) t->eos = id; }
void waste_tok_set_han_split(waste_tok *t, int on) { if (t) t->han_split = !!on; }

void waste_tok_set_digit_run(waste_tok *t, int n)
{
    /* A container that states something outside what any of these patterns
     * spell is a container this cannot honour; keep the default rather
     * than invent a third behaviour. */
    if (t && (n == 1 || n == 3)) t->digit_run = n;
}

/* ---- UTF-8 + the character classes the pattern needs -------------------- */

static int utf8_next(const char *s, int len, int *cp)
{
    const unsigned char *u = (const unsigned char *)s;
    if (len <= 0) { *cp = -1; return 0; }
    if (u[0] < 0x80) { *cp = u[0]; return 1; }
    if ((u[0] & 0xe0) == 0xc0 && len >= 2) { *cp = ((u[0] & 0x1f) << 6) | (u[1] & 0x3f); return 2; }
    if ((u[0] & 0xf0) == 0xe0 && len >= 3) {
        *cp = ((u[0] & 0x0f) << 12) | ((u[1] & 0x3f) << 6) | (u[2] & 0x3f); return 3;
    }
    if ((u[0] & 0xf8) == 0xf0 && len >= 4) {
        *cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3f) << 12) | ((u[2] & 0x3f) << 6) | (u[3] & 0x3f);
        return 4;
    }
    *cp = u[0];
    return 1;
}

static int is_han(int c)
{
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x20000 && c <= 0x2FA1F);
}

/* \p{L} plus \p{M}: ASCII and Latin-1 letters, Latin Extended, Greek,
 * Cyrillic, Hebrew, Arabic, Hangul, Hiragana/Katakana, combining marks.
 * Han is excluded on purpose — the pattern gives it its own branch. */
static int is_letter_ex(int c, int han_split)
{
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    if (is_han(c)) return !han_split;
    return (c >= 0xC0 && c <= 0x24F) || (c >= 0x300 && c <= 0x36F) ||
           (c >= 0x370 && c <= 0x3FF) || (c >= 0x400 && c <= 0x52F) ||
           (c >= 0x590 && c <= 0x6FF) || (c >= 0x3040 && c <= 0x30FF) ||
           (c >= 0xAC00 && c <= 0xD7AF) || (c >= 0x1E00 && c <= 0x1EFF);
}

__attribute__((unused)) static int is_upper(int c)
{
    if (c < 0x80) return c >= 'A' && c <= 'Z';
    return (c >= 0xC0 && c <= 0xDE) || (c >= 0x391 && c <= 0x3A9) ||
           (c >= 0x410 && c <= 0x42F);
}

static int is_digit(int c) { return c >= '0' && c <= '9'; }
static int is_space(int c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' ||
                                    c == '\f' || c == '\v' || c == 0xA0; }
static int is_nl(int c) { return c == '\r' || c == '\n'; }

/* Advances one pre-token, returning its byte length. Mirrors the branch
 * order of the model's pat_str. */
static int next_piece(const char *s, int len, int han_split, int digit_run)
{
    int cp, n = utf8_next(s, len, &cp), i;
    if (n == 0) return 0;

    if (han_split && is_han(cp)) {                      /* [\p{Han}]+ */
        i = n;
        while (i < len) { int c2, k = utf8_next(s + i, len - i, &c2);
                          if (!is_han(c2)) break; i += k; }
        return i;
    }

    /* (?i:'s|'t|'re|'ve|'m|'ll|'d) — both letter branches take it */
    #define SUFFIX_AT(i) do { \
        if ((i) < len && s[i] == '\'') { \
            static const char *sfx[] = { "s", "t", "re", "ve", "m", "ll", "d" }; \
            for (size_t q = 0; q < sizeof sfx / sizeof *sfx; q++) { \
                const int sl = (int)strlen(sfx[q]); \
                if ((i) + 1 + sl <= len && \
                    strncasecmp(s + (i) + 1, sfx[q], (size_t)sl) == 0) { \
                    (i) += 1 + sl; break; \
                } \
            } \
        } \
    } while (0)

    /* optional leading non-letter/non-digit, then a letter run */
    if (!is_letter_ex(cp, han_split) && !is_digit(cp) && !is_nl(cp)) {
        int c2, k2, j = n;
        k2 = utf8_next(s + j, len - j, &c2);
        if (k2 && is_letter_ex(c2, han_split)) {
            i = j;
            while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                              if (!is_letter_ex(c3, han_split)) break; i += k; }
            SUFFIX_AT(i);
            return i;
        }
    }
    if (is_letter_ex(cp, han_split)) {
        i = n;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (!is_letter_ex(c3, han_split)) break; i += k; }
        SUFFIX_AT(i);
        return i;
    }

    if (is_digit(cp)) {                     /* \p{N}{1,digit_run} */
        i = n;
        int cnt = 1;
        while (i < len && cnt < digit_run) { int c3, k = utf8_next(s + i, len - i, &c3);
                                             if (!is_digit(c3)) break; i += k; cnt++; }
        return i;
    }

    /* " ?[^\s\p{L}\p{N}]+[\r\n]*" */
    {
        int j = 0;
        if (cp == ' ') {
            int c2, k2 = utf8_next(s + n, len - n, &c2);
            if (k2 && !is_space(c2) && !is_letter_ex(c2, han_split) && !is_digit(c2)) j = n;
        }
        int save = j;
        i = j;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (is_space(c3) || is_letter_ex(c3, han_split) || is_digit(c3)) break; i += k; }
        if (i > save) {
            while (i < len && is_nl(s[i])) i++;
            return i;
        }
    }

    /* "\s*[\r\n]+" then "\s+(?!\S)" then "\s+" */
    if (is_space(cp)) {
        i = 0;
        int last_nl = -1;
        while (i < len) { int c3, k = utf8_next(s + i, len - i, &c3);
                          if (!is_space(c3)) break;
                          i += k;
                          if (is_nl(c3)) last_nl = i; }
        if (last_nl > 0) return last_nl;
        /* \s+(?!\S): keep all but the last space when more text follows */
        if (i < len && i > 1) return i - 1;
        return i;
    }
    return n;
}

/* ---- byte-pair merge ---------------------------------------------------- */

/* The merge loop below rescans the whole piece for each merge, so it is
 * O(n^2) in the piece length. This bounds that — it is a window, not a
 * limit: encode_piece feeds a longer pre-token through in windows. */
#define MAXPIECE 1024

static int encode_window(const waste_tok *t, const uint8_t *p, int len,
                         int32_t *out, int cap)
{
    if (len == 0) return 0;
    const int r = tok_rank(t, p, len);
    if (r >= 0) { if (cap < 1) return -1; out[0] = r; return 1; }

    int start[MAXPIECE + 1], rank[MAXPIECE + 1], n = 0;
    for (int i = 0; i < len; i++) { start[n] = i; rank[n] = INT32_MAX; n++; }
    start[n] = len; rank[n] = INT32_MAX;

    for (int i = 0; i + 1 < n; i++) {
        const int rr = tok_rank(t, p + start[i], start[i + 2] - start[i]);
        rank[i] = rr < 0 ? INT32_MAX : rr;
    }

    for (;;) {
        int best = -1, bestr = INT32_MAX;
        for (int i = 0; i + 1 < n; i++)
            if (rank[i] < bestr) { bestr = rank[i]; best = i; }
        if (best < 0) break;

        /* merge best with best+1 */
        for (int i = best + 1; i < n; i++) { start[i] = start[i + 1]; rank[i] = rank[i + 1]; }
        n--;
        for (int i = (best > 0 ? best - 1 : 0); i <= best && i + 1 < n; i++) {
            const int rr = tok_rank(t, p + start[i], start[i + 2] - start[i]);
            rank[i] = rr < 0 ? INT32_MAX : rr;
        }
    }

    if (n > cap) return -1;
    for (int i = 0; i < n; i++) {
        const int rr = tok_rank(t, p + start[i], start[i + 1] - start[i]);
        out[i] = rr < 0 ? 0 : rr;
    }
    return n;
}

/* One pre-token, every byte of it.
 *
 * A piece longer than the window is encoded in windows. It used to be
 * *truncated* to 256 bytes while the caller advanced past the whole
 * piece, so the rest was dropped from the prompt without a word. The
 * pre-tokenizer gives Han its own branch and consumes an entire
 * unpunctuated run, which put that cliff at 85 Chinese characters:
 * `好` x 86 went in as 258 bytes and came back as 256, cut mid-codepoint.
 * A markdown rule of 300 dashes and a run of 300 newlines went the same
 * way. tiktoken has no such limit and round-trips all of them.
 *
 * Windowing is still a divergence from the reference, which runs the BPE
 * over the whole regex match — but only at a seam, and only for pieces
 * past 1024 bytes, which no ordinary text reaches. Upstream splits runs
 * at 25000 characters for the same O(n^2) reason (transcribed in
 * serve/engine.py as _MAX_RUN_CHARS), so the compromise is theirs; this
 * one is merely tighter. Seams land on UTF-8 boundaries, so a window
 * never splits a codepoint the way the truncation did. */
static int encode_piece(const waste_tok *t, const uint8_t *p, int len,
                        int32_t *out, int cap)
{
    if (len <= MAXPIECE) return encode_window(t, p, len, out, cap);

    int n = 0, off = 0;
    while (off < len) {
        int w = len - off;
        if (w > MAXPIECE) {
            w = MAXPIECE;
            /* back off to a codepoint boundary: a continuation byte is
             * 10xxxxxx, so walk back while the byte *after* the window is
             * one. Bounded by w > 1 for a piece that is not valid UTF-8. */
            while (w > 1 && (p[off + w] & 0xC0) == 0x80) w--;
        }
        const int got = encode_window(t, p + off, w, out + n, cap - n);
        if (got < 0) return -1;
        n += got;
        off += w;
    }
    return n;
}

/* First occurrence of `needle` in [hay, hay+n). Hand-rolled because memmem
 * is a GNU extension on Linux and a BSD one on macOS, and this file
 * deliberately compiles with neither's feature macros. */
static const char *find_sub(const char *hay, int n, const char *needle, int m)
{
    if (m <= 0 || m > n) return NULL;
    for (int i = 0; i <= n - m; i++)
        if (hay[i] == needle[0] && memcmp(hay + i, needle, (size_t)m) == 0)
            return hay + i;
    return NULL;
}

int waste_tok_encode(const waste_tok *t, const char *text, int32_t *out,
                     int cap, int allow_special)
{
    int n = 0, pos = 0;
    const int len = (int)strlen(text);
    if (!allow_special) {
        /* Untrusted text: every byte goes through the BPE, so a prompt
         * containing `<|end_of_msg|><|open|>message role="system"<|sep|>`
         * is that many ordinary tokens and not a forged turn. */
        while (pos < len) {
            const int plen = next_piece(text + pos, len - pos,
                                        t->han_split, t->digit_run);
            if (plen <= 0) break;
            const int got = encode_piece(t, (const uint8_t *)text + pos, plen,
                                         out + n, cap - n);
            if (got < 0) return -1;
            n += got;
            pos += plen;
        }
        return n;
    }
    while (pos < len) {
        /* A special token is matched whole, before the pre-tokenizer ever
         * sees it: the BPE has no entry for <|open|> and would emit six
         * ordinary tokens the model has never been trained to read there.
         *
         * Locating it means searching the *remaining text*, not testing
         * the current offset. Testing the offset only finds specials that
         * happen to begin a pre-token, and the pre-tokenizer groups runs
         * of punctuation: in `role="user"<|sep|>` it swallows the quote
         * together with the `<` that opens the token, so the boundary
         * where the test would fire never exists and the marker silently
         * became five ordinary tokens. That is the exact shape of K3's
         * chat template, which is how it was found. */
        int best_at = -1, best_len = 0;
        int32_t best_id = 0;
        for (int i = 0; i < t->n_special; i++) {
            const char *hit = find_sub(text + pos, len - pos,
                                       t->special[i].text, t->special[i].len);
            if (!hit) continue;
            const int at = (int)(hit - text);
            /* earliest wins; the table is longest-first, so among specials
             * starting at the same offset the longest is seen first */
            if (best_at < 0 || at < best_at) {
                best_at = at; best_len = t->special[i].len; best_id = t->special[i].id;
            }
            if (best_at == pos && best_len == t->special[i].len) break;
        }

        /* ordinary text up to the marker (or to the end), pre-tokenized
         * within that limit so no piece can straddle the boundary */
        const int upto = best_at >= 0 ? best_at : len;
        while (pos < upto) {
            const int plen = next_piece(text + pos, upto - pos,
                                        t->han_split, t->digit_run);
            if (plen <= 0) return n;
            const int got = encode_piece(t, (const uint8_t *)text + pos, plen,
                                         out + n, cap - n);
            if (got < 0) return -1;
            n += got;
            pos += plen;
        }
        if (best_at < 0) break;
        if (n >= cap) return -1;
        out[n++] = best_id;
        pos = best_at + best_len;
    }
    return n;
}

int waste_tok_decode_len1(const waste_tok *t, int32_t id)
{
    /* Specials first: they are not in the rank table, and returning zero
     * bytes for them silently deleted them from every decode. That is how
     * `<|close|>response<|sep|>` reached a terminal as the bare word
     * "response" — and why a stop string written in markers could never
     * match the text it was compared against. Encode and decode have to
     * agree about what a token is. */
    for (int i = 0; i < t->n_special; i++)
        if (t->special[i].id == id) return t->special[i].len;
    if (id >= 0 && id <= t->max_id && t->by_id) {
        const int32_t i = t->by_id[id];
        if (i >= 0) return t->by_rank[i].len;
    }
    return 0;
}

int waste_tok_decode1(const waste_tok *t, int32_t id, char *buf, int cap)
{
    if (!buf || cap <= 0) return 0;
    const int full = waste_tok_decode_len1(t, id);
    const int l = full < cap ? full : cap;
    if (!l) return 0;
    for (int i = 0; i < t->n_special; i++)
        if (t->special[i].id == id) {
            memcpy(buf, t->special[i].text, (size_t)l);
            return l;
        }
    const int32_t i = t->by_id[id];
    memcpy(buf, t->by_rank[i].p, (size_t)l);
    return l;
}

int waste_tok_decode(const waste_tok *t, const int32_t *ids, int n,
                     char *buf, int cap)
{
    int w = 0;
    for (int i = 0; i < n && w < cap; i++)
        w += waste_tok_decode1(t, ids[i], buf + w, cap - w);
    return w;
}
