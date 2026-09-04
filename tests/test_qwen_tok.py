#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_qwen_tok.py — the Qwen pre-tokenizer, against the release's own.

Two halves, because only one of them needs 335 GB on disk.

The first runs anywhere: Qwen's `tokenizer.json` states a pre-tokenization
pattern that differs from Kimi's and GLM's in one way the engine cannot
ignore — `\\p{N}` where they have `\\p{N}{1,3}`, so every digit is its own
piece. tools/hf_tokenizer.py has to recognise that spelling and report
it, and convert.py has to carry it to the container as
`tokenizer_digit_run`; both are pinned here without any weights.

The second needs the pinned checkpoint (WASTE_QWEN_SRC) and the
`tokenizers` package, and compares src/tokenizer.c's ids against the
release's tokenizer for a handful of strings — including numbers, which
is exactly where a wrong digit run shows up and where nothing else in
this suite would catch it.

  python3 tests/test_qwen_tok.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hf_tokenizer as HFT  # noqa: E402

PINNED = os.environ.get("WASTE_QWEN_SRC",
                        "/Users/admin/mnt/llm/qwen38-flash-next/raw")

# What Qwen3.8-Flash-Next's tokenizer.json states, verbatim.
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?"
                r"[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|"
                r"\s*[\r\n]+|\s+(?!\S)|\s+")

STRINGS = (
    "hello",
    "Hello, world!",
    "你好",
    "Qwen3.8-Flash-Next",
    # The digit run. "2026" is one piece under \p{N}{1,3} and four under
    # \p{N}, and the ids differ with no error anywhere.
    "2026",
    "It cost 1234567 yen in 2026.",
    "3.14159",
)

fails = 0


def ck(cond, what):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def encode_c(tokdir, text, plain, digits):
    env = os.environ.copy()
    env["WASTE_TOK_PLAIN"] = "1" if plain else "0"
    env["WASTE_TOK_NOHAN"] = "1"
    env["WASTE_TOK_DIGITS"] = str(digits)
    out = subprocess.check_output(
        [os.path.join(ROOT, "test_tokenizer"), tokdir, text], env=env, text=True)
    parts = out.strip().split()
    n = int(parts[0])
    ids = [int(x) for x in parts[1:]]
    if len(ids) != n:
        raise RuntimeError(f"C count {n} vs {len(ids)} ids for {text!r}")
    return ids


def _digit_tokens(rank_text):
    """Vocabulary entries that are two or more digits, if any."""
    import base64
    for line in rank_text.splitlines():
        if not line:
            continue
        raw = base64.b64decode(line.split()[0])
        if len(raw) > 1 and all(0x30 <= b <= 0x39 for b in raw):
            yield raw.decode()


def digit_run_changes_pieces():
    r"""The flag on a vocabulary that *does* hold a multi-digit token.

    Model-independent: 256 single bytes plus "20". Under \p{N}{1,3} the
    pre-token is "202" and BPE reaches the "20" merge, so "2026" is three
    ids; under \p{N} every digit is its own pre-token and it is four.
    """
    import base64
    tmp = tempfile.mkdtemp(prefix="digitrun-")
    try:
        lines = [base64.b64encode(bytes([b])).decode() + f" {b}"
                 for b in range(256)]
        lines.append(base64.b64encode(b"20").decode() + " 256")
        with open(os.path.join(tmp, "tokenizer.model"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        three = encode_c(tmp, "2026", plain=True, digits=3)
        one = encode_c(tmp, "2026", plain=True, digits=1)
        ck(three == [256, 0x32, 0x36],
           f"\\p{{N}}{{1,3}} merges 20 out of 202: {three}")
        ck(one == [0x32, 0x30, 0x32, 0x36],
           f"\\p{{N}} keeps every digit apart: {one}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("Qwen's pattern is one src/tokenizer.c implements")
    ck(QWEN_PATTERN in HFT.KNOWN_PATTERNS,
       "hf_tokenizer recognises the Qwen spelling")
    han, digit_run = HFT.KNOWN_PATTERNS[QWEN_PATTERN]
    ck(han == 0, "Han has no branch of its own, so tokenizer_han_split is false")
    ck(digit_run == 1, f"one digit per piece ({digit_run})")
    ck(HFT.KNOWN_PATTERNS[HFT.PAT_HAN] == (1, 3), "Kimi's is unchanged")
    ck(HFT.KNOWN_PATTERNS[HFT.PAT_NO_HAN] == (0, 3), "GLM's is unchanged")

    if os.path.exists(os.path.join(ROOT, "test_tokenizer")):
        print("tokenizer_digit_run changes where a piece ends")
        digit_run_changes_pieces()

    if not os.path.isfile(os.path.join(PINNED, "tokenizer.json")):
        print(f"SKIP the parity half: no tokenizer.json under {PINNED} "
              f"(set WASTE_QWEN_SRC)")
        return 77 if not fails else 1
    if not os.path.exists(os.path.join(ROOT, "test_tokenizer")):
        print("SKIP the parity half: test_tokenizer is not built (make test)")
        return 77 if not fails else 1
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("SKIP the parity half: the `tokenizers` package is not installed")
        return 77 if not fails else 1

    print("C tokenizer vs the release's own")
    text, han, specials, digit_run = HFT.convert(PINNED, quiet=True)
    ck(not han and digit_run == 1,
       f"the pinned checkpoint reports han={han} digit_run={digit_run}")
    hf = Tokenizer.from_file(os.path.join(PINNED, "tokenizer.json"))
    tmp = tempfile.mkdtemp(prefix="qwen-tok-")
    try:
        with open(os.path.join(tmp, "tokenizer.model"), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(text)
        if specials:
            json.dump(specials, open(os.path.join(tmp, "specials.json"), "w"),
                      indent=1)
        for s in STRINGS:
            got = encode_c(tmp, s, plain=True, digits=digit_run)
            want = hf.encode(s, add_special_tokens=False).ids
            ck(got == want, f"{s!r} -> {got if got != want else got[:12]}"
                            + ("" if got == want else f"  want {want}"))

        # This vocabulary has no multi-digit token at all — it was trained
        # under \p{N} — so the two runs happen to agree on it: "202" has
        # no merge to reach and comes back out as three ids either way.
        # That is a property of Qwen's vocabulary, not of the setting, and
        # it is why the flag is checked on its own below rather than here.
        multi = list(_digit_tokens(text))
        ck(not multi, f"no multi-digit token in the Qwen vocabulary {multi[:5]}")

        # Markup vs content: a control token must not be forgeable from
        # ordinary text. See waste_tokenize / waste_tokenize_markup.
        marker = next((e["text"] for e in specials
                       if e["text"].startswith("<|") and e["text"].endswith("|>")),
                      None)
        if marker:
            as_markup = encode_c(tmp, marker, plain=False, digits=digit_run)
            as_plain = encode_c(tmp, marker, plain=True, digits=digit_run)
            ck(as_markup != as_plain,
               f"markup {marker!r} {as_markup} is not what content gives")
            ck(len(as_markup) == 1, f"markup {marker!r} is one control id")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("QWEN TOK FAILED" if fails else "QWEN TOK OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
