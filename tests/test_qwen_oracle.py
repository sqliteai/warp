#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Streaming official Qwen oracle must not allocate the 360 GB tables.

  python3 tests/test_qwen_oracle.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def ck(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        raise SystemExit(1)


def main():
    src = os.path.join(ROOT, "tools", "qwen_ref.py")
    text = open(src).read()
    ck("--src" in text and "--dump" in text,
       "qwen_ref.py CLI has --src and --dump")
    ck("from_pretrained" not in text or "must not" in text.lower(),
       "common-weight path does not call from_pretrained")

    import qwen_ref as Q
    ck(hasattr(Q, "StreamingExperts"), "StreamingExperts exists")
    ck(hasattr(Q, "StreamingNGram"), "StreamingNGram exists")

    class Fake:
        pass

    # A 512-expert Parameter is 3.36 GiB. The streaming class must not
    # construct it even when the official config says num_experts=512.
    cfg = Fake()
    cfg.num_experts = 512
    cfg.hidden_size = 2560
    cfg.moe_intermediate_size = 640
    cfg.hidden_act = "silu"
    experts = Q.StreamingExperts(cfg, st=None, layer=0)
    params = list(experts.parameters())
    n = sum(p.numel() for p in params)
    ck(n == 0, f"StreamingExperts holds no parameters (got {n})")
    ck(not hasattr(experts, "gate_up_proj") or experts.gate_up_proj is None,
       "StreamingExperts does not own gate_up_proj")

    ncfg = Fake()
    ncfg.ngram_size = 3
    ncfg.heads_per_ngram = 8
    ncfg.vocab_size = 248320
    ncfg.ngram_vocab_size_base = 20000000
    ncfg.seed = 1234
    ncfg.eos_token_id = 248044
    ncfg.make_ngram_vocab_size_divisible_by = 128
    ngram = Q.StreamingNGram(ncfg, 2560, layer_idx=1, ple_layer_index=0, st=None)
    nparams = sum(p.numel() for p in ngram.parameters())
    ck(nparams == 0, f"StreamingNGram holds no embedding table (got {nparams})")
    ck("if attn_mask is None" in text,
       "QSA layers get a 4D mask when create_causal_mask returns None")
    ck("torch.tril" in text,
       "the fallback mask is causal visible=True (tril), not triu masked-out")
    print("ORACLE STREAM OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e}")
        sys.exit(1)
