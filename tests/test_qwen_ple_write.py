#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Q8G PLE heads must quantize in row batches, not as one 12 GiB tensor.

  uv run --with torch --no-project python tests/test_qwen_ple_write.py
"""
from __future__ import annotations

import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed")
    raise SystemExit(77)

import convert as C  # noqa: E402


def ck(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        raise SystemExit(1)


def main():
    torch.manual_seed(0)
    W = torch.randn(96, 160)
    q_full, sc_full, shape = C.quantize_q8g(W)
    ck(shape == [96, 160], f"shape {shape}")

    buf = io.BytesIO()
    meta = C.write_q8g_row_chunks(buf, [W[:32], W[32:64], W[64:]], 160)
    ck(meta["shape"] == [96, 160], f"streamed shape {meta['shape']}")
    blob = buf.getvalue()
    q_n = q_full.numel()
    sc_n = sc_full.numel() * 2  # fp16
    ck(meta["off"] == 0, "payload starts at 0")
    ck(meta["scale_off"] == q_n, f"scales follow int8 ({meta['scale_off']} vs {q_n})")
    ck(len(blob) == q_n + sc_n, f"bytes {len(blob)} vs {q_n + sc_n}")
    ck(blob[:q_n] == C.raw_bytes(q_full), "chunked Q8G payload matches full-head")
    ck(blob[q_n:] == C.raw_bytes(sc_full), "chunked Q8G scales match full-head")

    q8 = torch.arange(256, dtype=torch.int8)
    got = C.raw_bytes(q8)
    ck(got == bytes(range(256)), "raw_bytes of int8 is the storage bytes")

    from verify_container import dequant_q8g, dequant_q8g_row
    full = dequant_q8g(C.raw_bytes(q_full), C.raw_bytes(sc_full), [96, 160])
    pad = 256  # 160 padded to 2*128
    q_b = C.raw_bytes(q_full)
    sc_b = C.raw_bytes(sc_full)
    row = dequant_q8g_row(q_b[5 * pad:6 * pad], sc_b[5 * 4:6 * 4], 160)
    ck(torch.allclose(row, full[5], atol=1e-5),
       "dequant_q8g_row matches a slice of the full-head dequant")

    print("PLE WRITE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
