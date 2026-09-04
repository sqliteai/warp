#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Synthetic tests for qwen_compare_oracle near-tie route gate."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from qwen_compare_oracle import check_route_ids, score_err_bound, _tolerance


def ck(cond, msg):
    if not cond:
        print(f"FAIL {msg}")
        raise SystemExit(1)
    print(f"  ok  {msg}")


def main():
    cfg = {"num_hidden_layers": 48, "hidden_size": 2560,
           "num_experts_per_tok": 10, "hc_count": 4}
    h_bound = _tolerance("hidden", cfg)
    score_err = score_err_bound(cfg, h_bound)
    w_bound = _tolerance("route_w", cfg)
    print(f"score_err_bound={score_err:.3e} w_bound={w_bound:.3e}")

    ci = [10, 20, 30, 40]
    pi = [10, 20, 30, 40]
    cw = [0.4, 0.3, 0.2, 0.1]
    pw = [0.4, 0.3, 0.2, 0.1]
    ok, _ = check_route_ids(ci, pi, cw, pw, score_err, w_bound, True)
    ck(ok, "exact match passes")

    # True wrong expert outside near-tie interval
    pi_bad = [10, 20, 30, 99]
    ok, why = check_route_ids(ci, pi_bad, cw, pw, score_err, w_bound, True)
    ck(not ok and "mismatch" in why, f"wrong expert fails: {why}")

    # Near-tie swap: scores 0.100 vs 0.101, margin << score_err
    ci = [1, 2, 3, 4]
    pi = [1, 3, 2, 4]
    cw = [0.30, 0.100, 0.101, 0.05]
    pw = [0.30, 0.101, 0.100, 0.05]
    ok, why = check_route_ids(ci, pi, cw, pw, score_err, w_bound, True)
    ck(ok and why == "near-tie reorder", f"near-tie swap accepted: {why}")

    # Near-tie swap but weight error too large
    pw_bad = [0.30, 0.101, 0.050, 0.05]
    ok, why = check_route_ids(ci, pi, cw, pw_bad, score_err, w_bound, True)
    ck(not ok and "weight" in why, f"near-tie weight fail: {why}")

    # Near-tie swap but hidden layer over bound
    ok, why = check_route_ids(ci, pi, cw, pw, score_err, w_bound, False)
    ck(not ok and "hidden" in why, f"near-tie hidden fail: {why}")

    # Non-near-tie reorder: large margin between swapped experts
    ci = [1, 2, 3, 4]
    pi = [1, 3, 2, 4]
    cw = [0.30, 0.50, 0.10, 0.05]
    pw = [0.30, 0.10, 0.50, 0.05]
    ok, why = check_route_ids(ci, pi, cw, pw, score_err, w_bound, True)
    ck(not ok and "non-near-tie reorder" in why, f"large margin fails: {why}")

    print("NEAR-TIE GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
