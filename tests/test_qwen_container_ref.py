#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Minimal Qwen container-native oracle on the synthetic --qwen fixture.

  python3 tests/test_qwen_container_ref.py
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

try:
    import torch  # noqa: F401
except ImportError:
    print("SKIP: torch is not installed")
    raise SystemExit(77)


def ck(cond, what):
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        raise SystemExit(1)


def f32(path):
    b = open(path, "rb").read()
    n = len(b) // 4
    return list(struct.unpack(f"<{n}f", b)) if n else []


def main():
    import qwen_container_ref as Q

    ck(hasattr(Q, "QwenContainer"), "QwenContainer exists")
    ck(hasattr(Q, "QwenRef"), "QwenRef exists")
    ck(issubclass(Q.QwenContainer, Q.Container),
       "QwenContainer subclasses kimi_ref.Container")

    tmp = tempfile.mkdtemp(prefix="qwen-cref-")
    try:
        cont = os.path.join(tmp, "qwen.waste")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "make_test_container.py"),
             "--qwen", cont],
            cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL fixture", r.stderr[-400:] or r.stdout[-400:])
            return 1
        man = json.load(open(os.path.join(cont, "manifest.json")))
        cfg = man["config"]
        hid = int(cfg["hidden_size"])
        nL = int(cfg["num_hidden_layers"])
        hc = int(cfg["hc_count"])
        vocab = int(cfg["vocab_size"])
        top_k = int(cfg.get("num_experts_per_tok")
                    or cfg.get("num_experts_per_token") or 0)
        ids = "3,7,11"
        ntok = 3

        c = Q.QwenContainer(cont)
        ck(c.iblock == 64, f"index_block is 64 (got {c.iblock})")
        shapes = c.expert_shapes()
        moe = int(cfg["moe_intermediate_size"])
        ck(shapes == [(moe, hid), (moe, hid), (hid, moe)],
           f"expert_shapes gate/up {moe}x{hid}, down {hid}x{moe}")

        dump = os.path.join(tmp, "logits.bin")
        hidden = os.path.join(tmp, "hidden.bin")
        routes = os.path.join(tmp, "routes.txt")
        cli = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "qwen_container_ref.py"),
             "--container", cont, "--ids", ids,
             "--dump", dump, "--hidden", hidden, "--routes", routes],
            cwd=ROOT, capture_output=True, text=True)
        if cli.returncode != 0:
            print("FAIL cli", cli.stderr[-800:] or cli.stdout[-800:])
            return 1
        ck(os.path.isfile(dump), "wrote logits.bin")
        ck(os.path.isfile(hidden), "wrote hidden.bin")
        ck(os.path.isfile(routes), "wrote routes.txt")

        lg = f32(dump)
        ck(len(lg) == vocab, f"logits {len(lg)} floats (vocab {vocab})")

        hid_n = len(f32(hidden))
        want_all = ntok * nL * hc * hid
        want_last = nL * hc * hid
        ck(hid_n in (want_all, want_last),
           f"hidden {hid_n} floats "
           f"(all-tok {want_all} or last-tok {want_last})")

        lines = [ln for ln in open(routes).read().splitlines() if ln.strip()]
        ck(len(lines) == ntok * nL,
           f"routes {len(lines)} lines (want {ntok} tok x {nL} layers)")
        for ln in lines:
            parts = ln.split()
            ck(len(parts) == 2 + 2 * top_k,
               f"route '{ln}' has pos layer + {top_k} ids + {top_k} weights")

        fwd = os.path.join(ROOT, "test_forward")
        if not os.path.isfile(fwd):
            print("CONTAINER REF OK (no test_forward; C diff skipped)")
            return 0

        c_logits = os.path.join(tmp, "c_logits.bin")
        c_hidden = os.path.join(tmp, "c_hidden.bin")
        env = dict(os.environ)
        env["WASTE_DUMP_HIDDEN"] = c_hidden
        env["WASTE_DUMP_ROUTE"] = os.path.join(tmp, "c_routes.txt")
        env["WASTE_Q8"] = "0"
        env["WASTE_BACKEND"] = "cpu"
        fr = subprocess.run(
            [fwd, cont, ids, c_logits, "0"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if fr.returncode != 0:
            print("FAIL test_forward", fr.stderr[-400:] or fr.stdout[-400:])
            return 1
        cl, ol = f32(c_logits), lg
        ck(len(cl) == len(ol), f"C/Python logits length {len(cl)} vs {len(ol)}")
        d = [abs(a - b) for a, b in zip(cl, ol)]
        mx = max(d) if d else 0.0
        ai, bi = cl.index(max(cl)), ol.index(max(ol))
        ck(ai == bi, f"logits argmax {ai} vs {bi}")
        ck(mx < 1e-4, f"logits max|diff| {mx:.3e} < 1e-4")
        cmp = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "qwen_compare_oracle.py"),
             "--container", cont,
             "--c-hidden", c_hidden, "--c-logits", c_logits,
             "--py-hidden", hidden, "--py-logits", dump,
             "--c-routes", os.path.join(tmp, "c_routes.txt"),
             "--py-routes", routes,
             "--tokens", str(ntok)],
            cwd=ROOT, capture_output=True, text=True)
        if cmp.returncode != 0:
            print("FAIL compare", cmp.stdout[-800:] or cmp.stderr[-800:])
            return 1
        print(f"CONTAINER REF OK logits max|diff| {mx:.3e} argmax {ai}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {e}")
        sys.exit(1)
