#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Qwen WASTE_DUMP_HIDDEN writes one hyper-state vector after every layer.

  python3 tests/test_qwen_dump.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    tmp = tempfile.mkdtemp(prefix="qwen-dump-")
    try:
        cont = os.path.join(tmp, "qwen.waste")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tools", "make_test_container.py"),
             "--qwen", cont],
            cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL fixture", r.stderr[-400:])
            return 1
        man = json.load(open(os.path.join(cont, "manifest.json")))
        cfg = man["config"]
        hid = int(cfg["hidden_size"])
        nL = int(cfg["num_hidden_layers"])
        hc = int(cfg["hc_count"])
        dump = os.path.join(tmp, "h.bin")
        env = dict(os.environ)
        env["WASTE_DUMP_HIDDEN"] = dump
        fwd = subprocess.run(
            [os.path.join(ROOT, "test_forward"), cont, "3,7,11",
             os.path.join(tmp, "logits.bin"), "0"],
            cwd=ROOT, env=env, capture_output=True, text=True)
        if fwd.returncode != 0:
            print("FAIL test_forward", fwd.stderr[-400:] or fwd.stdout[-400:])
            return 1
        if not os.path.isfile(dump):
            print("FAIL WASTE_DUMP_HIDDEN wrote no file")
            return 1
        ntok = 3
        want = ntok * nL * hc * hid * 4
        got = os.path.getsize(dump)
        if got != want:
            print(f"FAIL dump {got} bytes, want {want} "
                  f"({ntok} tok x {nL} layers x {hc} streams x {hid} hid x 4)")
            return 1
        print(f"DUMP OK {got} bytes ({ntok}x{nL}x{hc}x{hid})")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
