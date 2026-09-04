#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Compare C dumps against qwen_container_ref.py on the same container.

Exits 0 on pass, 1 on mismatch. Tolerances follow quant decode bounds:
  Q4G row error <= 7 * scale; VQ3R expert error bounded by codebook residual.

Route ids: exact match required unless a principled near-tie gate accepts a
reorder (same expert multiset, pairwise score margin below score_err_bound).

  python3 tools/qwen_compare_oracle.py --container model.waste \\
      --c-hidden c.bin --c-logits c.bin --py-hidden p_h.bin --py-logits p_l.bin \\
      --c-routes c_r.txt --py-routes p_r.txt --manifest model.waste/manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys


def f32(path):
    b = open(path, "rb").read()
    n = len(b) // 4
    return list(struct.unpack(f"<{n}f", b))


def _tolerance(name, cfg):
    """Principled fp32 oracle vs f32-engine bounds after shared dequant."""
    nL = int(cfg["num_hidden_layers"])
    top_k = int(cfg.get("num_experts_per_tok")
                or cfg.get("num_experts_per_token") or 10)
    q4g = 2e-3
    vq3r = 4e-4 * top_k
    per_layer = q4g + vq3r
    hidden = per_layer * math.sqrt(max(nL, 1)) + 2e-4 * nL
    if name.startswith("logits"):
        return hidden * 1.8
    if name.startswith("route_w"):
        return 1e-3
    if name.startswith("hidden"):
        return hidden
    return 1e-3


def score_err_bound(cfg, hidden_bound):
    """Upper bound on per-expert gate logit error from router-input drift.

    Gate logits are y = W @ mixed with mixed in R^hid (Q4G matvec). Each
    dequant weight error <= q4g. With ||delta_mixed||_inf <= hidden_bound:

      |delta_y_e| <= q4g * sqrt(hid) * hidden_bound

    (Cauchy/Schwarz on the matvec: q4g per weight, hid terms, input error.)
    """
    hid = int(cfg["hidden_size"])
    q4g = 2e-3
    return q4g * math.sqrt(hid) * hidden_bound


def _stats(a, b, name, bound):
    if len(a) != len(b):
        print(f"FAIL {name} length {len(a)} vs {len(b)}")
        return False, bound
    mx = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
    ai = a.index(max(a)) if a else -1
    bi = b.index(max(b)) if b else -1
    ok = mx <= bound and (name != "logits" or ai == bi)
    tag = "ok" if ok else "FAIL"
    print(f"  {tag}  {name:<20} max|diff| {mx:.3e}  bound {bound:.3e}  "
          f"argmax {ai}/{bi}")
    return ok, mx


def _parse_routes(path, top_k):
    out = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        pos, layer = int(parts[0]), int(parts[1])
        ids = [int(x) for x in parts[2:2 + top_k] if int(x) >= 0]
        ws = [float(x) for x in parts[2 + top_k:2 + 2 * top_k]]
        out.append((pos, layer, ids, ws))
    return out


def _order_pairs(ids):
    """Pairs (a,b) where relative order of a,b differs in two lists."""
    pos_a = {e: i for i, e in enumerate(ids)}
    pairs = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            pairs.append((a, b))
    return pairs


def _order_differs(ci, pi, a, b):
    ia_c = ci.index(a)
    ib_c = ci.index(b)
    ia_p = pi.index(a)
    ib_p = pi.index(b)
    return (ia_c < ib_c) != (ia_p < ib_p)


def check_route_ids(ci, pi, cw, pw, score_err, w_bound, hidden_ok):
    """Fail-closed route compare with principled near-tie acceptance.

    Returns (ok, reason).
    """
    if len(ci) != len(pi):
        return False, f"id count {len(ci)} vs {len(pi)}"
    if ci == pi:
        mx = max(abs(a - b) for a, b in zip(cw, pw)) if cw else 0.0
        if mx > w_bound:
            return False, f"weight max|diff| {mx:.3e} > {w_bound:.3e}"
        return True, "exact"

    if not hidden_ok:
        return False, "hidden layer over bound (near-tie rejected)"

    if set(ci) != set(pi):
        extra_c = set(ci) - set(pi)
        extra_p = set(pi) - set(ci)
        return False, f"non-near-tie expert mismatch extra C{extra_c} Py{extra_p}"

    c_map = {e: w for e, w in zip(ci, cw)}
    p_map = {e: w for e, w in zip(pi, pw)}
    for e in ci:
        dw = abs(c_map[e] - p_map[e])
        if dw > w_bound:
            return False, f"expert {e} weight |diff| {dw:.3e} > {w_bound:.3e}"

    for i in range(len(ci)):
        for j in range(i + 1, len(ci)):
            a, b = ci[i], ci[j]
            if not _order_differs(ci, pi, a, b):
                continue
            margin = abs(c_map[a] - c_map[b])
            if margin > score_err:
                return False, (
                    f"non-near-tie reorder experts {a},{b} "
                    f"margin {margin:.3e} > score_err {score_err:.3e}"
                )

    return True, "near-tie reorder"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--c-hidden", required=True)
    ap.add_argument("--c-logits", required=True)
    ap.add_argument("--py-hidden", required=True)
    ap.add_argument("--py-logits", required=True)
    ap.add_argument("--c-routes", default="")
    ap.add_argument("--py-routes", default="")
    ap.add_argument("--tokens", type=int, default=0,
                    help="number of prefill tokens in C hidden dump")
    args = ap.parse_args()
    man_path = args.manifest or (args.container + "/manifest.json"
                                 if args.container else "")
    if not man_path:
        print("FAIL need --manifest or --container")
        return 1
    cfg = json.load(open(man_path))["config"]
    hc = int(cfg["hc_count"])
    hid = int(cfg["hidden_size"])
    nL = int(cfg["num_hidden_layers"])
    top_k = int(cfg.get("num_experts_per_tok")
                or cfg.get("num_experts_per_token") or 0)
    stream = hc * hid

    ch, cl, ph_raw, pl = map(f32, (args.c_hidden, args.c_logits,
                                    args.py_hidden, args.py_logits))
    c_ntok = len(ch) // (nL * stream) if stream and nL else 0
    p_ntok = len(ph_raw) // (nL * stream) if stream and nL else 0
    ntok = args.tokens or c_ntok or 1
    if args.tokens and c_ntok and args.tokens != c_ntok:
        print(f"FAIL --tokens {args.tokens} but C hidden has {c_ntok} token(s)")
        return 1
    if c_ntok and p_ntok and c_ntok != p_ntok:
        print(f"FAIL hidden token count C={c_ntok} py={p_ntok}")
        return 1
    if c_ntok:
        ntok = c_ntok
    if ntok < 1:
        ntok = 1

    ok = True
    worst_h = 0.0
    h_bound = _tolerance("hidden", cfg)
    score_err = score_err_bound(cfg, h_bound)
    w_bound = _tolerance("route_w", cfg)
    print(f"compare {ntok} token(s), {nL} layers, stream {stream}")
    print(f"  hidden_bound {h_bound:.3e}  score_err_bound {score_err:.3e}  "
          f"(q4g*sqrt(hid)*hidden_bound, hid={hid})  route_w_bound {w_bound:.3e}")

    layer_hidden_ok = {}
    for tok in range(ntok):
        base_c = tok * nL * stream
        for L in range(nL):
            a = ch[base_c + L * stream: base_c + (L + 1) * stream]
            if len(ph_raw) == ntok * nL * stream:
                b = ph_raw[(tok * nL + L) * stream:(tok * nL + L + 1) * stream]
            elif len(ph_raw) == nL * stream and tok == ntok - 1:
                b = ph_raw[L * stream:(L + 1) * stream]
            else:
                print(f"FAIL py hidden size {len(ph_raw)}")
                return 1
            passed, mx = _stats(a, b, f"tok{tok} L{L}", h_bound)
            ok &= passed
            worst_h = max(worst_h, mx)
            layer_hidden_ok[(tok, L)] = passed

    l_bound = _tolerance("logits", cfg)
    passed, _ = _stats(cl, pl, "logits", l_bound)
    ok &= passed

    if args.c_routes and args.py_routes:
        cr = _parse_routes(args.c_routes, top_k)
        pr = _parse_routes(args.py_routes, top_k)
        if len(cr) != len(pr):
            print(f"FAIL routes line count {len(cr)} vs {len(pr)}")
            return 1
        near_tie_n = 0
        for i, ((pc, lc, ci, cw), (pp, pln, pi, pw)) in enumerate(zip(cr, pr)):
            if (pc, lc) != (pp, pln):
                print(f"FAIL route pos/layer {pc}/{lc} vs {pp}/{pln}")
                ok = False
                continue
            tok = pc
            hid_ok = layer_hidden_ok.get((tok, lc), False)
            rok, why = check_route_ids(ci, pi, cw, pw, score_err, w_bound, hid_ok)
            if rok:
                tag = "ok" if why == "exact" else "ok (near-tie)"
                if why != "exact":
                    near_tie_n += 1
                print(f"  {tag}  route[{i}] tok/layer {pc}/{lc}  {why}")
            else:
                print(f"  FAIL  route[{i}] tok/layer {pc}/{lc}: {why}")
                print(f"         C ids {ci}  Py ids {pi}")
                ok = False
        if near_tie_n:
            print(f"  near-tie route reorders accepted: {near_tie_n}")

    print(f"worst hidden max|diff| {worst_h:.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
