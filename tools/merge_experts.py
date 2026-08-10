#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
merge_experts.py — build a "-mini" container: every MoE layer's E routed
experts collapsed into k, and all k run on every token.

    Wbar_c = sum_e alpha[c][e] * W_e          sum_e alpha[c][e] = 1

with each cluster's share of the layer's routed weight folded into its own
down projection, so no router is needed and none is used.

The point of the exercise is what it does to the *system*, not to the
weights: at k = 1 the expert bank is a few hundred MB instead of a
terabyte, nothing streams and the cache is irrelevant; at k = 16 the bank
is exactly one token's working set, so it is resident and the hit rate is
100% by construction. Whether the model still says anything is a separate
question, and this tool is written so both can be measured rather than
argued about.

Choosing the alphas is the interesting part, and there are four:

  uniform   1/E. The null hypothesis.
  trace     alpha_e proportional to the total routed weight expert e
            received in a WASTE_DUMP_ROUTE trace, i.e. an estimate of
            E[w_e * 1(e routed)] — the expert's expected contribution to
            the layer's output. This is the first-order-correct choice
            if you have to pick one vector: it is the coefficient the
            expert already has, averaged over the input distribution.
  top1      all mass on the single most-used expert. Not a merge; the
            control that says how much of any merge's quality is just
            "the busiest expert alone".
  bias      from the trained e_score_correction_bias, which is the
            load-balancing prior and needs no trace. Cheap on K3, where
            a trace long enough to estimate 896 alphas is expensive.

Measured: **it does not matter which one you pick — and not because they
are the same vector.** On Kimi-Linear `trace` and `uniform` land at KL
2.668 and 2.687 nats from the unmerged model, a 0.7% difference on a
2.7-nat catastrophe, while being genuinely different vectors: the measured
usage is concentrated (perplexity 124 of 256, busiest expert 15x uniform,
top-16 holding 34.5% of the routed weight) and `uniform` is flat by
definition. `bias` looks informative and is not — the trained prior is flat
to three digits (perplexity 895.8 of 896 on K3, nothing above 1.08x
uniform), which is what a load-balancing term is trained to produce, so it
is `uniform` wearing a different name. `top1` is the only one that moves
the number, and it moves it the wrong way (3.710).

So a correctly measured, substantially concentrated weighting buys nothing
over 1/E. That is the useful form of the result, because it rules out "you
used the wrong alphas". docs/LEARNED.md §53 and §54 have the reason:
distinct experts are mutually orthogonal, so no convex combination of them
resembles any of them, and what a merge destroys is the per-token
selection rather than the weights.

The heavy pass is tools/merge_layer.c; this decides the alphas, drives it
one layer at a time and writes the container around it.
"""

import argparse
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(HERE, "merge_layer")

# Files that travel with a container unchanged. codebooks.bin is in the list
# on purpose: merge_layer re-encodes the merged matrices against the source
# layer's own codebooks, so every codebook_id in the new records still means
# what it meant, and cb_base needs no renumbering.
CARRY = ("trunk.bin", "codebooks.bin", "tokenizer.model", "specials.json",
         "chat.json", "vision.json")


def moe_layers(man):
    return sorted(int(k) for k in man["layers"])


def shapes(man):
    """(M, N) for gate, up, down, exactly as the engine computes them."""
    c = man["config"]
    hid = c["hidden_size"]
    lat = c.get("routed_expert_hidden_size") or hid
    inter = c["moe_intermediate_size"]
    return [(inter, lat), (inter, lat), (lat, inter)]


# ------------------------------------------------------------- alphas ----

def alphas_uniform(E, layers):
    return {L: [1.0 / E] * E for L in layers}


def alphas_from_trace(path, E, layers, top_k):
    """Sum the routed weight each expert received, per layer.

    The trace line is `pos L id0..idK-1 w0..wK-1 look0..lookK-1`; the ids
    are in selection order and the weights are the renormalized ones the
    engine actually multiplied by, which is what makes the sum an estimate
    of the expert's expected contribution rather than of its hit count.
    """
    acc = {L: [0.0] * E for L in layers}
    seen = 0
    for ln in open(path):
        f = ln.split()
        if len(f) < 2 + 2 * top_k:
            continue
        L = int(f[1])
        if L not in acc:
            continue
        ids = [int(x) for x in f[2:2 + top_k]]
        ws = [float(x) for x in f[2 + top_k:2 + 2 * top_k]]
        for e, w in zip(ids, ws):
            if 0 <= e < E:
                acc[L][e] += w
        seen += 1
    if not seen:
        sys.exit(f"{path}: no usable trace lines (top_k={top_k}?)")
    out = {}
    for L in layers:
        s = sum(acc[L])
        # A layer the trace never reached would divide by zero, and silently
        # falling back to uniform there would mix two schemes inside one
        # container without saying so.
        if s <= 0:
            sys.exit(f"trace covers no routing for layer {L}")
        out[L] = [a / s for a in acc[L]]
    return out, seen


def alphas_top1(per_layer):
    out = {}
    for L, a in per_layer.items():
        best = max(range(len(a)), key=lambda e: a[e])
        v = [0.0] * len(a)
        v[best] = 1.0
        out[L] = v
    return out


def alphas_from_bias(model, man, E, layers):
    """sigmoid-free: the trained load-balancing bias, softmaxed per layer.

    e_score_correction_bias is what the router adds in probability space to
    decide *who* is selected, so it is the model's own statement about how
    often each expert should be picked — available without running anything.
    """
    import math
    tr = {t["name"]: t for t in man["trunk"]}
    pre = man.get("tensor_prefix", "")
    out = {}
    with open(os.path.join(model, "trunk.bin"), "rb") as f:
        for L in layers:
            name = (f"{pre}model.layers.{L}.block_sparse_moe.gate."
                    "e_score_correction_bias")
            t = tr.get(name)
            if t is None or t.get("fmt") != 0:
                sys.exit(f"{name}: absent or not f32; --weights bias needs it")
            f.seek(t["off"])
            b = struct.unpack(f"<{E}f", f.read(4 * E))
            m = max(b)
            ex = [math.exp(x - m) for x in b]
            s = sum(ex)
            out[L] = [x / s for x in ex]
    return out


# ---------------------------------------------------------- clustering ----

def cluster_groups(order, k, how, seed=0):
    """Partition expert ids into k groups. `order` is hottest-first.

    roundrobin  deal the usage ranking around the table, so every cluster
                gets one of the hottest experts, one of the next tier, and
                so on. Balanced in total routed weight by construction,
                which is what keeps the per-cluster gains near 1/k.
    random      the control. If it ties roundrobin, the ranking carried no
                information worth clustering on.
    prune       not a merge at all: keep the k hottest experts, each alone
                in its own cluster, and drop the other E-k. The baseline
                any merge has to beat, because it is the one policy that
                leaves real experts intact.
    """
    if how == "prune":
        return [[e] for e in order[:k]]
    if how == "random":
        ids = list(order)
        random.Random(seed).shuffle(ids)
        return [ids[i::k] for i in range(k)]
    return [order[i::k] for i in range(k)]


def layer_alphas(weights, k, how, seed=0):
    """(alpha[k][E], gain[k]) from a per-expert weight vector summing to 1.

    Each alpha row is normalized within its cluster so the merged expert is
    a proper mean — the SiLU downstream has to see inputs of the right
    magnitude. The cluster's share of the layer's routed weight comes back
    separately as a gain, which merge_layer folds into the down projection.
    """
    E = len(weights)
    order = sorted(range(E), key=lambda e: -weights[e])
    groups = cluster_groups(order, k, how, seed)
    A = [[0.0] * E for _ in range(k)]
    gain = []
    for c, members in enumerate(groups):
        tot = sum(weights[e] for e in members)
        gain.append(tot)
        if tot > 0:
            for e in members:
                A[c][e] = weights[e] / tot
        else:                       # a cluster the trace never routed to
            for e in members:
                A[c][e] = 1.0 / len(members)
    s = sum(gain)
    gain = [g / s for g in gain] if s > 0 else [1.0 / k] * k
    return A, gain


# -------------------------------------------------------------- driver ----

def merge_one(model, out, man, L, alpha, gain, args, tmp):
    sh = shapes(man)
    ent = man["layers"][str(L)]
    afile = os.path.join(tmp, f"alpha-L{L}.f32")
    flat = [x for row in alpha for x in row]
    with open(afile, "wb") as f:
        f.write(struct.pack(f"<{len(flat)}f", *flat))
    gfile = os.path.join(tmp, f"gain-L{L}.f32")
    with open(gfile, "wb") as f:
        f.write(struct.pack(f"<{len(gain)}f", *gain))
    jfile = os.path.join(tmp, f"job-L{L}.txt")
    eq = man["expert_quant"]
    with open(jfile, "w") as f:
        f.write(f"bank={os.path.join(model, ent['file'])}\n")
        f.write(f"n_experts={ent['experts']}\n")
        f.write(f"codebooks={os.path.join(model, 'codebooks.bin')}\n")
        f.write(f"cb_base={ent['codebook_base']}\n")
        f.write(f"stages={eq['stages']}\nentries={eq['entries']}\n")
        f.write(f"vec_dim={eq['vec_dim']}\nlayer={L}\n")
        f.write(f"threads={args.threads}\ncheck={1 if args.check else 0}\n")
        for i, (m, n) in enumerate(sh):
            f.write(f"m{i}={m}\nn{i}={n}\n")
        f.write(f"alpha={afile}\n")
        f.write(f"gains={gfile}\n")
        f.write(f"clusters={len(alpha)}\n")
        f.write(f"out={os.path.join(out, ent['file'])}\n")
    r = subprocess.run([WORKER, jfile], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"L{L}: {r.stderr.strip() or 'merge_layer failed'}")
    if r.stderr.strip():
        print("   ", r.stderr.strip().replace("\n", "\n    "))
    return os.path.getsize(os.path.join(out, ent["file"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--weights", default="trace",
                    choices=("uniform", "trace", "top1", "bias"))
    ap.add_argument("--trace", help="a WASTE_DUMP_ROUTE dump")
    ap.add_argument("--clusters", type=int, default=1,
                    help="records per layer; 1 collapses the layer to one "
                         "expert, k>1 merges within k groups and runs all k "
                         "on every token")
    ap.add_argument("--cluster-by", default="roundrobin",
                    choices=("roundrobin", "random", "prune"),
                    help="how the experts are partitioned when --clusters > 1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", help="subset, e.g. 1-4 (for quick checks)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="report the re-quantization error per matrix")
    args = ap.parse_args()

    if not os.path.exists(WORKER):
        sys.exit(f"{WORKER}: build it with\n"
                 f"  cc -O2 -std=gnu11 -o merge_layer tools/merge_layer.c "
                 f"libwaste.a -lm -lpthread")

    man = json.load(open(os.path.join(args.model, "manifest.json")))
    cfg = man["config"]
    E = cfg["num_experts"]
    K = cfg["num_experts_per_token"]
    if E < 2:
        sys.exit(f"{args.model} already has {E} expert(s) per layer")
    layers = moe_layers(man)
    if args.layers:
        a, _, b = args.layers.partition("-")
        lo, hi = int(a), int(b or a)
        layers = [L for L in layers if lo <= L <= hi]

    if args.weights == "uniform":
        alpha = alphas_uniform(E, layers)
        how = "uniform 1/E"
    elif args.weights == "bias":
        alpha = alphas_from_bias(args.model, man, E, layers)
        how = "softmax(e_score_correction_bias)"
    else:
        if not args.trace:
            sys.exit("--weights trace/top1 needs --trace")
        alpha, n = alphas_from_trace(args.trace, E, layers, K)
        how = f"routed weight over {n} trace lines"
        if args.weights == "top1":
            alpha = alphas_top1(alpha)
            how = f"top-1 expert by {how}"

    if args.clusters > 1 and args.weights == "top1":
        sys.exit("--weights top1 leaves one nonzero weight, so it cannot fill "
                 f"{args.clusters} clusters. Use --cluster-by prune instead.")
    if args.clusters > 64:
        sys.exit("--clusters > 64: top_k is capped at 64 by the engine")

    # Per-expert weights become a partition plus one gain per cluster. k = 1
    # is the same code path with a single group, so the k = 1 containers this
    # tool built before clustering existed still come out byte-identical.
    alpha = {L: layer_alphas(alpha[L], args.clusters, args.cluster_by, args.seed)
             for L in layers}
    how += f", {args.clusters} cluster(s) by {args.cluster_by}"

    os.makedirs(args.out, exist_ok=True)
    for name in CARRY:
        src = os.path.join(args.model, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out, name))

    print(f"{args.model} -> {args.out}")
    print(f"  {len(layers)} MoE layers, {E} experts -> {args.clusters}, "
          f"alphas: {how}")
    t0 = time.time()
    total = 0
    tmp = tempfile.mkdtemp(prefix="wmerge-")
    try:
        for i, L in enumerate(layers):
            t1 = time.time()
            n = merge_one(args.model, args.out, man, L,
                          alpha[L][0], alpha[L][1], args, tmp)
            total += n
            print(f"  [{i + 1}/{len(layers)}] L{L}: {n / 1e6:.1f} MB "
                  f"({time.time() - t1:.1f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # The manifest the engine will read. num_experts_per_token is 1 and
    # num_experts is 1, which is also the flag moe_layer keys the
    # no-router path off: a single expert has nothing to choose between,
    # and with K=1 the engine's renormalize branch does not fire, so the
    # weight has to come from routed_scaling_factor rather than from a
    # sigmoid nobody renormalized.
    out_man = json.loads(json.dumps(man))
    out_man["config"]["num_experts"] = args.clusters
    out_man["config"]["num_experts_per_token"] = args.clusters
    # Every expert in the bank runs on every token. The engine keys the
    # no-router path off this flag rather than off num_experts == 1, because
    # at k > 1 the trunk's [E, hidden] router still has the wrong number of
    # rows and there is still nothing to select between.
    out_man["config"]["moe_no_router"] = True
    out_man["layers"] = {
        str(L): dict(man["layers"][str(L)],
                     experts=args.clusters,
                     bytes=os.path.getsize(
                         os.path.join(args.out, man["layers"][str(L)]["file"])))
        for L in layers}
    out_man["merged_from"] = {"model": os.path.abspath(args.model),
                              "experts": E, "top_k": K, "weights": args.weights,
                              "clusters": args.clusters,
                              "cluster_by": args.cluster_by}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(out_man, f, indent=1)

    trunk = os.path.getsize(os.path.join(args.out, "trunk.bin"))
    print(f"  experts {total / 1e9:.3f} GB + trunk {trunk / 1e9:.3f} GB "
          f"= {(total + trunk) / 1e9:.3f} GB in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
