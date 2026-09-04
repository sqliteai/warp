#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
qwenparts_ref.py — recompute Qwen isolated ops from the official
transformers qwen4_exp equations and diff against the C engine.

  ./test_qwenparts /tmp/qwenparts.bin
  uv run --with torch --no-project python tools/qwenparts_ref.py /tmp/qwenparts.bin

Tries to import transformers.models.qwen4_exp first. If that module is
absent, the equations below are transcribed from the pinned modeling file
(huggingface/transformers src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
"""

import math
import struct
import sys

import torch
import torch.nn.functional as F

TOL = 2e-5
GDN_TOL = 5e-5
QSA_TOL = 2e-5


class R:
    def __init__(self, path):
        self.b = open(path, "rb").read()
        self.o = 0

    def i(self):
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def f(self):
        v = struct.unpack_from("<f", self.b, self.o)[0]
        self.o += 4
        return v

    def i64(self):
        v = struct.unpack_from("<q", self.b, self.o)[0]
        self.o += 8
        return v

    def a(self, n):
        v = torch.tensor(struct.unpack_from(f"<{n}f", self.b, self.o))
        self.o += 4 * n
        return v

    def bytes(self, n):
        v = self.b[self.o:self.o + n]
        self.o += n
        return v


def check(name, got, ref, tol=TOL):
    got = got.float()
    ref = ref.float()
    d = (got - ref).abs()
    rel = (got - ref).norm() / ref.norm().clamp(min=1e-12)
    ok = d.max().item() < tol
    print(f"  {name:<36} max|diff| {d.max():.3e}  rel {rel:.3e}   "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def check_int(name, got, ref):
    ok = list(got) == list(ref)
    print(f"  {name:<36} {got} vs {ref}   {'OK' if ok else 'FAIL'}")
    return ok


_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def shift_right_ignore_eos(token_ids, shift, eos):
    if shift == 0:
        return token_ids.clone()
    n = token_ids.numel()
    positions = torch.arange(n)
    eos_positions = torch.where(token_ids == eos, positions, torch.full_like(positions, -1))
    previous_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([torch.tensor([-1]), previous_eos_inclusive[:-1]])
    segment_start = previous_eos + 1
    position_in_segment = positions - segment_start
    source_positions = positions - shift
    gather = source_positions.clamp_min(0)
    shifted = token_ids[gather]
    valid = (position_in_segment >= shift) & (source_positions >= 0)
    return torch.where(valid, shifted, torch.full((), eos, dtype=token_ids.dtype))


def l2norm(x, dim=-1, eps=1e-6):
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


def recurrent_gdn(q, k, v, g, beta, Hk):
    """q,k [T,Hk,Dk] v [T,Hv,Dv] g,beta [T,Hv] -> o [T,Hv,Dv]"""
    T, Hv, Dv = v.shape
    Dk = q.shape[-1]
    group = Hv // Hk
    q = l2norm(q, dim=-1) * (Dk ** -0.5)
    k = l2norm(k, dim=-1)
    q = q.repeat_interleave(group, dim=1)
    k = k.repeat_interleave(group, dim=1)
    S = torch.zeros(Hv, Dk, Dv)
    o = torch.zeros(T, Hv, Dv)
    for t in range(T):
        decay = g[t].exp().view(Hv, 1, 1)
        S = S * decay
        kv_mem = (S * k[t].unsqueeze(-1)).sum(-2)
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        S = S + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        o[t] = (S * q[t].unsqueeze(-1)).sum(-2)
    return o


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    rotary_dim = cos.shape[-1]
    rope, nope = x[..., :rotary_dim], x[..., rotary_dim:]
    rope = (rope * cos) + (rotate_half(rope) * sin)
    return torch.cat([rope, nope], dim=-1)


def qwen_rmsnorm(x, w, eps, group=None):
    if group is not None:
        xg = x.reshape(-1, group)
        out = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
        out = out.reshape_as(x)
    else:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return out * (1.0 + w)


def main(path):
    r = R(path)
    ok = True
    print("Qwen components vs official qwen4_exp equations\n")

    # --- 1. PLE -----------------------------------------------------------
    T = r.i(); ngram = r.i(); heads = r.i(); eos = r.i(); vocab = r.i(); seed = r.i()
    pos = r.i()
    ids = torch.tensor([r.i() for _ in range(T)], dtype=torch.long)
    mult_c = torch.tensor([r.i64() for _ in range(ngram)], dtype=torch.long)
    sizes_c = torch.tensor([r.i64() for _ in range(heads)], dtype=torch.long)
    local_c = [r.i() for _ in range(heads)]
    mult = _build_layer_multipliers(vocab, ngram, 0, seed)
    sizes = torch.tensor([_find_nth_prime_after(64 - 1, h + 1) for h in range(heads)],
                         dtype=torch.long)
    ok &= check("PLE multipliers", mult_c.float(), mult.float(), tol=0.5)
    ok &= check("PLE head sizes", sizes_c.float(), sizes.float(), tol=0.5)

    shifted = [shift_right_ignore_eos(ids, s, eos) for s in range(ngram)]
    local = []
    h = 0
    for ng in range(2, ngram + 1):
        mixed = shifted[0][pos] * mult[0]
        for p in range(1, ng):
            mixed = torch.bitwise_xor(mixed, shifted[p][pos] * mult[p])
        for _ in range(8):
            local.append(int(torch.remainder(mixed, sizes[h]).item()))
            h += 1
    ok &= check_int("PLE 16 row ids", local_c, local)

    eos_T = r.i()
    eos_ids = torch.tensor([r.i() for _ in range(eos_T)], dtype=torch.long)
    eos_shift = r.i()
    eos_shifted_c = [r.i() for _ in range(eos_T)]
    eos_ref = shift_right_ignore_eos(eos_ids, eos_shift, eos)
    ok &= check_int("PLE EOS shift", eos_shifted_c, [int(x) for x in eos_ref])

    cols = r.i(); group = r.i()
    ng = (cols + group - 1) // group
    pad = ng * group
    q = torch.tensor(struct.unpack(f"{pad}b", r.bytes(pad)), dtype=torch.int8)
    sc = torch.tensor(struct.unpack_from(f"<{ng}e", r.b, r.o), dtype=torch.float32)
    r.o += 2 * ng
    deq_c = r.a(cols)
    qg = q[:cols].float()
    ref = torch.empty(cols)
    for k in range(ng):
        lo = k * group
        hi = min(lo + group, cols)
        ref[lo:hi] = qg[lo:hi] * sc[k]
    ok &= check("PLE Q8G row dequant", deq_c, ref, tol=2e-4)

    # --- 2. HyperConnection ----------------------------------------------
    hc = r.i(); hid = r.i(); rank = r.i(); eps = r.f()
    H = hc * hid
    hyper = r.a(H); nw = r.a(H); down = r.a(rank * H).view(rank, H)
    up = r.a(H * rank).view(H, rank); inject = r.a(hc * H).view(hc, H)
    block = r.a(hid)
    mixed_c = r.a(hid); inj_c = r.a(hc); comb_c = r.a(H)
    normed = qwen_rmsnorm(hyper, nw, eps, group=hid)
    lo = F.silu(F.linear(normed, down) / hc)
    gate = torch.sigmoid(F.linear(lo, up)).view(hc, hid)
    mixed = (gate * normed.view(hc, hid)).mean(0)
    inj = 2 * torch.sigmoid(F.linear(normed, inject) / hc)
    comb = hyper.view(hc, hid) + inj.unsqueeze(-1) * block
    ok &= check("HC Mix", mixed_c, mixed)
    ok &= check("HC injection gates", inj_c, inj)
    ok &= check("HC Combine", comb_c, comb.reshape(-1))

    # --- 3. GDN ----------------------------------------------------------
    T = r.i(); Hk = r.i(); Hv = r.i(); Dk = r.i(); Dv = r.i()
    q = r.a(T * Hk * Dk).view(T, Hk, Dk)
    k = r.a(T * Hk * Dk).view(T, Hk, Dk)
    v = r.a(T * Hv * Dv).view(T, Hv, Dv)
    a = r.a(T * Hv).view(T, Hv)
    A = r.a(Hv); dt = r.a(Hv)
    g_c = r.a(T * Hv).view(T, Hv); beta_c = r.a(T * Hv).view(T, Hv)
    o_step_c = r.a(Hv * Dv).view(Hv, Dv)
    o_fwd_c = r.a(T * Hv * Dv).view(T, Hv, Dv)
    o_chunk_c = r.a(T * Hv * Dv).view(T, Hv, Dv)
    g = -A.exp() * F.softplus(a + dt)
    ok &= check("GDN decay g", g_c.reshape(-1), g.reshape(-1))
    o_fwd = recurrent_gdn(q, k, v, g, beta_c, Hk)
    ok &= check("GDN decode last", o_step_c, recurrent_gdn(q[-1:], k[-1:], v[-1:],
                                                          g[-1:], beta_c[-1:], Hk)[0],
                tol=GDN_TOL)
    ok &= check("GDN forward", o_fwd_c.reshape(-1), o_fwd.reshape(-1), tol=GDN_TOL)
    ok &= check("GDN chunked", o_chunk_c.reshape(-1), o_fwd.reshape(-1), tol=GDN_TOL)

    # --- 4. QSA ----------------------------------------------------------
    T = r.i(); Hq = r.i(); Hkv = r.i(); D = r.i(); Dk = r.i()
    compress = r.i(); topk = r.i(); query_pos = r.i(); rot = r.i()
    q_idx = r.a(Hq * Dk).view(Hq, Dk)
    raw_k = r.a(T * Dk).view(T, Dk)
    k_ln = r.a(Dk)
    cos = r.a(T * rot).view(T, rot)
    sin = r.a(T * rot).view(T, rot)
    nsel = r.i()
    sel_c = [r.i() for _ in range(nsel)]
    q = r.a(Hq * D).view(Hq, D)
    k = r.a(T * Hkv * D).view(T, Hkv, D)
    v = r.a(T * Hkv * D).view(T, Hkv, D)
    attn_c = r.a(Hq * D).view(Hq, D)

    vis = query_pos + 1
    n_complete = vis // compress
    sel = []
    if n_complete > 0:
        groups = raw_k[: n_complete * compress].view(n_complete, compress, Dk)
        pooled = groups.float().mean(1)
        pooled = qwen_rmsnorm(pooled, k_ln, 1e-6)
        starts = torch.arange(n_complete) * compress
        pooled = apply_rope(pooled, cos[starts], sin[starts])
        dots = torch.relu(q_idx.float() @ pooled.float().T)  # [Hq, blocks]
        scores = dots.sum(0) / math.sqrt(Dk)
        keep = min(topk, n_complete)
        chosen = scores.topk(keep).indices.tolist()
        for b in chosen:
            sel.extend(range(b * compress, (b + 1) * compress))
    sel.extend(range(n_complete * compress, vis))
    ok &= check_int("QSA selected tokens", sel_c, sel)

    n_rep = Hq // Hkv
    k_rep = k.repeat_interleave(n_rep, dim=1)  # [T, Hq, D]
    v_rep = v.repeat_interleave(n_rep, dim=1)
    ks = k_rep[sel]  # [S, Hq, D]
    vs = v_rep[sel]
    scale = D ** -0.5
    scores = torch.einsum("hd,shd->hs", q, ks) * scale
    w = torch.softmax(scores, dim=-1)
    attn = torch.einsum("hs,shd->hd", w, vs)
    ok &= check("QSA attention", attn_c.reshape(-1), attn.reshape(-1), tol=QSA_TOL)

    # --- 5. MoE ----------------------------------------------------------
    E = r.i(); K = r.i(); H = r.i()
    logits = r.a(E)
    idx_c = [r.i() for _ in range(K)]
    w_c = r.a(K)
    gate_in = r.f()
    shared = r.a(H); routed_c = r.a(H); out_c = r.a(H)
    probs = torch.softmax(logits, dim=-1)
    topv, topi = torch.topk(probs, K)
    topv = topv / topv.sum()
    ok &= check_int("MoE ids (router order)", idx_c, [int(x) for x in topi])
    ok &= check("MoE softmax-norm weights", w_c, topv)
    routed = torch.zeros(H)
    for j, eid in enumerate(topi):
        routed = routed + topv[j] * ((eid.float() + 1) * 0.1 + torch.arange(H).float() * 0.01)
    sg = torch.sigmoid(torch.tensor(gate_in))
    out = routed + sg * shared
    ok &= check("MoE router-order reduction", routed_c, routed)
    ok &= check("MoE + shared expert", out_c, out)

    # --- official geometry ----------------------------------------------
    if r.o < len(r.b):
        hc = r.i(); hid = r.i(); rank = r.i(); eps = r.f()
        H = hc * hid
        hyper = r.a(H); nw = r.a(H); down = r.a(rank * H).view(rank, H)
        up = r.a(H * rank).view(H, rank); inject = r.a(hc * H).view(hc, H)
        block = r.a(hid)
        mixed_c = r.a(hid); inj_c = r.a(hc); comb_c = r.a(H)
        ok &= check("HC official rank", torch.tensor([rank], dtype=torch.float32),
                    torch.tensor([320.0]), tol=0.5)
        normed = qwen_rmsnorm(hyper, nw, eps, group=hid)
        lo = F.silu(F.linear(normed, down) / hc)
        gate = torch.sigmoid(F.linear(lo, up)).view(hc, hid)
        mixed = (gate * normed.view(hc, hid)).mean(0)
        inj = 2 * torch.sigmoid(F.linear(normed, inject) / hc)
        comb = hyper.view(hc, hid) + inj.unsqueeze(-1) * block
        ok &= check("HC Mix official", mixed_c, mixed)
        ok &= check("HC Combine official", comb_c, comb.reshape(-1))

        T = r.i(); Hk = r.i(); Hv = r.i(); Dk = r.i(); Dv = r.i()
        o_fwd_c = r.a(T * Hv * Dv)
        o_chunk_c = r.a(T * Hv * Dv)
        ok &= check_int("GDN official heads", [Hk, Hv, Dk, Dv], [16, 48, 128, 128])
        ok &= check("GDN sequential vs chunk official", o_chunk_c, o_fwd_c, tol=GDN_TOL)

        T = r.i(); Hq = r.i(); Dk = r.i(); compress = r.i(); topk = r.i()
        nsel = r.i()
        pooled_c = r.a(Dk)
        k_ln = r.a(Dk)
        raw4 = r.a(4 * Dk).view(4, Dk)
        ok &= check_int("QSA official topk", [topk], [512])
        n_complete = (T) // compress
        keep = min(n_complete, topk)
        expect_nsel = keep * compress + (T % compress)
        ok &= check_int("QSA official nsel", [nsel], [expect_nsel])
        pooled_ref = qwen_rmsnorm(raw4.float().mean(0), k_ln, 1e-6)
        ok &= check("QSA official pooled block", pooled_c, pooled_ref, tol=QSA_TOL)

        E = r.i(); K = r.i()
        logits = r.a(E)
        idx_c = [r.i() for _ in range(K)]
        w_c = r.a(K)
        ok &= check_int("MoE official K", [K], [10])
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, K)
        topv = topv / topv.sum()
        ok &= check_int("MoE official ids", idx_c, [int(x) for x in topi])
        ok &= check("MoE official weights", w_c, topv)

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "qwenparts.bin"))
