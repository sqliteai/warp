#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""qwen_container_ref.py — container-native Qwen oracle.

Reads a format-v0 WEXP container the same way kimi_ref.py does, then runs
the Qwen forward in src/model.c qwen_step: embed, hyper-state, PLE, GDN or
QSA, MoE with a shared expert. The C engine can dump the same tensors.

  uv run --with torch --no-project python tools/qwen_container_ref.py \
      --container model.waste --ids 1,2,3 \
      --dump logits.bin --hidden hidden.bin --routes routes.txt
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import torch
import torch.nn.functional as F

_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from kimi_ref import (  # noqa: E402
    HDR,
    HDR_SIZE,
    KINDS,
    VEC_DIM,
    Container,
    _LazyTrunk,
)
from qwenparts_ref import (  # noqa: E402
    _build_layer_multipliers,
    apply_rope,
    l2norm,
    qwen_rmsnorm,
    shift_right_ignore_eos,
)


class QwenContainer(Container):
    """Same record path as Kimi; Qwen experts are inter x hidden, not latent."""

    def __init__(self, path, device="cpu", c_matvec=None):
        super().__init__(path, device)
        # Keep few trunk tensors resident — the checkpoint trunk is 80+ GiB.
        self.t = _LazyTrunk(self, cap=8)
        if c_matvec is None:
            c_matvec = os.environ.get("WASTE_Q8", "1") == "0"
        self.c_matvec = c_matvec

    def expert_shapes(self):
        h = self.cfg["hidden_size"]
        i = self.cfg["moe_intermediate_size"]
        return [(i, h), (i, h), (h, i)]

    def expert(self, L, eid):
        """Dequantize one expert. Index blocks are padded to index_block rows.

        kimi_ref.Container.expert reads nvec*stages bytes. That is right when
        M is a multiple of index_block (real Qwen: 640). The synthetic fixture
        has M=16, and the record stores a full padded tile.
        """
        f, meta = self.banks[L]
        rec_bytes = meta["bytes"] // meta["experts"]
        f.seek(eid * rec_bytes)
        buf = f.read(rec_bytes)
        h = struct.unpack(HDR, buf[:HDR_SIZE])
        cb_base, g_off, u_off, d_off, corr_off = h[5], h[9], h[10], h[11], h[12]
        assert h[0] == 0x50584557 and h[2] == eid
        shapes = self.expert_shapes()
        out, sc_cur = {}, corr_off
        for i, kind in enumerate(KINDS):
            M, N = shapes[i]
            nvec = M * N // VEC_DIM
            beg = (g_off, u_off, d_off)[i]
            if self.iblock:
                B, nvr = self.iblock, N // VEC_DIM
                nb = (M + B - 1) // B
                nbytes = nb * nvr * B * self.stages
                raw = torch.frombuffer(bytearray(buf[beg:beg + nbytes]),
                                       dtype=torch.uint8)
                idx = (raw.view(nb, nvr, B, self.stages).permute(0, 2, 1, 3)
                          .reshape(nb * B, nvr, self.stages)[:M]
                          .reshape(nvec, self.stages).long())
            else:
                raw = torch.frombuffer(
                    bytearray(buf[beg:beg + nvec * self.stages]),
                    dtype=torch.uint8)
                idx = raw.view(nvec, self.stages).long()
            recon = torch.zeros(nvec, VEC_DIM)
            for s in range(self.stages):
                recon += self.books[cb_base + i * self.stages + s].cpu()[idx[:, s]]
            sc = torch.frombuffer(bytearray(buf[sc_cur:sc_cur + M * 2]),
                                  dtype=torch.float16).float().view(M, 1)
            sc_cur += M * 2
            out[kind] = (recon.view(M, N) * sc).to(self.dev)
        return out


def _t(owner, name):
    return owner.t[owner.p + name]


def _linear(owner, name, x):
    """F.linear; quantized trunk weights always use matvec (matches C)."""
    full = owner.p + name
    meta = owner.c._meta.get(full)
    if meta is None:
        return F.linear(x, _t(owner, full))
    mv = owner.c.matvec_c if owner.c.c_matvec else owner.c.matvec
    if meta["fmt"] != 0:
        return mv(full, x)
    if owner.c.c_matvec and len(meta["shape"]) == 2:
        return mv(full, x)
    return F.linear(x, _t(owner, full))


def short_conv_step(x, w, state):
    """One causal depthwise conv + SiLU. Matches waste_short_conv_step."""
    w2 = w.reshape(w.shape[0], -1)
    k = w2.shape[-1]
    r = k - 1
    acc = x * w2[:, r]
    if r > 0:
        acc = acc + (state * w2[:, :r]).sum(-1)
    y = acc * torch.sigmoid(acc)
    if r <= 0:
        return y, state
    if r == 1:
        return y, x.unsqueeze(-1)
    return y, torch.cat([state[:, 1:], x.unsqueeze(-1)], -1)


def dilated_conv_step(x, w, ring, dil):
    """Dilated causal conv + SiLU. Matches qwen_dilated_conv_step."""
    w2 = w.reshape(w.shape[0], -1)
    ks = w2.shape[-1]
    r = (ks - 1) * dil
    acc = x * w2[:, ks - 1]
    for k in range(ks - 1):
        acc = acc + ring[:, k * dil] * w2[:, k]
    y = acc * torch.sigmoid(acc)
    if r <= 0:
        return y, ring
    if r == 1:
        return y, x.unsqueeze(-1)
    return y, torch.cat([ring[:, 1:], x.unsqueeze(-1)], -1)


def gdn_step(q, k, v, g, beta, hk, s):
    """One token of qwenparts_ref.recurrent_gdn; s [Hv,Dk,Dv] is updated."""
    hv, dv = v.shape
    dk = q.shape[-1]
    group = hv // hk
    qn = l2norm(q, dim=-1) * (dk ** -0.5)
    kn = l2norm(k, dim=-1)
    qn = qn.repeat_interleave(group, dim=0)
    kn = kn.repeat_interleave(group, dim=0)
    decay = g.exp().view(hv, 1, 1)
    s.mul_(decay)
    kv_mem = (s * kn.unsqueeze(-1)).sum(-2)
    delta = (v - kv_mem) * beta.unsqueeze(-1)
    s.add_(kn.unsqueeze(-1) * delta.unsqueeze(-2))
    return (s * qn.unsqueeze(-1)).sum(-2)


def _c_l2_rnorm(vals):
    s = sum(v * v for v in vals)
    return 1.0 / math.sqrt(s + 1e-6)


def _c_gdn_decay(a, a_log, dt):
    out = []
    for h in range(len(a)):
        z = float(a[h]) + float(dt[h])
        sp = math.log(1.0 + math.exp(-abs(z))) + (z if z > 0.0 else 0.0)
        out.append(-math.exp(float(a_log[h])) * sp)
    return out


def _c_gdn_step(q, k, v, g, beta, hk, s):
    """waste_qwen_gdn_step / gdn_one — in-place on s [Hv,Dk,Dv]."""
    hv, dk, dv = v.shape[0], q.shape[1], v.shape[1]
    group = hv // hk if hk else 1
    qscale = 1.0 / math.sqrt(float(dk))
    out = torch.zeros(hv, dv, dtype=torch.float32)
    for h in range(hv):
        src = h // group
        qh = [float(x) for x in q[src].tolist()]
        kh = [float(x) for x in k[src].tolist()]
        vh = [float(x) for x in v[h].tolist()]
        qn = _c_l2_rnorm(qh) * qscale
        kn = _c_l2_rnorm(kh)
        decay = math.exp(float(g[h]))
        b = float(beta[h])
        u = [0.0] * dv
        for kk in range(dk):
            row = [float(x) for x in s[h, kk].tolist()]
            kv = kh[kk] * kn
            for i in range(dv):
                row[i] *= decay
                u[i] += row[i] * kv
            s[h, kk] = torch.tensor(row, dtype=torch.float32)
        for i in range(dv):
            u[i] = b * (vh[i] - u[i])
        oh = [0.0] * dv
        for kk in range(dk):
            row = [float(x) for x in s[h, kk].tolist()]
            kv = kh[kk] * kn
            qv = qh[kk] * qn
            for i in range(dv):
                row[i] += u[i] * kv
                oh[i] += row[i] * qv
            s[h, kk] = torch.tensor(row, dtype=torch.float32)
        out[h] = torch.tensor(oh, dtype=torch.float32)
    return out


def _c_short_conv_step(x, w, state):
    """waste_short_conv_step — one causal depthwise conv + SiLU."""
    w2 = w.reshape(w.shape[0], -1)
    c, ks = w2.shape
    r = ks - 1
    xv = [float(v) for v in x.tolist()]
    y = []
    for ci in range(c):
        wc = [float(v) for v in w2[ci].tolist()]
        rc = [float(v) for v in state[ci].tolist()] if r > 0 else []
        acc = 0.0
        for j in range(r):
            acc += rc[j] * wc[j]
        acc += xv[ci] * wc[r]
        y.append(acc / (1.0 + math.exp(-acc)))
    if r <= 0:
        return torch.tensor(y, dtype=torch.float32), state
    if r == 1:
        return torch.tensor(y, dtype=torch.float32), x.unsqueeze(-1)
    return torch.tensor(y, dtype=torch.float32), torch.cat([state[:, 1:], x.unsqueeze(-1)], -1)


def _c_rmsnorm_gated(x, gate, weight, eps):
    """waste_rmsnorm_gated per head vector."""
    xv = [float(v) for v in x.tolist()]
    gv = [float(v) for v in gate.tolist()]
    wv = [float(v) for v in weight.tolist()]
    c = len(xv)
    s = sum(v * v for v in xv)
    r = 1.0 / math.sqrt(s / float(c) + eps)
    out = []
    for i in range(c):
        g = 1.0 / (1.0 + math.exp(-gv[i]))
        out.append(xv[i] * r * wv[i] * g)
    return torch.tensor(out, dtype=torch.float32)


def rmsnorm_gated(x, gate, weight, eps):
    """waste_rmsnorm_gated: RMSNorm * weight * sigmoid(gate), not (1+w)."""
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r * weight * torch.sigmoid(gate)


def mrope_interleave(ft, fh, fw, section, half):
    out = ft.clone()
    src = (ft, fh, fw)
    for dim in (1, 2):
        length = int(section[dim]) * 3
        i = dim
        while i < length and i < half:
            out[i] = src[dim][i]
            i += 3
    return out


def rope_cs(pos, inv_freq, rotary_dim, section):
    half = rotary_dim // 2
    a = float(pos) * inv_freq
    freqs = mrope_interleave(a, a, a, section, half)
    c, s = torch.cos(freqs), torch.sin(freqs)
    return torch.cat([c, c]), torch.cat([s, s])


def moe_route(logits, k, renorm, c_style=False):
    """waste_qwen_moe_route: softmax all experts, greedy top-k, optional renorm."""
    E = int(logits.numel())
    if c_style:
        lg = [float(x) for x in logits.detach().float().cpu().view(-1).tolist()]
        m = max(lg) if lg else 0.0
        prob = [math.exp(v - m) for v in lg]
        z = sum(prob)
        if z < 1e-20:
            z = 1e-20
        prob = [p / z for p in prob]
        used = [False] * E
        idx, w = [], []
        for _ in range(k):
            best, bv = -1, -1.0
            for e in range(E):
                if used[e]:
                    continue
                pe = prob[e]
                if pe > bv:
                    bv = pe
                    best = e
            idx.append(best if best >= 0 else 0)
            w.append(bv if best >= 0 else 0.0)
            if best >= 0:
                used[best] = True
        wt = w
        if renorm and k > 1:
            s = sum(wt)
            if s > 1e-20:
                wt = [x / s for x in wt]
        return idx, torch.tensor(wt, dtype=torch.float32)
    lg = logits.detach().float().cpu().view(-1)
    m = float(lg.max()) if E else 0.0
    prob = torch.exp(lg - m)
    z = float(prob.sum())
    if z < 1e-20:
        z = 1e-20
    prob = prob / z
    used = [False] * E
    idx, w = [], []
    for _ in range(k):
        best, bv = -1, -1.0
        for e in range(E):
            if used[e]:
                continue
            pe = float(prob[e])
            if pe > bv:
                bv = pe
                best = e
        idx.append(best if best >= 0 else 0)
        w.append(bv if best >= 0 else 0.0)
        if best >= 0:
            used[best] = True
    wt = torch.tensor(w, dtype=torch.float32)
    if renorm and k > 1:
        s = float(wt.sum())
        if s > 1e-20:
            wt = wt / s
    return idx, wt


def qsa_select(q_heads, raw_k, query_pos, cos, sin, rotary_dim, k_ln, eps,
               compress, block_topk):
    """waste_qwen_qsa_select: pool blocks, ReLU-dot, greedy top blocks, tail."""
    t = raw_k.shape[0]
    dk = raw_k.shape[-1]
    if t < 1:
        return []
    if query_pos < 0:
        query_pos = 0
    if query_pos >= t:
        query_pos = t - 1
    vis = query_pos + 1
    n_complete = vis // compress
    sel = []
    if n_complete > 0:
        scores = []
        pooled = []
        for b in range(n_complete):
            po = raw_k[b * compress:(b + 1) * compress].mean(0)
            po = qwen_rmsnorm(po, k_ln, eps)
            if rotary_dim > 0 and cos is not None:
                po = apply_rope(po, cos[b * compress], sin[b * compress])
            pooled.append(po)
            dots = torch.relu(q_heads.float() @ po.float())
            scores.append(dots.sum() * (dk ** -0.5))
        scores = torch.stack(scores)
        taken = [False] * n_complete
        keep = n_complete if n_complete < block_topk else block_topk
        for _ in range(keep):
            best, bv = -1, -1e30
            for b in range(n_complete):
                if taken[b]:
                    continue
                v = float(scores[b])
                if v > bv:
                    bv, best = v, b
            if best < 0:
                break
            taken[best] = True
            sel.extend(range(best * compress, (best + 1) * compress))
    sel.extend(range(n_complete * compress, vis))
    return sel


def qsa_attn(q, k, v, sel, scale):
    if not sel:
        return torch.zeros_like(q)
    hq, d = q.shape
    hkv = k.shape[1]
    n_rep = hq // hkv if hkv else 1
    idx = torch.tensor(sel, dtype=torch.long)
    ks = k.index_select(0, idx).repeat_interleave(n_rep, dim=1)
    vs = v.index_select(0, idx).repeat_interleave(n_rep, dim=1)
    scores = torch.einsum("hd,shd->hs", q, ks) * scale
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,shd->hd", w, vs)


def _c_rmsnorm(x, w, eps, group):
    """waste_qwen_rmsnorm — scalar fp32 loop."""
    xv = x.detach().float().cpu().view(-1).tolist()
    wv = w.detach().float().cpu().view(-1).tolist()
    n = len(xv)
    ng = n // group
    out = [0.0] * n
    for g in range(ng):
        base = g * group
        s = sum(xv[base + i] * xv[base + i] for i in range(group))
        r = 1.0 / math.sqrt(s / float(group) + eps)
        for i in range(group):
            out[base + i] = xv[base + i] * r * (1.0 + wv[base + i])
    return torch.tensor(out, dtype=torch.float32)


def _c_silu(x):
    return x / (1.0 + math.exp(-x))


def _c_sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def hc_mix(hyper, owner, nw_name, down_name, up_name, inject_name, hc, hid, eps):
    nw = _t(owner, nw_name)
    if getattr(owner.c, "c_matvec", False):
        H = hc * hid
        normed = _c_rmsnorm(hyper, nw, eps, hid)
        lo = _linear(owner, down_name, normed)
        lo = torch.tensor([_c_silu(float(v) / hc) for v in lo.tolist()],
                          dtype=torch.float32)
        gate = _linear(owner, up_name, lo)
        gate = torch.tensor([_c_sigmoid(float(v)) for v in gate.tolist()],
                            dtype=torch.float32)
        nv = normed.tolist()
        gv = gate.tolist()
        mixed = []
        for d in range(hid):
            s = sum(gv[b * hid + d] * nv[b * hid + d] for b in range(hc))
            mixed.append(s / float(hc))
        mixed = torch.tensor(mixed, dtype=torch.float32)
        inj = None
        if inject_name is not None:
            tmp = _linear(owner, inject_name, normed)
            inj = torch.tensor([2.0 * _c_sigmoid(float(v) / hc) for v in tmp.tolist()],
                               dtype=torch.float32)
        return mixed.to(hyper.device), (inj.to(hyper.device) if inj is not None else None)
    normed = qwen_rmsnorm(hyper, nw, eps, group=hid)
    lo = F.silu(_linear(owner, down_name, normed) / hc)
    gate = torch.sigmoid(_linear(owner, up_name, lo)).view(hc, hid)
    mixed = (gate * normed.view(hc, hid)).mean(0)
    inj = None
    if inject_name is not None:
        inj = 2.0 * torch.sigmoid(_linear(owner, inject_name, normed) / hc)
    return mixed, inj


def hc_combine(hyper, block, inj, hc, hid, c_ops=False):
    if inj is None:
        return hyper
    if c_ops:
        h = hyper.detach().float().cpu().view(-1).tolist()
        b = block.detach().float().cpu().view(-1).tolist()
        iw = inj.detach().float().cpu().tolist()
        out = [0.0] * (hc * hid)
        for bi in range(hc):
            for d in range(hid):
                out[bi * hid + d] = h[bi * hid + d] + iw[bi] * b[d]
        return torch.tensor(out, dtype=torch.float32).to(hyper.device)
    return hyper.view(hc, hid) + inj.unsqueeze(-1) * block


def ple_row_ids(ids, pos, eos, ngram, heads_per, mult, sizes):
    shifted = [shift_right_ignore_eos(ids, s, eos) for s in range(ngram)]
    local = []
    h = 0
    for ng in range(2, ngram + 1):
        mixed = shifted[0][pos] * mult[0]
        for p in range(1, ng):
            mixed = torch.bitwise_xor(mixed, shifted[p][pos] * mult[p])
        for _ in range(heads_per):
            sz = int(sizes[h]) if h < sizes.numel() else 0
            if sz <= 0:
                local.append(0)
            else:
                r = int(torch.remainder(mixed, sz).item())
                local.append(r)
            h += 1
    return local


def write_f32(path, vec, mode="wb"):
    v = vec.detach().float().contiguous().flatten()
    with open(path, mode) as f:
        f.write(struct.pack(f"<{v.numel()}f", *v.tolist()))


def _expert_record(c: QwenContainer, L, eid):
    f, meta = c.banks[L]
    rec_bytes = meta["bytes"] // meta["experts"]
    f.seek(eid * rec_bytes)
    buf = f.read(rec_bytes)
    h = struct.unpack(HDR, buf[:HDR_SIZE])
    return buf, h


def _expert_indices(buf, off, M, N, iblock, stages):
    """Index tensor [M, nvr, stages] in the layout vq_rows reads."""
    vd = VEC_DIM
    nv = N // vd
    if iblock:
        B, nvr = iblock, nv
        nb = (M + B - 1) // B
        nbytes = nb * nvr * B * stages
        raw = torch.frombuffer(bytearray(buf[off:off + nbytes]), dtype=torch.uint8)
        return (raw.view(nb, nvr, B, stages).permute(0, 2, 1, 3)
                   .reshape(nb * B, nvr, stages)[:M].long())
    nvec = M * N // vd
    raw = torch.frombuffer(bytearray(buf[off:off + nvec * stages]), dtype=torch.uint8)
    return raw.view(M, nv, stages).long()


def _vq_lut_build(x, books, cb_base, stages, entries, vec_dim):
    """dst[c] = dot(x_v, C[c]) for every codebook entry — waste_lutb_range."""
    x = x.float().cpu()
    nv = x.numel() // vec_dim
    lut = torch.empty(nv * stages * entries, dtype=torch.float32)
    for v in range(nv):
        xv = x[v * vec_dim:(v + 1) * vec_dim]
        for s in range(stages):
            book = books[cb_base + s].cpu()
            base = (v * stages + s) * entries
            lut[base:base + entries] = xv @ book.T
    return lut.view(nv, stages, entries)


def _vq_apply_rows(y, idx, scales, lut, M, stages, iblock=64):
    """One VQ3R matrix-vector product — waste vq_rows, serial path."""
    nv, entries = lut.shape[0], lut.shape[2]
    VQ_TILE = iblock
    y.zero_()
    for r0 in range(0, M, VQ_TILE * 2):
        rows = min(VQ_TILE * 2, M - r0)
        nblk = (rows + VQ_TILE - 1) // VQ_TILE
        acc = torch.zeros(rows, dtype=torch.float32)
        for v in range(nv):
            blk = lut[v]
            for j in range(nblk):
                nr = VQ_TILE if (j + 1) * VQ_TILE <= rows else rows - j * VQ_TILE
                tile = r0 // VQ_TILE + j
                ix = idx[tile, v, :nr, :]
                ac = acc[j * VQ_TILE:j * VQ_TILE + nr]
                for r in range(nr):
                    t = blk[0, ix[r, 0]].item()
                    for s in range(1, stages):
                        t += blk[s, ix[r, s]].item()
                    ac[r] += t
        for r in range(rows):
            y[r0 + r] = acc[r] * scales[r0 + r].item()


def _vq_matvec(x, buf, cb_base, g_off, corr_off, M, N, kind, inter, hid, c):
    """One expert matrix (gate/up/down) times x, matching qwen_moe_layer."""
    stages = c.stages
    idx = _expert_indices(buf, g_off, M, N, c.iblock, stages)
    sc_off = corr_off + (0 if kind == 0 else inter * 2 if kind == 1 else inter * 4)
    sc = torch.frombuffer(bytearray(buf[sc_off:sc_off + M * 2]), dtype=torch.float16).float()
    x = x.float().cpu()
    vd = VEC_DIM
    nv = N // vd
    y = torch.zeros(M, dtype=torch.float32)
    cb = cb_base + kind * stages
    for r in range(M):
        acc = 0.0
        for v in range(nv):
            xv = x[v * vd:(v + 1) * vd]
            t = 0.0
            for s in range(stages):
                ix = int(idx[r, v, s])
                t += float(xv @ c.books[cb + s].cpu()[ix])
            acc += t
        y[r] = acc * sc[r].item()
    return y


def _moe_vq(c: QwenContainer, L, eid, x, inter, hid):
    """Routed expert gate/up/down through the same VQ path as src/model.c."""
    buf, h = _expert_record(c, L, eid)
    cb_base = h[5]
    g_off, u_off, d_off, corr_off = h[9], h[10], h[11], h[12]
    ga = _vq_matvec(x, buf, cb_base, g_off, corr_off, inter, hid, 0, inter, hid, c)
    ub = _vq_matvec(x, buf, cb_base, u_off, corr_off, inter, hid, 1, inter, hid, c)
    hv = F.silu(ga) * ub
    return _vq_matvec(hv, buf, cb_base, d_off, corr_off, hid, inter, 2, inter, hid, c)


class QwenRef:
    def __init__(self, c: QwenContainer):
        self.c, self.t, self.cfg = c, c.t, c.cfg
        self.p = c.prefix
        cfg = self.cfg
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.n_layers = int(cfg["num_hidden_layers"])
        self.hid = int(cfg["hidden_size"])
        self.hc = int(cfg.get("hc_count", 4))
        self.rank = int(cfg.get("hc_lowrank", 0))
        self.vocab = int(cfg["vocab_size"])
        self.eos = int(cfg.get("eos_token_id", 0))
        self.top_k = int(cfg.get("num_experts_per_token")
                         or cfg.get("num_experts_per_tok") or 0)
        self.renorm = True  # Qwen4Exp router always renormalizes (model.c)
        kinds = cfg.get("layer_types") or []
        self.full = [k == "full_attention" for k in kinds]
        while len(self.full) < self.n_layers:
            self.full.append(False)
        ids = cfg.get("ple_layer_ids") or []
        self.ple_layer = (int(ids[0]) - 1) if ids and int(ids[0]) > 0 else -1
        self.ngram = int(cfg.get("ngram_size", 3))
        self.heads_per = int(cfg.get("heads_per_ngram", 8))
        self.ple_embed = int(cfg.get("ple_embed_dim") or self.hid)
        self.ple_ks = int(cfg.get("ple_conv_kernel_size", 4))
        self.hk = int(cfg.get("linear_num_key_heads", 0))
        self.hv = int(cfg.get("linear_num_value_heads", 0))
        self.dk = int(cfg.get("linear_key_head_dim", 0))
        self.dv = int(cfg.get("linear_value_head_dim", 0))
        self.conv_k = int(cfg.get("linear_conv_kernel_dim", 4))
        self.hq = int(cfg.get("num_attention_heads", 0))
        self.hkv = int(cfg.get("num_key_value_heads", 0))
        self.qsa_d = int(cfg.get("head_dim", 0))
        self.idx_n = int(cfg.get("indexer_n_heads", 4))
        self.idx_kv = int(cfg.get("indexer_kv_heads", 1))
        self.idx_d = int(cfg.get("indexer_head_dim", 128))
        self.idx_budget = int(cfg.get("indexer_budget", 2048))
        self.compress = int(cfg.get("indexer_compress_ratio", 4)) or 4
        rp = cfg.get("rope_parameters") or {}
        pf = rp.get("partial_rotary_factor",
                    cfg.get("partial_rotary_factor", 0.25))
        self.rotary_dim = int(self.qsa_d * float(pf))
        self.section = list(rp.get("mrope_section") or [11, 11, 10])
        while len(self.section) < 3:
            self.section.append(0)
        theta = float(rp.get("rope_theta", cfg.get("rope_theta", 1e7)))
        half = self.rotary_dim // 2
        if half > 0:
            i = torch.arange(half, dtype=torch.float32)
            self.inv_freq = 1.0 / (theta ** (2.0 * i / self.rotary_dim))
        else:
            self.inv_freq = torch.zeros(0)
        heads = (self.ngram - 1) * self.heads_per
        mul = cfg.get("ple_layer_multipliers")
        if mul:
            self.ple_mult = torch.tensor(mul, dtype=torch.long)
        else:
            self.ple_mult = _build_layer_multipliers(
                int(cfg.get("vocab_size", 1)), self.ngram, 0,
                int(cfg.get("seed", 0)))
        sz = cfg.get("ple_head_vocab_sizes")
        if sz:
            self.ple_sz = torch.tensor(sz, dtype=torch.long)
        else:
            self.ple_sz = torch.ones(heads, dtype=torch.long)
        self.reset()

    def reset(self):
        qkv = 2 * self.hk * self.dk + self.hv * self.dv
        r = max(self.conv_k - 1, 0)
        self.S, self.conv = {}, {}
        self.qsa_k, self.qsa_v, self.qsa_rawk = {}, {}, {}
        for L in range(self.n_layers):
            if not self.full[L]:
                self.S[L] = torch.zeros(self.hv, self.dk, self.dv)
                self.conv[L] = torch.zeros(qkv, r)
            else:
                self.qsa_k[L] = []
                self.qsa_v[L] = []
                self.qsa_rawk[L] = []
        pr = (self.ple_ks - 1) * self.ngram if self.ple_ks > 1 else 0
        h = self.hc * self.hid
        self.ple_ring = torch.zeros(h, pr if pr > 0 else 1)
        self.ple_prev = [self.eos] * max(self.ngram - 1, 0)
        self.hcx = torch.zeros(h)

    def _ple_inject(self, token):
        L = self.ple_layer
        if L < 0:
            return
        ngram, heads_per = self.ngram, self.heads_per
        ctxn = ngram - 1
        hist = list(self.ple_prev[:ctxn]) + [int(token)]
        ids = torch.tensor(hist, dtype=torch.long)
        pos = ctxn if ctxn >= 0 else 0
        if pos >= ids.numel():
            pos = ids.numel() - 1
        heads = (ngram - 1) * heads_per
        local = ple_row_ids(ids, pos, self.eos, ngram, heads_per,
                            self.ple_mult, self.ple_sz)
        emb = torch.zeros(self.ple_embed)
        off = 0
        for h in range(heads):
            if h < self.ple_sz.numel() and int(self.ple_sz[h]) <= 0:
                return
            name = (f"model.layers.{L}.ple.ple_embedding."
                    f"ngram_head.{h}.weight")
            full = f"{self.p}{name}"
            meta = self.c._meta.get(full)
            if not meta:
                continue
            width = int(meta["shape"][-1])
            if off + width > self.ple_embed:
                break
            row = int(local[h]) if h < len(local) else 0
            if width > 0 and row >= 0:
                emb[off:off + width] = self.c.table_row(full, row)[:width]
            off += width
        key = _linear(self, f"model.layers.{L}.ple.key_proj.weight", emb)
        val = _linear(self, f"model.layers.{L}.ple.value_proj.weight", emb)
        key = qwen_rmsnorm(key, _t(self, f"model.layers.{L}.ple.norm_key.weight"),
                           self.eps, group=self.hid)
        qn = qwen_rmsnorm(self.hcx,
                          _t(self, f"model.layers.{L}.ple.norm_query.weight"),
                          self.eps, group=self.hid)
        inv = 1.0 / math.sqrt(self.hid)
        gated = torch.zeros_like(self.hcx)
        for b in range(self.hc):
            sl = slice(b * self.hid, (b + 1) * self.hid)
            g = float((key[sl] * qn[sl]).sum() * inv)
            mag = math.sqrt(1e-6 if abs(g) < 1e-6 else abs(g))
            g = math.copysign(mag, g)
            gated[sl] = torch.sigmoid(torch.tensor(g)) * val
        gnorm = qwen_rmsnorm(
            gated, _t(self, f"model.layers.{L}.ple.norm_conv.weight"),
            self.eps, group=self.hid)
        cw = _t(self, f"model.layers.{L}.ple.conv1d.weight")
        conv_y, self.ple_ring = dilated_conv_step(
            gnorm, cw, self.ple_ring, ngram)
        self.hcx = self.hcx + gated + conv_y
        if ctxn > 0:
            if ctxn > 1:
                self.ple_prev[:ctxn - 1] = self.ple_prev[1:ctxn]
            self.ple_prev[ctxn - 1] = int(token)

    def _gdn(self, L, x):
        hid, hk, hv, dk, dv = self.hid, self.hk, self.hv, self.dk, self.dv
        qkv_n = 2 * hk * dk + hv * dv
        pre = f"model.layers.{L}.linear_attn."
        mixed = _linear(self, pre + "in_proj_qkv.weight", x)
        cw = _t(self, pre + "conv1d.weight")
        conv_y, self.conv[L] = short_conv_step(mixed, cw, self.conv[L])
        z = _linear(self, pre + "in_proj_z.weight", x)
        a = _linear(self, pre + "in_proj_a.weight", x)
        b = torch.sigmoid(_linear(self, pre + "in_proj_b.weight", x))
        a_log = _t(self, pre + "A_log").flatten()[:hv]
        dt = _t(self, pre + "dt_bias").flatten()[:hv]
        g = -a_log.exp() * F.softplus(a + dt)
        q = conv_y[:hk * dk].view(hk, dk)
        k = conv_y[hk * dk:2 * hk * dk].view(hk, dk)
        v = conv_y[2 * hk * dk:qkv_n].view(hv, dv)
        core = gdn_step(q, k, v, g, b, hk, self.S[L])
        nw = _t(self, pre + "norm.weight")
        z = z.view(hv, dv)
        normed = rmsnorm_gated(core, z, nw, self.eps)
        return _linear(self, pre + "out_proj.weight", normed.reshape(-1))

    def _qsa(self, L, x, pos):
        hq, hkv, d = self.hq, self.hkv, self.qsa_d
        dk, rot = self.idx_d, self.rotary_dim
        qd, kvd = hq * d, hkv * d
        pre = f"model.layers.{L}.self_attn."
        qgate = _linear(self, pre + "q_proj.weight", x)
        k = _linear(self, pre + "k_proj.weight", x)
        v = _linear(self, pre + "v_proj.weight", x)
        idx = _linear(self, pre + "indexer.index_qk_proj.weight", x)
        qg = qgate.view(hq, 2, d)
        q, gate = qg[:, 0, :], qg[:, 1, :]
        k = k.view(hkv, d)
        v = v.view(hkv, d)
        q = qwen_rmsnorm(q, _t(self, f"model.layers.{L}.self_attn.q_norm.weight"),
                         self.eps)
        k = qwen_rmsnorm(k, _t(self, f"model.layers.{L}.self_attn.k_norm.weight"),
                         self.eps)
        cos, sin = rope_cs(pos, self.inv_freq, rot, self.section) if rot else (None, None)
        if rot > 0:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        self.qsa_k[L].append(k.to(torch.bfloat16).float())
        self.qsa_v[L].append(v.to(torch.bfloat16).float())
        q_idx = idx[:self.idx_n * dk].view(self.idx_n, dk)
        raw_k = idx[self.idx_n * dk:self.idx_n * dk + dk]
        q_idx = qwen_rmsnorm(
            q_idx, _t(self, f"model.layers.{L}.self_attn.indexer.q_layernorm.weight"),
            self.eps)
        if rot > 0:
            q_idx = apply_rope(q_idx, cos, sin)
        self.qsa_rawk[L].append(raw_k)
        raw = torch.stack(self.qsa_rawk[L], 0)
        kt = torch.stack(self.qsa_k[L], 0)
        vt = torch.stack(self.qsa_v[L], 0)
        tlen = raw.shape[0]
        if rot > 0:
            cs = [rope_cs(t, self.inv_freq, rot, self.section) for t in range(tlen)]
            full_cos = torch.stack([c for c, _ in cs], 0)
            full_sin = torch.stack([s for _, s in cs], 0)
        else:
            full_cos = full_sin = None
        k_ln = _t(self, f"model.layers.{L}.self_attn.indexer.k_layernorm.weight")
        block_topk = self.idx_budget // self.compress
        sel = qsa_select(q_idx, raw, pos, full_cos, full_sin, rot, k_ln,
                         self.eps, self.compress, block_topk)
        attn = qsa_attn(q, kt, vt, sel, 1.0 / math.sqrt(d))
        attn = attn * torch.sigmoid(gate)
        return _linear(self, pre + "o_proj.weight", attn.reshape(-1))

    def _moe(self, L, x):
        """Routed VQ experts plus the gated shared expert."""
        pre = f"model.layers.{L}.mlp."
        logits = _linear(self, pre + "gate.weight", x)
        idx, w = moe_route(logits, self.top_k, self.renorm, c_style=self.c.c_matvec)
        inter, hid = int(self.cfg["moe_intermediate_size"]), self.hid
        y = torch.zeros(hid)
        for eid, wj in zip(idx, w):
            y += float(wj) * _moe_vq(self.c, L, int(eid), x, inter, hid)
        sg = _linear(self, pre + "shared_expert.gate_proj.weight", x)
        su = _linear(self, pre + "shared_expert.up_proj.weight", x)
        sh = _linear(self, pre + "shared_expert.down_proj.weight", F.silu(sg) * su)
        g = torch.sigmoid(_linear(self, pre + "shared_expert_gate.weight", x))
        y += g.reshape(()) * sh
        return y, idx, w

    def step(self, token, pos):
        x = self.c.embed_row(token)
        self.hcx = x.repeat(self.hc)
        hid, hc = self.hid, self.hc
        hiddens = []
        routes = []
        for L in range(self.n_layers):
            if L == self.ple_layer:
                self._ple_inject(token)
            pre = f"model.layers.{L}."
            mixed, inj = hc_mix(
                self.hcx, self,
                pre + "attn_hyper_connection.hc_norm.weight",
                pre + "attn_hyper_connection.input_mix_weight_down.weight",
                pre + "attn_hyper_connection.input_mix_weight_up.weight",
                pre + "attn_hyper_connection.block_inject_weight.weight",
                hc, hid, self.eps)
            block = self._qsa(L, mixed, pos) if self.full[L] else self._gdn(L, mixed)
            cop = self.c.c_matvec
            self.hcx = hc_combine(self.hcx, block, inj, hc, hid, c_ops=cop).reshape(-1)
            mixed, inj = hc_mix(
                self.hcx, self,
                pre + "mlp_hyper_connection.hc_norm.weight",
                pre + "mlp_hyper_connection.input_mix_weight_down.weight",
                pre + "mlp_hyper_connection.input_mix_weight_up.weight",
                pre + "mlp_hyper_connection.block_inject_weight.weight",
                hc, hid, self.eps)
            block, idx, w = self._moe(L, mixed)
            self.hcx = hc_combine(self.hcx, block, inj, hc, hid, c_ops=cop).reshape(-1)
            hiddens.append(self.hcx.clone())
            routes.append((pos, L, idx, w))
        mixed, _ = hc_mix(
            self.hcx, self,
            "model.hyper_connection_mixer.hc_norm.weight",
            "model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.hyper_connection_mixer.input_mix_weight_up.weight",
            None, hc, hid, self.eps)
        logits = _linear(self, "lm_head.weight", mixed)
        return logits, hiddens, routes

    def forward(self, ids):
        self.reset()
        last, all_h, all_r = None, [], []
        for pos, tok in enumerate(ids):
            last, h, r = self.step(int(tok), pos)
            all_h.append(h)
            all_r.extend(r)
        return last, all_h, all_r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--ids", default="1,2,3",
                    help="comma-separated token ids")
    ap.add_argument("--dump", default="", help="last-token logits as f32")
    ap.add_argument("--hidden", default="",
                    help="hyper-state after every layer, token-major like C")
    ap.add_argument("--routes", default="",
                    help="lines: pos layer id0..idK-1 w0..wK-1")
    ap.add_argument("--token-pos", type=int, default=None,
                    help="dump hidden for this 0-based token only; default is "
                         "every token (C WASTE_DUMP_HIDDEN). Negative from end.")
    ap.add_argument("--c-matvec", action="store_true",
                    help="C dotf summation order (default when WASTE_Q8=0)")
    ap.add_argument("--no-c-matvec", action="store_true",
                    help="use torch @ even when WASTE_Q8=0")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.replace(" ", ",").split(",") if x]
    c_matvec = None
    if args.c_matvec:
        c_matvec = True
    elif args.no_c_matvec:
        c_matvec = False
    c = QwenContainer(args.container, c_matvec=c_matvec)
    ref = QwenRef(c)
    with torch.no_grad():
        logits, hiddens, routes = ref.forward(ids)
    if args.dump:
        write_f32(args.dump, logits)
        print(f"dumped logits -> {args.dump}")
    if args.hidden:
        ntok = len(hiddens)
        pos = args.token_pos
        if pos is None:
            seq = range(ntok)
        else:
            if pos < 0:
                pos = ntok + pos
            seq = [pos]
        mode = "wb"
        for t in seq:
            for layer_h in hiddens[t]:
                write_f32(args.hidden, layer_h, mode)
                mode = "ab"
        print(f"dumped hidden -> {args.hidden}")
    if args.routes:
        with open(args.routes, "w") as f:
            for pos, layer, idx, w in routes:
                ids_s = " ".join(str(i) for i in idx)
                w_s = " ".join(f"{float(x):.6g}" for x in w)
                f.write(f"{pos} {layer} {ids_s} {w_s}\n")
        print(f"dumped routes -> {args.routes}")
    print(f"prefill {len(ids)} tok; logits {tuple(logits.shape)}, "
          f"argmax {int(logits.argmax())}, max {float(logits.max()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
