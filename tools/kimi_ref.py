#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
kimi_ref.py — pure-PyTorch reference for this family, off a WASTE container.

Kimi-Linear, Kimi K3 and GLM-5.3-Flash: one file, because they share a KDA
recurrence, an MLA and a sigmoid/top-k router, and three oracles that
implement the same three things differently is three chances to be wrong in
a way the diff cannot see.

Purpose: an oracle the C engine can be diffed against, on this Mac. The HF
modeling code needs fla's Triton kernels, which do not exist on macOS; the
only piece we actually need from fla is `naive_recurrent_kda`, which is
plain PyTorch. Everything else (short conv, gated RMSNorm, MLA with NoPE,
sigmoid/grouped-topk router) is reimplemented here from the reference
source, and matches src/kda.c by construction.

Reads weights from a WASTE container: trunk dequantized once, experts
dequantized on demand per token — the same access pattern the engine has.

  uv run --with torch --with fla-core python tools/kimi_ref.py \
      --container model.waste --tokens 8

  # dump a batch-1 routing trace for routing_stats.py simulate (Gate 2)
  ... --trace trace_kimi.jsonl --tokens 300
"""

import argparse
import glob
import importlib.util
import json
import mmap
import os
import struct
import sys
import time

import torch
import torch.nn.functional as F

VEC_DIM, CB_ENTRIES, ALIGN, HDR_SIZE = 8, 256, 4096, 48
HDR = "<IHHBBHHHIIIIIIII"
KINDS = ("gate", "up", "down")


def load_naive_kda():
    """fla's package __init__ imports Triton; load the pure-torch file only."""
    for root in sys.path:
        for p in glob.glob(os.path.join(root or ".", "fla", "ops", "kda", "naive.py")):
            spec = importlib.util.spec_from_file_location("kda_naive", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m.naive_recurrent_kda
    raise ImportError("fla-core not found (uv run --with fla-core)")


# ------------------------------------------------------------- container ---

class _LazyTrunk:
    """dict-like view over the trunk; materializes on first access."""

    def __init__(self, owner, cap=64):
        self.o, self.cap, self.cache = owner, cap, {}

    def __contains__(self, k):
        return k in self.o._meta

    def __getitem__(self, k):
        v = self.cache.get(k)
        if v is None:
            v = self.o._materialize(k)
            if v is None:
                raise KeyError(k)
            if len(self.cache) >= self.cap:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = v
        return v

    def get(self, k, d=None):
        return self[k] if k in self else d


class Container:
    def __init__(self, path, device="cpu"):
        self.path = path
        self.dev = device
        self.man = json.load(open(os.path.join(path, "manifest.json")))
        self.cfg = self.man["config"]
        self.prefix = self.man.get("tensor_prefix", "")
        self.stages = self.man["expert_quant"]["stages"]
        self.iblock = self.man["expert_quant"].get("index_block", 0)
        self._load_codebooks()
        self._load_trunk()
        self.banks = {}
        for L, meta in self.man["layers"].items():
            f = open(os.path.join(path, meta["file"]), "rb")
            self.banks[int(L)] = (f, meta)

    def _load_codebooks(self):
        data = open(os.path.join(self.path, "codebooks.bin"), "rb").read()
        rec = 16 + CB_ENTRIES * VEC_DIM * 2
        self.books = []
        for off in range(0, len(data), rec):
            t = torch.frombuffer(bytearray(data[off + 16:off + rec]),
                                 dtype=torch.float16).view(CB_ENTRIES, VEC_DIM)
            self.books.append(t.float().to(self.dev))

    def _load_trunk(self):
        """Keep the raw blob and dequantize on demand.

        Expanding everything to f32 costs 4x the stored size — fine for a
        2 GB trunk, impossible for K3's 31 GB (it would want ~124 GB). The
        blob stays as read and each tensor is materialized the first time
        it is asked for, with a bounded cache of the big ones."""
        trunk_path = os.path.join(self.path, "trunk.bin")
        self._trunk_f = open(trunk_path, "rb")
        # mmap, not read(): a Qwen trunk is tens of GiB and must not be
        # copied into the oracle process just to reach one row at a time.
        self._blob = mmap.mmap(self._trunk_f.fileno(), 0, access=mmap.ACCESS_READ)
        self._meta = {e["name"]: e for e in self.man["trunk"]}
        self.t = _LazyTrunk(self)

    def _deq_row(self, name, row, cols=None):
        """One trunk row as f32 — matches waste_deq_row."""
        e = self._meta[name]
        shape, off = e["shape"], e["off"]
        N = shape[-1] if cols is None else cols
        blob = self._blob
        if e["fmt"] == 0:
            base = off + row * N * 4
            return torch.frombuffer(bytearray(blob[base:base + N * 4]),
                                    dtype=torch.float32).clone()
        g = e["group"]
        ng = (N + g - 1) // g
        fmt, rowbytes = e["fmt"], None
        if fmt == 3:
            rowbytes = ng * g // 2
        elif fmt == 7:
            rowbytes = (ng * g * 3 + 7) // 8 + 1
        else:
            rowbytes = ng * g
        qoff = off + row * rowbytes
        soff = e["scale_off"] + row * ng * 2
        if fmt == 3:
            p4 = torch.frombuffer(bytearray(blob[qoff:qoff + rowbytes]),
                                  dtype=torch.uint8)
            sc = torch.frombuffer(bytearray(blob[soff:soff + ng * 2]),
                                  dtype=torch.float16).view(ng).float()
            idx = torch.arange(N)
            k = idx // g
            byte = p4[idx // 2]
            v = torch.where((idx & 1) == 0, byte & 0x0F, byte >> 4).float() - 8.0
            return (v * sc[k]).float()
        elif fmt == 2:
            q = torch.frombuffer(bytearray(blob[qoff:qoff + rowbytes]),
                                 dtype=torch.int8).view(ng, g)
            sc = torch.frombuffer(bytearray(blob[soff:soff + ng * 2]),
                                  dtype=torch.float16).view(ng).float()
            return (q.float() * sc.unsqueeze(-1)).reshape(-1)[:N].float()
        else:
            raise ValueError(f"unsupported trunk fmt {fmt} for row dequant")

    def matvec(self, name, x, batch=1024):
        """y = W @ x for 2-D trunk weight W [rows, cols]. Never materializes W."""
        e = self._meta[name]
        rows = 1
        for s in e["shape"][:-1]:
            rows *= s
        cols = e["shape"][-1]
        x = x.detach().float().cpu().reshape(-1)
        y = torch.empty(rows, dtype=torch.float32)
        for r0 in range(0, rows, batch):
            r1 = min(r0 + batch, rows)
            w = torch.stack([self._deq_row(name, r, cols) for r in range(r0, r1)])
            y[r0:r1] = w @ x
        return y.to(self.dev)

    def matvec_c(self, name, x):
        """y = W @ x with C mv_rows / dotf summation order.

        Matches the engine when WASTE_Q8=0 has dequantized the trunk to f32.
        Torch batched @ can reorder sums enough to swap near-tie MoE routes."""
        e = self._meta[name]
        rows = 1
        for s in e["shape"][:-1]:
            rows *= s
        cols = e["shape"][-1]
        xv = x.detach().float().cpu().reshape(-1).tolist()
        y = [0.0] * rows
        for r in range(rows):
            wr = self._deq_row(name, r, cols).tolist()
            acc = 0.0
            for i in range(cols):
                acc += wr[i] * xv[i]
            y[r] = acc
        return torch.tensor(y, dtype=torch.float32).to(self.dev)

    def embed_row(self, token):
        name = f"{self.prefix}model.embed_tokens.weight"
        return self._deq_row(name, int(token)).to(self.dev)

    def table_row(self, name, row):
        """One row from any trunk matrix (embed, PLE ngram tables, …)."""
        return self._deq_row(name, int(row)).to(self.dev)

    def _materialize(self, name):
        e = self._meta[name]
        blob = self._blob
        shape, off = e["shape"], e["off"]
        rows = 1
        for s in shape[:-1]:
            rows *= s
        cols = shape[-1]
        if e["fmt"] != 0 and rows * cols > 100_000_000:
            raise MemoryError(
                f"{name} is {rows}x{cols} quantized — use matvec() or embed_row()")
        if e["fmt"] == 0:                                  # F32
            n = rows * cols
            x = torch.frombuffer(bytearray(blob[off:off + n * 4]),
                                 dtype=torch.float32).view(*shape)
        else:                                              # Q8G / Q4G
            N, g = cols, e["group"]
            ng = (N + g - 1) // g
            if e["fmt"] == 3:
                b = torch.frombuffer(
                    bytearray(blob[off:off + rows * ng * g // 2]),
                    dtype=torch.uint8).view(rows, ng * g // 2).int()
                q = torch.stack([b & 0x0F, b >> 4], -1)
                q = (q.view(rows, ng, g) - 8).float()
            else:
                q = torch.frombuffer(bytearray(blob[off:off + rows * ng * g]),
                                     dtype=torch.int8).view(rows, ng, g).float()
            sc = torch.frombuffer(
                bytearray(blob[e["scale_off"]:e["scale_off"] + rows * ng * 2]),
                dtype=torch.float16).view(rows, ng, 1).float()
            x = (q * sc).view(rows, ng * g)[:, :N].reshape(*shape)
        return x.to(self.dev)

    def expert(self, L, eid):
        """Dequantize one expert: exactly one pread of its 4 KiB-aligned record."""
        f, meta = self.banks[L]
        # every record in a layer has the same size, so the offset is direct
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
            raw = torch.frombuffer(bytearray(buf[beg:beg + nvec * self.stages]),
                                   dtype=torch.uint8)
            if self.iblock:      # [M/B][nvr][B][stage] -> [nvec][stage]
                B, nvr = self.iblock, N // VEC_DIM
                nb = (M + B - 1) // B
                idx = (raw.view(nb, nvr, B, self.stages).permute(0, 2, 1, 3)
                          .reshape(nb * B, nvr, self.stages)[:M]
                          .reshape(nvec, self.stages).long())
            else:
                idx = raw.view(nvec, self.stages).long()
            recon = torch.zeros(nvec, VEC_DIM)
            for s in range(self.stages):
                recon += self.books[cb_base + i * self.stages + s].cpu()[idx[:, s]]
            sc = torch.frombuffer(bytearray(buf[sc_cur:sc_cur + M * 2]),
                                  dtype=torch.float16).float().view(M, 1)
            sc_cur += M * 2
            out[kind] = (recon.view(M, N) * sc).to(self.dev)
        return out

    def expert_shapes(self):
        h = self.cfg.get("routed_expert_hidden_size") or self.cfg["hidden_size"]
        i = self.cfg["moe_intermediate_size"]
        return [(i, h), (i, h), (h, i)]          # gate, up, down


# ----------------------------------------------------------------- model ---

def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def short_conv(x, w, state):
    """Causal depthwise conv (kernel 4) + SiLU. x [T, C]; state [C, K-1]."""
    C, _, K = w.shape
    seq = torch.cat([state.T, x], 0)                      # [K-1+T, C]
    y = torch.empty_like(x)
    for t in range(x.shape[0]):
        win = seq[t:t + K]                                # [K, C]
        y[t] = (win * w.view(C, K).T).sum(0)
    new_state = seq[-(K - 1):].T.contiguous()
    return F.silu(y), new_state


def situ(g, u, beta, lbeta):
    """K3's SituAndMul."""
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    return a * (lbeta * torch.tanh(u / lbeta) if lbeta else u)


def clamped_swiglu(g, u, limit):
    """GLM's SwiGLU: the gate clamped above, the up half on both sides,
    before the SiLU rather than after it."""
    return F.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)


def apply_attn_res(prefix_sum, blocks, norm_w, proj_w, eps):
    """_apply_attn_res: softmax attention over the block-residual history."""
    v = torch.cat([blocks, prefix_sum.unsqueeze(-2)], -2)     # [..., nb+1, hid]
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w * proj_w)).sum(-1)
    return (scores.softmax(-1).unsqueeze(-2) @ v).squeeze(-2)


class KimiRef:
    def __init__(self, c: Container):
        self.c, self.t, self.cfg = c, c.t, c.cfg
        self.p = c.prefix
        self.eps = self.cfg["rms_norm_eps"]
        self.latent = self.cfg.get("routed_expert_hidden_size")
        self.latent_norm = self.cfg.get("latent_moe_use_norm", False)
        self.ares = self.cfg.get("attn_res_block_size")
        self.full_gate = (self.cfg.get("linear_attn_config", {})
                          .get("use_full_rank_gate", False))
        self.gate_lb = (self.cfg.get("linear_attn_config", {})
                        .get("gate_lower_bound"))
        self.situ = self.cfg.get("hidden_act") == "situ"
        self.swiglu_limit = self.cfg.get("swiglu_limit") or 0.0
        # GLM-5.3-Flash: mHC streams and the DSA indexer. Both read back
        # falsy on every Kimi container and cost nothing there.
        self.hc = self.cfg.get("hc_mult") or 0
        self.hc_iters = self.cfg.get("hc_sinkhorn_iters") or 0
        self.hc_eps = self.cfg.get("hc_eps") or 1e-6
        self.index_topk = self.cfg.get("index_topk") or 0
        self.index_kpool = self.cfg.get("index_kpool") or 1
        self.index_heads = self.cfg.get("index_n_heads") or 0
        self.index_dim = self.cfg.get("index_head_dim") or 0
        self.index_tail = self.cfg.get("index_kpool_always_select_tail", False)
        self.sb = self.cfg.get("activation_situ_beta", 1.0)
        self.slb = self.cfg.get("activation_situ_linear_beta")
        lac = self.cfg["linear_attn_config"]
        self.kda_set = set(lac["kda_layers"])
        self.kda_h, self.kda_d = lac["num_heads"], lac["head_dim"]
        self.conv_k = lac["short_conv_kernel_size"]
        self.naive_kda = load_naive_kda()
        self.n_layers = self.cfg["num_hidden_layers"]
        self.reset()

    def is_kda(self, i):
        return (i + 1) in self.kda_set

    def reset(self):
        self.S = {}
        self.conv = {}
        self.kv = {}
        self.ik = {}              # the indexer's per-layer (key, gate) cache

    def act(self, g, u):
        """SwiGLU in whichever of the family's three forms this model uses."""
        if self.situ:
            return situ(g, u, self.sb, self.slb)
        if self.swiglu_limit:
            return clamped_swiglu(g, u, self.swiglu_limit)
        return F.silu(g) * u

    def kda(self, L, x):
        p = f"{self.p}model.layers.{L}.self_attn."
        T = x.shape[0]
        H, D = self.kda_h, self.kda_d
        if L not in self.conv:
            self.conv[L] = [torch.zeros(H * D, self.conv_k - 1) for _ in range(3)]
        qkv = []
        for i, n in enumerate(("q", "k", "v")):
            proj = x @ self.t[p + f"{n}_proj.weight"].T
            y, st = short_conv(proj, self.t[p + f"{n}_conv1d.weight"], self.conv[L][i])
            self.conv[L][i] = st
            qkv.append(y.view(T, H, D))
        q, k, v = qkv

        g = (x @ self.t[p + "f_a_proj.weight"].T) @ self.t[p + "f_b_proj.weight"].T
        g = g.view(T, H, D)
        # One log-scale per head (tech report eq. 5). K3 pads the tensor to
        # head_dim, so take the first H rather than trusting its length.
        A_log = self.t[p + "A_log"].flatten()[:H].view(H, 1)
        z = g + self.t[p + "dt_bias"].view(H, D)
        if self.gate_lb is not None:
            g = self.gate_lb * torch.sigmoid(A_log.exp() * z)
        else:
            g = -A_log.exp() * F.softplus(z)
        beta = torch.sigmoid(x @ self.t[p + "b_proj.weight"].T)          # [T, H]

        qn = F.normalize(q, dim=-1, p=2)
        kn = F.normalize(k, dim=-1, p=2)
        o, S = self.naive_kda(q=qn.unsqueeze(0), k=kn.unsqueeze(0),
                              v=v.unsqueeze(0), g=g.unsqueeze(0),
                              beta=beta.unsqueeze(0),
                              initial_state=self.S.get(L),
                              output_final_state=True)
        self.S[L] = S
        o = o[0]                                                          # [T,H,D]
        if self.full_gate:
            gate = (x @ self.t[p + "g_proj.weight"].T).view(T, H, D)
        else:
            gate = ((x @ self.t[p + "g_a_proj.weight"].T)
                    @ self.t[p + "g_b_proj.weight"].T).view(T, H, D)
        o = rms_norm(o, self.t[p + "o_norm.weight"], self.eps) * torch.sigmoid(gate)
        return o.reshape(T, H * D) @ self.t[p + "o_proj.weight"].T

    def hc_site(self, L, site, x):
        """One mHC site: the mapping that turns the hc_mult parallel streams
        into (post, comb) and collapses them into the one sequence the
        sublayer runs on. x is [T, H, hid]."""
        p = f"{self.p}model.layers.{L}.hc_{site}_"
        H = self.hc
        flat = x.flatten(1).float()
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + self.eps)
        mix = flat @ self.t[p + "fn"].float().T
        pre_w, post_w, comb_w = mix.split([H, H, H * H], dim=-1)
        pre_b, post_b, comb_b = self.t[p + "base"].float().split([H, H, H * H])
        pre_s, post_s, comb_s = self.t[p + "scale"].float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_s + post_b)
        logits = comb_w.view(*comb_w.shape[:-1], H, H) * comb_s + comb_b.view(H, H)
        comb = logits.softmax(-1) + self.hc_eps
        # Sinkhorn-Knopp onto the doubly-stochastic manifold: columns once,
        # then hc_sinkhorn_iters-1 rounds of (rows, columns).
        comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + self.hc_eps)
        return post, comb, (pre.unsqueeze(-1) * x).sum(dim=1)

    def dsa_mask(self, L, x, q_resid, T, S):
        """DeepSeek Sparse Attention with k-pool compression: which of the S
        cached positions each of the T queries may attend to.

        Pools are index_kpool adjacent tokens compressed to one key by a
        per-channel softmax over a learned gate; a query keeps the best
        index_topk/index_kpool of the pools that are entirely behind it, plus
        the tail that has not filled a pool yet. When a query has no more
        complete pools than it is allowed to keep, that is every visible
        token — which is why short contexts are dense and not approximate."""
        p = f"{self.p}model.layers.{L}.self_attn.indexer."
        D, P = self.index_dim, self.index_kpool
        k = x @ self.t[p + "wk.weight"].T
        # LayerNorm, not RMSNorm, and with a bias — the one place in this
        # model that centres before it scales.
        k = F.layer_norm(k, (D,), self.t[p + "k_norm.weight"],
                         self.t[p + "k_norm.bias"], 1e-6)
        gate = x @ self.t[p + "index_kpool_compress_gate"].T
        if L in self.ik:
            k = torch.cat([self.ik[L][0], k], 0)
            gate = torch.cat([self.ik[L][1], gate], 0)
        self.ik[L] = (k, gate)

        npool = S // P
        ape = self.t[p + "index_kpool_compress_ape"].view(P, D).float()
        prob = (gate[:npool * P].view(npool, P, D).float() + ape).softmax(dim=1)
        pool_keys = (prob * k[:npool * P].view(npool, P, D).float()).sum(1)

        q = (q_resid @ self.t[p + "wq_b.weight"].T).view(T, self.index_heads, D)
        sc = F.relu(torch.einsum("thd,pd->thp", q.float(), pool_keys)
                    * D ** -0.5)
        w = (x @ self.t[p + "weights_proj.weight"].T).float() \
            * self.index_heads ** -0.5
        index_scores = torch.einsum("th,thp->tp", w, sc)

        mask = torch.zeros(T, S, dtype=torch.bool)
        pool_end = torch.arange(npool) * P + P - 1
        dump = os.environ.get("WASTE_DUMP_DSA")
        for t in range(T):
            kvl = S - T + t + 1                       # what query t can see
            visible = pool_end < kvl
            nvis = int(visible.sum())
            keep = min(self.index_topk // P, nvis)
            if keep:
                sel = index_scores[t].masked_fill(
                    ~visible, float("-inf")).topk(keep).indices
                for pi in sel.tolist():
                    mask[t, pi * P:(pi + 1) * P] = True
            if self.index_tail:
                mask[t, kvl - kvl % P:kvl] = True
            # Same line the engine's WASTE_DUMP_DSA writes, so the two
            # selections can be diffed directly rather than inferred from a
            # logit difference — which cannot tell a different ranking from
            # a tie broken the other way.
            if dump and keep < nvis:
                with open(dump, "a") as f:
                    ids = ",".join(str(i) for i in sorted(sel.tolist())) + ","
                    sc = " ".join(f"{v:.9g}" for v in
                                  index_scores[t][:nvis].tolist())
                    f.write(f"{L} {S - T + t} {nvis} {keep} {ids} : {sc}\n")
        return mask

    def mla(self, L, x):
        p = f"{self.p}model.layers.{L}.self_attn."
        cfg, T = self.cfg, x.shape[0]
        nh = cfg["num_attention_heads"]
        qk_n, qk_r, vh = cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]
        qd = qk_n + qk_r
        # K3 factorizes the query projection (q_lora_rank 1536); Kimi-Linear
        # does not, and keeps a single q_proj.
        qa = None
        if cfg.get("q_lora_rank"):
            qa = x @ self.t[p + "q_a_proj.weight"].T
            qa = rms_norm(qa, self.t[p + "q_a_layernorm.weight"], self.eps)
            q = (qa @ self.t[p + "q_b_proj.weight"].T).view(T, nh, qd)
        else:
            q = (x @ self.t[p + "q_proj.weight"].T).view(T, nh, qd)
        ckv = x @ self.t[p + "kv_a_proj_with_mqa.weight"].T
        kpass, krot = ckv.split([cfg["kv_lora_rank"], qk_r], dim=-1)
        kpass = rms_norm(kpass, self.t[p + "kv_a_layernorm.weight"], self.eps)
        kb = (kpass @ self.t[p + "kv_b_proj.weight"].T).view(T, nh, qk_n + vh)
        knope, val = kb.split([qk_n, vh], dim=-1)
        k = torch.cat([knope, krot.view(T, 1, qk_r).expand(T, nh, qk_r)], -1)
        # NoPE: mla_use_nope, so no rotary is applied to the "rot" dims
        if L in self.kv:
            k = torch.cat([self.kv[L][0], k], 0)
            val = torch.cat([self.kv[L][1], val], 0)
        self.kv[L] = (k, val)
        S = k.shape[0]
        att = torch.einsum("thd,shd->hts", q, k) * (qd ** -0.5)
        mask = torch.full((T, S), float("-inf")).triu(S - T + 1)
        if self.index_topk:
            # The indexer runs after the cache update, so a query always
            # sees itself, and its answer is shared by every head.
            mask = mask.masked_fill(~self.dsa_mask(L, x, qa, T, S),
                                    float("-inf"))
        att = (att + mask).softmax(-1)
        o = torch.einsum("hts,shd->thd", att, val).reshape(T, nh * vh)
        if cfg.get("mla_use_output_gate"):
            o = o * torch.sigmoid(x @ self.t[p + "g_proj.weight"].T)
        return o @ self.t[p + "o_proj.weight"].T

    def moe(self, L, x, trace=None):
        p = f"{self.p}model.layers.{L}.block_sparse_moe."
        cfg, T = self.cfg, x.shape[0]
        logits = x.float() @ self.t[p + "gate.weight"].float().T
        scores = torch.sigmoid(logits)
        choice = scores + self.t[p + "gate.e_score_correction_bias"].unsqueeze(0)
        k = cfg["num_experts_per_token"]
        topk_idx = torch.topk(choice, k=k, dim=-1, sorted=False)[1]
        w = scores.gather(1, topk_idx)
        if cfg.get("moe_renormalize", True):
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * cfg["routed_scaling_factor"]
        if trace is not None:
            trace.append(sorted(topk_idx[-1].tolist()))
        xin = x
        if self.latent:
            xin = x @ self.t[p + "routed_expert_down_proj.weight"].T
        y = torch.zeros_like(xin)
        for t in range(T):
            for j in range(k):
                e = int(topk_idx[t, j])
                E = self.c.expert(L, e)
                a, b = xin[t] @ E["gate"].T, xin[t] @ E["up"].T
                y[t] += w[t, j] * (self.act(a, b) @ E["down"].T)
        if self.latent:
            if self.latent_norm:
                y = rms_norm(y, self.t[p + "routed_expert_norm.weight"], self.eps)
            y = y @ self.t[p + "routed_expert_up_proj.weight"].T
        sg = self.t[p + "shared_experts.gate_proj.weight"]
        su = self.t[p + "shared_experts.up_proj.weight"]
        sd = self.t[p + "shared_experts.down_proj.weight"]
        sa, sbv = x @ sg.T, x @ su.T
        return y + self.act(sa, sbv) @ sd.T

    def dense_mlp(self, L, x):
        p = f"{self.p}model.layers.{L}.mlp."
        a, b = x @ self.t[p + "gate_proj.weight"].T, x @ self.t[p + "up_proj.weight"].T
        return self.act(a, b) @ self.t[p + "down_proj.weight"].T

    def forward(self, ids, trace=None):
        x = self.t[self.p + "model.embed_tokens.weight"][ids]
        if self.hc:
            # mHC: every stream starts as a copy of the embedding.
            x = x.unsqueeze(1).expand(-1, self.hc, -1).contiguous()
        blocks, ps = None, None
        for L in range(self.n_layers):
            pre = f"{self.p}model.layers.{L}."
            if self.hc:
                post, comb, col = self.hc_site(L, "attn", x)
                h = rms_norm(col, self.t[pre + "input_layernorm.weight"], self.eps)
                att = self.kda(L, h) if self.is_kda(L) else self.mla(L, h)
                x = (post.unsqueeze(-1) * att.unsqueeze(-2)
                     + comb.transpose(-1, -2) @ x)

                post, comb, col = self.hc_site(L, "ffn", x)
                h = rms_norm(col, self.t[pre + "post_attention_layernorm.weight"],
                             self.eps)
                has_moe = f"{pre}block_sparse_moe.gate.weight" in self.t
                ffn = self.moe(L, h, trace) if has_moe else self.dense_mlp(L, h)
                x = (post.unsqueeze(-1) * ffn.unsqueeze(-2)
                     + comb.transpose(-1, -2) @ x)
                if os.environ.get("WASTE_DUMP_HIDDEN"):
                    with open(os.environ["WASTE_DUMP_HIDDEN"],
                              "ab" if L else "wb") as f:
                        v = x[-1].float().flatten().tolist()
                        f.write(struct.pack(f"<{len(v)}f", *v))
                continue
            if self.ares:
                ps = x
                if blocks is not None and blocks.shape[-2] > 0:
                    x = apply_attn_res(ps, blocks,
                                       self.t[pre + "self_attention_res_norm.weight"],
                                       self.t[pre + "self_attention_res_proj.weight"],
                                       self.eps)
                if L % self.ares == 0:
                    b = ps.unsqueeze(-2)
                    blocks = b if blocks is None else torch.cat([blocks, b], -2)
                    ps = None
            h = rms_norm(x, self.t[pre + "input_layernorm.weight"], self.eps)
            att = self.kda(L, h) if self.is_kda(L) else self.mla(L, h)
            if self.ares:
                ps = att if ps is None else ps + att
                x = apply_attn_res(ps, blocks,
                                   self.t[pre + "mlp_res_norm.weight"],
                                   self.t[pre + "mlp_res_proj.weight"], self.eps)
            else:
                x = x + att
            h = rms_norm(x, self.t[pre + "post_attention_layernorm.weight"], self.eps)
            has_moe = f"{pre}block_sparse_moe.gate.weight" in self.t
            ffn = self.moe(L, h, trace) if has_moe else self.dense_mlp(L, h)
            if self.ares:
                ps = ps + ffn
                x = ps
            else:
                x = x + ffn
            if os.environ.get("WASTE_DUMP_HIDDEN"):
                with open(os.environ["WASTE_DUMP_HIDDEN"], "ab" if L else "wb") as f:
                    v = x[-1].float().tolist()
                    f.write(struct.pack(f"<{len(v)}f", *v))
        if self.hc:
            x = x.mean(1)                 # hc_head: an unweighted mean
        if self.ares and blocks is not None and blocks.shape[-2] > 0:
            x = apply_attn_res(x, blocks,
                               self.t[self.p + "model.output_attn_res_norm.weight"],
                               self.t[self.p + "model.output_attn_res_proj.weight"],
                               self.eps)
        x = rms_norm(x, self.t[self.p + "model.norm.weight"], self.eps)
        return x @ self.t[self.p + "lm_head.weight"].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--prompt-ids", default="1,100,200,300,400")
    ap.add_argument("--prompt", default="", help="text prompt (needs tiktoken)")
    ap.add_argument("--src", default="/Volumes/WasteDisk/kimi-linear",
                    help="where tiktoken.model lives")
    ap.add_argument("--trace", default="")
    ap.add_argument("--dump", default="", help="save logits+state for the C diff")
    args = ap.parse_args()

    t0 = time.time()
    c = Container(args.container)
    ref = KimiRef(c)
    print(f"loaded in {time.time()-t0:.1f}s: {ref.n_layers} layers, "
          f"{sum(1 for i in range(ref.n_layers) if ref.is_kda(i))} KDA / "
          f"{sum(1 for i in range(ref.n_layers) if not ref.is_kda(i))} MLA")

    enc = None
    if args.prompt:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import kimi_tok
        enc, _sp = kimi_tok.load(args.src)
        ids = torch.tensor(enc.encode(args.prompt))
        print(f"prompt: {args.prompt!r} -> {len(ids)} tokens")
    else:
        ids = torch.tensor([int(x) for x in args.prompt_ids.split(",")])
    trace_f = open(args.trace, "w") if args.trace else None
    torch.manual_seed(0)

    with torch.no_grad():
        t0 = time.time()
        logits = ref.forward(ids)
        if args.dump:
            with open(args.dump, "wb") as f:
                b = logits[-1].contiguous().float().flatten()
                f.write(struct.pack(f"<{b.numel()}f", *b.tolist()))
            print(f"dumped logits -> {args.dump}")
        print(f"prefill {len(ids)} tok in {time.time()-t0:.1f}s; "
              f"logits {tuple(logits.shape)}, "
              f"argmax {int(logits[-1].argmax())}, "
              f"max {logits[-1].max():.3f}")
        cur = int(logits[-1].argmax())
        out = [cur]
        if enc:
            print(f"  [pre] {enc.decode([cur])!r}")
        for i in range(args.tokens):
            t0 = time.time()
            layers = [] if trace_f else None
            logits = ref.forward(torch.tensor([cur]), trace=layers)
            if trace_f:
                trace_f.write(json.dumps({"tok": i, "layers": layers}) + "\n")
                trace_f.flush()
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            if enc:
                print(f"  [{i:>3}] {enc.decode([nxt])!r}  ({time.time()-t0:.1f}s)",
                      flush=True)
            else:
                print(f"  tok {i}: {cur} -> {nxt}  ({time.time()-t0:.1f}s)")
            cur = nxt
        if enc:
            print(f"\n=== continuation ===\n{args.prompt}{enc.decode(out)}")
    if trace_f:
        trace_f.close()
        print(f"wrote {args.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
