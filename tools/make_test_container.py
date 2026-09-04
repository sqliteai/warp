#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
make_test_container.py — a valid WASTE container of a few megabytes.

The end-to-end checks in tests/run.sh all need a container, and the
smallest real one is 19 GB. That put the engine's most valuable tests —
chunked prefill against token-at-a-time, the expert cache against no
cache, session state round-trip, the RAM budget — out of reach of CI and
of anyone who has not downloaded a model.

None of those checks look at whether the weights mean anything. They
compare the engine against itself: same tokens, two paths, same logits.
So this writes a structurally valid container full of deterministic
noise, at dimensions that make a forward pass take milliseconds.

Deliberately stdlib-only. The converter needs torch and a C extension;
requiring either here would just move the barrier rather than remove it.

  python3 tools/make_test_container.py /tmp/tiny.waste
  ./waste run /tmp/tiny.waste "hello" -n 4 --budget 1G
"""

import argparse
import json
import os
import random
import struct
import sys
import zlib

MAGIC_EXPERT = 0x50584557        # 'WEXP'
MAGIC_CODEBOOK = 0x4B424357      # 'WCBK'
ALIGN = 4096
FMT_F32, FMT_Q8G, FMT_Q4G, FMT_VQ3R, FMT_VQ4P = 0, 2, 3, 4, 8
VEC_DIM, CB_ENTRIES, STAGES, IDX_BLOCK = 8, 256, 3, 64
# --index-bits 6 switches all three: the engine accepts 6 only as 4 stages
# of 64 entries (src/model.c), which is also the only combination
# convert.py writes. PACKED and INDEX_BITS are set in main() after the
# arguments are read; the default leaves every byte below identical.
PACKED, INDEX_BITS = False, 8
GROUP = 128
KINDS = ("gate", "up", "down")

# A Kimi-Linear at 1/18 scale. The shapes are the ones the engine reads,
# not a guess: every dimension below is the small counterpart of something
# in a real manifest, and the layer mix keeps one dense layer, KDA layers
# and at least one Gated MLA layer so all three paths are exercised.
H_KDA, D_KDA = 4, 32                      # KDA heads, head dim -> C = 128
CFG = {
    "model_type": "kimi_linear",
    # a real converted config carries this, and `info` names the model from
    # it — model_type says kimi_linear on every model in the family
    "architectures": ["KimiLinearForCausalLM"],
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "first_k_dense_replace": 1,           # layer 0 dense, the rest MoE
    "intermediate_size": 256,             # the dense layer's FFN
    "moe_intermediate_size": 64,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "routed_scaling_factor": 2.0,
    "num_attention_heads": 4,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "kv_lora_rank": 32,
    "q_lora_rank": None,
    "mla_use_nope": True,
    "rms_norm_eps": 1e-5,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "linear_attn_config": {
        "head_dim": D_KDA,
        "num_heads": H_KDA,
        "short_conv_kernel_size": 4,
        # 1-based in the manifest, as the engine expects
        "kda_layers": [1, 2, 4],
        "full_attn_layers": [3],
    },
}
C_KDA = H_KDA * D_KDA

# GLM's tower at 1/32 scale. Every field the engine reads, and the ratios
# that matter kept: head_dim 16 so the rotary table has something to
# interleave, merge 2 so the downsample is a real reshape, and two blocks
# rather than one so a residual has somewhere to accumulate.
VISION = {
    "model_type": "glm5_next_vision",
    "depth": 2,
    "hidden_size": 32,
    "num_heads": 2,
    "intermediate_size": 64,
    "out_hidden_size": 128,          # = CFG hidden_size
    "projection_intermediate_size": 96,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
    "rms_norm_eps": 1e-5,
    "swiglu_limit": 10.0,
    "attention_bias": True,
    "hidden_act": "silu",
}

# --glm turns the above into a GLM-5.3-Flash at the same scale. Three things
# are new and none of them exist on any Kimi container, so this is the only
# shape that reaches them:
#
#   mHC             the residual stream is HC_MULT parallel copies
#   swiglu_limit    gate/up clamped before the SiLU-and-multiply
#   the DSA indexer full-attention layers score pools of INDEX_KPOOL tokens
#                   and attend over the best INDEX_TOPK/INDEX_KPOOL of them
#
# index_topk is deliberately tiny. At 8 the sparse branch is reached after
# eight tokens, which a test can prompt for; at GLM's own 2048 it would need
# a 2049-token prompt and would never run in CI.
HC_MULT, HC_ITERS = 4, 20
INDEX_HEADS, INDEX_DIM, INDEX_KPOOL, INDEX_TOPK = 2, 16, 4, 8
GLM = {
    "model_type": "glm5_next_text",
    "architectures": ["Glm5NextForConditionalGeneration"],
    "q_lora_rank": 48,
    "qk_rope_head_dim": 0,            # NoPE all the way: GLM has no rotary
    "hidden_act": "silu",
    "swiglu_limit": 10.0,
    "hc_mult": HC_MULT,
    "hc_sinkhorn_iters": HC_ITERS,
    "hc_eps": 1e-6,
    "index_n_heads": INDEX_HEADS,
    "index_head_dim": INDEX_DIM,
    "index_kpool": INDEX_KPOOL,
    "index_topk": INDEX_TOPK,
    "index_kpool_always_select_tail": True,
}

# --qwen writes a text-only Qwen3.8-Flash-Next fixture at the same scale.
# Shapes match official qwen4_exp: GDN qkv = 2*Hk*Dk+Hv*Dv, QSA o_proj is
# hid x n_heads*head_dim, indexer is (n+kv)*idx_dim, 16 PLE heads.
H_GDN, D_GDN = 4, 8
QWEN_CFG = {
    "model_type": "qwen4_exp_text",
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "num_experts_per_tok": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_key_heads": 4,
    "linear_key_head_dim": D_GDN,
    "linear_num_value_heads": 12,
    "linear_value_head_dim": D_GDN,
    "linear_conv_kernel_dim": 4,
    "hc_count": 4,
    "hc_lowrank": 8,
    "ple_embed_dim": 128,
    "ple_layer_ids": [2],
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "ngram_vocab_size_base": 64,
    "split_ngram_parts": 8,
    "heads_per_ngram": 8,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 8,
    "indexer_budget": 32,
    "indexer_compress_ratio": 4,
    "partial_rotary_factor": 0.5,
    "rope_parameters": {
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "mrope_section": [2, 2, 2],
        "rope_type": "default",
        "mrope_interleaved": True,
    },
    "ple_head_offsets": [0] * 16,
    "ple_head_vocab_sizes": [32] * 16,
    "ple_layer_multipliers": [3, 5, 7],
}
PLE_HEADS, PLE_HEAD_WIDTH = 16, 8

def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


QWEN_PLE_SIZES = [_nth_prime_after(10, h + 1) for h in range(PLE_HEADS)]
QWEN_PLE_MULT = [3, 5, 7]
QWEN_CFG["ple_head_vocab_sizes"] = QWEN_PLE_SIZES
_off, _offsets = 0, []
for _s in QWEN_PLE_SIZES:
    _offsets.append(_off)
    _off += _s
QWEN_CFG["ple_head_offsets"] = _offsets
QWEN_CFG["ple_layer_multipliers"] = QWEN_PLE_MULT

# --rope turns the above into a DeepSeek-V3 at the same scale, which is the
# only shape that reaches src/model.c's rotary: the Kimi models set
# mla_use_nope and pass the qk_rope dims through unrotated, so a container
# built from CFG as it stands leaves rope_init and rope_apply dead.
#
# The rope block is Kimi-K2-Instruct's config.json verbatim. DeepSeek-V3 and
# R1 ship the same shape with factor 40 and beta_fast 32; K2's beta_fast ==
# beta_slow == 1.0 is the more awkward of the two because it collapses YaRN's
# correction range to a two-dim ramp, so it is the one worth pinning.
V3_ROPE = {
    "rope_theta": 50000.0,
    "rope_scaling": {"beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0,
                     "mscale": 1.0, "mscale_all_dim": 1.0,
                     "original_max_position_embeddings": 4096,
                     "type": "yarn"},
}


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def f16(vals):
    return struct.pack("<%de" % len(vals), *vals)


class Trunk:
    """Accumulates trunk.bin and its index, in the layout model.c reads:
    F32 verbatim, Q8G/Q4G as quantized rows followed by one fp16 scale per
    group."""

    def __init__(self, rng, prefix=""):
        self.buf = bytearray()
        self.index = []
        self.rng = rng
        self.prefix = prefix

    def f32(self, name, shape, prefixed=True):
        n = 1
        for s in shape:
            n *= s
        off = len(self.buf)
        self.buf += f32([self.rng.uniform(-0.5, 0.5) for _ in range(n)])
        self.index.append({"name": (self.prefix if prefixed else "") + name,
                           "fmt": FMT_F32,
                           "off": off, "shape": list(shape),
                           "bytes": len(self.buf) - off})

    def quant(self, name, shape, bits=None, prefixed=True):
        """Quantized rows plus one fp16 scale per group.

        The width mirrors tools/convert.py: 4 bits for the bulk, 8 at the
        two ends of the network where error is least forgiving. Emitting
        Q8G everywhere — which this did until a Q4G trunk turned out not to
        load under WASTE_Q8=0 — leaves the 4-bit paths with no container CI
        can reach, and a default conversion produces almost nothing else.
        """
        rows, N = 1, shape[-1]
        for s in shape[:-1]:
            rows *= s
        if bits is None:
            bits = 8 if name.endswith(("embed_tokens.weight",
                                       "lm_head.weight")) else 4
        ng = (N + GROUP - 1) // GROUP
        n = ng * GROUP          # payload is padded to whole groups, as the
        off = len(self.buf)     # converter does
        if bits == 8:
            self.buf += bytes((self.rng.randrange(-127, 128) & 0xFF)
                              for _ in range(rows * n))
        else:
            # two signed nibbles per byte, low one first, biased by +8 —
            # the packing src/model.c's 4-bit decode assumes. n is a whole
            # number of groups of 128, so the pairs never straddle a row.
            v = [self.rng.randrange(-8, 8) for _ in range(rows * n)]
            self.buf += bytes(((v[i] + 8) & 0x0F) | (((v[i + 1] + 8) & 0x0F) << 4)
                              for i in range(0, len(v), 2))
        soff = len(self.buf)
        self.buf += f16([self.rng.uniform(0.002, 0.02)
                         for _ in range(rows * ng)])
        self.index.append({"name": (self.prefix if prefixed else "") + name,
                           "fmt": FMT_Q8G if bits == 8 else FMT_Q4G,
                           "off": off, "shape": list(shape), "group": GROUP,
                           "scale_off": soff, "bytes": len(self.buf) - off})


def block_indices(idx, M, N):
    """[stages][M*N/8] -> [M/B][pos][row_in_block][stage], the layout the
    gather loop walks. Rows are padded to a whole block with zeros."""
    nvr = N // VEC_DIM
    pad = (-M) % IDX_BLOCK
    nb = (M + pad) // IDX_BLOCK
    out = bytearray(nb * nvr * IDX_BLOCK * STAGES)
    for b in range(nb):
        for v in range(nvr):
            for r in range(IDX_BLOCK):
                row = b * IDX_BLOCK + r
                dst = ((b * nvr + v) * IDX_BLOCK + r) * STAGES
                if row >= M:
                    continue                      # padding stays zero
                src = row * nvr + v
                for s in range(STAGES):
                    out[dst + s] = idx[s][src]
    return bytes(out)


def block_indices_packed6(idx, M, N):
    """block_indices, then four 6-bit stages squeezed into three bytes per
    row — the VQ4P layout. Same [M/B][pos][row_in_block] blocking; only the
    trailing per-row run changes, from four whole bytes to three packed
    ones, which keeps a VQ4P record the same size as VQ3R's. Little-endian
    bit order, LSB of stage 0 at bit 0, byte-for-byte the packing
    tools/convert.py's block_indices_packed writes, so the engine's
    P6_J0..P6_J3 unpack recovers the stages in order."""
    nvr = N // VEC_DIM
    pad = (-M) % IDX_BLOCK
    nb = (M + pad) // IDX_BLOCK
    out = bytearray(nb * nvr * IDX_BLOCK * 3)
    for b in range(nb):
        for v in range(nvr):
            for r in range(IDX_BLOCK):
                row = b * IDX_BLOCK + r
                if row >= M:
                    continue                      # padding stays zero
                src = row * nvr + v
                s0, s1, s2, s3 = (idx[s][src] for s in range(4))
                dst = ((b * nvr + v) * IDX_BLOCK + r) * 3
                out[dst]     = (s0 | (s1 << 6)) & 0xFF
                out[dst + 1] = ((s1 >> 2) | (s2 << 4)) & 0xFF
                out[dst + 2] = ((s2 >> 4) | (s3 << 2)) & 0xFF
    return bytes(out)


def write_expert(f, layer, eid, cb_base, shapes, rng):
    hdr_size = 48
    off, offsets, body = hdr_size, [], bytearray()
    for (M, N) in shapes:
        nvec = M * N // VEC_DIM
        idx = [[rng.randrange(CB_ENTRIES) for _ in range(nvec)]
               for _ in range(STAGES)]
        offsets.append(off)
        b = block_indices_packed6(idx, M, N) if PACKED \
            else block_indices(idx, M, N)
        body += b
        off += len(b)
    corr_off = off
    for (M, _N) in shapes:
        b = f16([rng.uniform(0.01, 0.05) for _ in range(M)])
        body += b
        off += len(b)

    total = hdr_size + len(body)
    blocks = (total + ALIGN - 1) // ALIGN
    hdr = struct.pack("<IHHBBHHHIIIIIIII",
                      MAGIC_EXPERT, layer, eid,
                      FMT_VQ4P if PACKED else FMT_VQ3R, 0, cb_base, 0, 0,
                      blocks, offsets[0], offsets[1], offsets[2], corr_off,
                      zlib.crc32(bytes(body)) & 0xFFFFFFFF, 0, 0)
    assert len(hdr) == hdr_size
    f.write(hdr)
    f.write(body)
    f.write(b"\0" * (blocks * ALIGN - total))
    return blocks * ALIGN


# ---- optional tokenizer ---------------------------------------------------
#
# Off by default: tests/run.sh states, in several places, that the synthetic
# container carries no tokenizer, and quietly changing that would turn its
# SKIPs into checks nobody asked for. `--tokenizer` is for the callers that
# need one — serve/ has to encode a chat template against real specials, and
# a 983 GB download is not a test dependency.
#
# The vocabulary is the 256 single bytes plus a few merges, which is enough
# to make BPE actually merge rather than emit one token per character, and
# the ranks are the ids, as in a tiktoken mergeable_ranks file.

MERGES = [
    b"he", b"in", b"re", b"on", b"at", b"en", b"nd", b"ti", b"es", b"or",
    b" t", b" a", b" s", b" w", b" o", b" i", b" c", b" b", b" f", b" m",
    b"the", b"ing", b"and", b" th", b" the", b" an", b" to", b" of",
    b"hello", b"world", b"weather", b"Paris", b"city", b"json", b"true",
    b"false", b"null", b"message", b"role", b"user", b"assistant", b"system",
    b"tool", b"call", b"argument", b"response", b"think", b"index", b"type",
]

# K3's four XTML markers, the media block, and the reserved block the
# tokenizer positions BOS/EOS in. Order matters: waste_tok_open puts
# [BOS] at n_tokens, [EOS] at n_tokens+1 and treats n_tokens+2 as the id
# generation_config.json names — <|end_of_msg|> on both Kimi releases.
SPECIALS = [
    "[BOS]", "[EOS]", "<|end_of_msg|>",
    "<|open|>", "<|close|>", "<|sep|>",
    "<|media_begin|>", "<|media_content|>", "<|media_pad|>", "<|media_end|>",
    "<|kimi_image_placeholder|>",
]


def write_tokenizer(outdir):
    """tokenizer.model + specials.json. Returns the total vocabulary size."""
    import base64

    tokens = [bytes([b]) for b in range(256)] + MERGES
    with open(os.path.join(outdir, "tokenizer.model"), "wb") as f:
        for rank, tok in enumerate(tokens):
            f.write(base64.b64encode(tok) + b" " + str(rank).encode() + b"\n")

    base = len(tokens)
    specials = [{"id": base + i, "text": s} for i, s in enumerate(SPECIALS)]
    # newline="\n" on every JSON the container carries. Python's text mode
    # translates on Windows, so the same seed builds a byte-different
    # container there. The engine parses it either way and nothing fails
    # loudly — what breaks is any gate that hashes a container to prove
    # provenance. #36 gap 2, one layer in.
    with open(os.path.join(outdir, "specials.json"), "w",
              newline="\n") as f:
        json.dump(specials, f, indent=1)

    return base + len(SPECIALS)


def write_qwen_container(args, rng):
    """A structurally valid Qwen text fixture. Format v0, WEXP unchanged."""
    cfg = dict(QWEN_CFG)
    hid = cfg["hidden_size"]
    moe = cfg["moe_intermediate_size"]
    hc, lr = cfg["hc_count"], cfg["hc_lowrank"]
    hc_w = hc * hid
    t = Trunk(rng, "")
    t.quant("model.embed_tokens.weight", [cfg["vocab_size"], hid])
    t.f32("model.hyper_connection_mixer.hc_norm.weight", [hc_w])
    t.quant("model.hyper_connection_mixer.input_mix_weight_down.weight", [lr, hc_w])
    t.quant("model.hyper_connection_mixer.input_mix_weight_up.weight", [hc_w, lr])
    for L, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{L}."
        for side in ("attn_hyper_connection", "mlp_hyper_connection"):
            t.f32(p + side + ".hc_norm.weight", [hc_w])
            t.quant(p + side + ".block_inject_weight.weight", [hc, hc_w])
            t.quant(p + side + ".input_mix_weight_down.weight", [lr, hc_w])
            t.quant(p + side + ".input_mix_weight_up.weight", [hc_w, lr])
        if kind == "linear_attention":
            a = p + "linear_attn."
            hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
            dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
            qkv = 2 * hk * dk + hv * dv
            t.f32(a + "A_log", [hv])
            t.f32(a + "dt_bias", [hv])
            t.f32(a + "conv1d.weight", [qkv, 1, cfg["linear_conv_kernel_dim"]])
            t.quant(a + "in_proj_qkv.weight", [qkv, hid])
            t.quant(a + "in_proj_z.weight", [hv * dv, hid])
            t.quant(a + "in_proj_a.weight", [hv, hid])
            t.quant(a + "in_proj_b.weight", [hv, hid])
            t.f32(a + "norm.weight", [dv])
            t.quant(a + "out_proj.weight", [hid, hv * dv])
        else:
            a = p + "self_attn."
            qd = cfg["num_attention_heads"] * cfg["head_dim"]
            kvd = cfg["num_key_value_heads"] * cfg["head_dim"]
            idxd = ((cfg["indexer_n_heads"] + cfg["indexer_kv_heads"])
                    * cfg["indexer_head_dim"])
            t.quant(a + "q_proj.weight", [qd * 2, hid])
            t.quant(a + "k_proj.weight", [kvd, hid])
            t.quant(a + "v_proj.weight", [kvd, hid])
            t.quant(a + "o_proj.weight", [hid, qd])
            t.f32(a + "q_norm.weight", [cfg["head_dim"]])
            t.f32(a + "k_norm.weight", [cfg["head_dim"]])
            t.quant(a + "indexer.index_qk_proj.weight", [idxd, hid])
            t.f32(a + "indexer.q_layernorm.weight", [cfg["indexer_head_dim"]])
            t.f32(a + "indexer.k_layernorm.weight", [cfg["indexer_head_dim"]])
        m = p + "mlp."
        t.quant(m + "gate.weight", [cfg["num_experts"], hid])
        t.quant(m + "shared_expert.gate_proj.weight", [moe, hid])
        t.quant(m + "shared_expert.up_proj.weight", [moe, hid])
        t.quant(m + "shared_expert.down_proj.weight", [hid, moe])
        t.quant(m + "shared_expert_gate.weight", [1, hid])
        if L == 1:
            pe = cfg["ple_embed_dim"]
            t.quant(p + "ple.key_proj.weight", [hc_w, pe])
            t.quant(p + "ple.value_proj.weight", [hid, pe])
            t.f32(p + "ple.conv1d.weight", [hc_w, 1, cfg["ple_conv_kernel_size"]])
            t.f32(p + "ple.norm_conv.weight", [hc_w])
            t.f32(p + "ple.norm_key.weight", [hc_w])
            t.f32(p + "ple.norm_query.weight", [hc_w])
            ngram_heads = (cfg["ngram_size"] - 1) * cfg["heads_per_ngram"]
            head_w = pe // ngram_heads
            for h in range(PLE_HEADS):
                t.quant(p + f"ple.ple_embedding.ngram_head.{h}.weight",
                        [QWEN_PLE_SIZES[h], head_w], bits=8)
    t.quant("lm_head.weight", [cfg["vocab_size"], hid])
    with open(os.path.join(args.out, "trunk.bin"), "wb") as f:
        f.write(t.buf)

    shapes = [(moe, hid), (moe, hid), (hid, moe)]
    layers, cb_base = {}, 0
    with open(os.path.join(args.out, "codebooks.bin"), "wb") as cf:
        for L in range(cfg["num_hidden_layers"]):
            for ki in range(len(KINDS)):
                for si in range(STAGES):
                    cid = cb_base + ki * STAGES + si
                    cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK,
                                         cid & 0xFFFF, FMT_VQ3R, VEC_DIM,
                                         CB_ENTRIES, 0))
                    cf.write(f16([rng.uniform(-0.3, 0.3)
                                  for _ in range(CB_ENTRIES * VEC_DIM)]))
            name = f"experts-L{L}.bin"
            with open(os.path.join(args.out, name), "wb") as bf:
                total = sum(write_expert(bf, L, e, cb_base, shapes, rng)
                            for e in range(cfg["num_experts"]))
            layers[str(L)] = {"file": name, "experts": cfg["num_experts"],
                              "bytes": total, "codebook_base": cb_base}
            cb_base += len(KINDS) * STAGES

    manifest = {
        "format_version": 0,
        "arch": "qwen4_exp_text",
        "tensor_prefix": "",
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R", "stages": STAGES, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
                         "bits_per_weight": STAGES},
        "layers": layers,
        "trunk": t.index,
    }
    with open(os.path.join(args.out, "manifest.json"), "w",
              newline="\n") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"wrote {args.out}: qwen4_exp_text, {cfg['num_hidden_layers']} layers, "
          f"{cfg['num_experts']} experts, {total / (1 << 20):.1f} MB")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed, byte-identical container")
    ap.add_argument("--tokenizer", action="store_true",
                    help="also write a small tiktoken-style vocabulary with "
                         "K3's XTML specials, so the container can be driven "
                         "with text instead of raw ids")
    ap.add_argument("--index-bits", type=int, default=8, choices=(6, 8),
                    help="6 writes a VQ4P container: 4 stages of 64 entries, "
                         "indices packed 4x6 bits into 3 bytes, the only "
                         "combination the engine accepts at index_bits 6 and "
                         "the one convert.py writes. Default is 8, and at 8 "
                         "every byte of the container is exactly what this "
                         "script wrote before the flag existed.")
    ap.add_argument("--prefix", default="", metavar="PFX",
                    help="put the text tensors under a tensor_prefix, e.g. "
                         "language_model., and add one tensor outside it — "
                         "K3's shape, and the one the loader skips")
    ap.add_argument("--glm", action="store_true",
                    help="a GLM-5.3-Flash instead of a Kimi-Linear: mHC "
                         "residual streams, a clamped SwiGLU and the DSA "
                         "indexer on the full-attention layers")
    ap.add_argument("--vision", action="store_true",
                    help="--glm only: add a GLM vision tower at test scale, "
                         "so src/vision.c's second tower has a container CI "
                         "can reach")
    ap.add_argument("--index-topk", type=int, metavar="N",
                    help="--glm only: override index_topk. The default 8 puts "
                         "the sparse branch two pools in, so a 12-token "
                         "prompt reaches it; a value above the prompt length "
                         "keeps every pool and must give dense attention")
    ap.add_argument("--rope", action="store_true",
                    help="a DeepSeek-V3 instead of a Kimi-Linear: every layer "
                         "MLA, no mla_use_nope, and rope_theta with YaRN — the "
                         "only shape that reaches the engine's rotary")
    ap.add_argument("--qk-rope", type=int, metavar="N",
                    help="override qk_rope_head_dim. With --rope, a slice "
                         "wider than the build's WASTE_MAX_ROPE_HALF pair "
                         "table has to be refused at load, not run unrotated")
    ap.add_argument("--nope", metavar="JSON",
                    help="write mla_use_nope with this JSON value rather than "
                         "omitting the key. `false` is the same model, and a "
                         "loader that tests for presence reads it as NoPE and "
                         "skips the rotation; `1` or `\"true\"` say nothing a "
                         "loader may act on, and have to be refused")
    ap.add_argument("--rope-scaling", choices=("null", "empty", "drop", "notype"),
                    metavar="SHAPE",
                    help="replace the YaRN block: null | empty ({}) | drop "
                         "(no key) all mean no scaling and must load as plain "
                         "RoPE; notype is an object with a factor and no type, "
                         "which must be refused")
    ap.add_argument("--rope-type", metavar="T",
                    help="override rope_scaling.type, e.g. linear — a scaling "
                         "the engine does not implement has to be refused, not "
                         "quietly run as plain RoPE")
    ap.add_argument("--mscale", type=float, metavar="X",
                    help="override rope_scaling.mscale, leaving mscale_all_dim "
                         "at 1.0. Unequal mscales put a ratio on cos/sin that "
                         "the engine does not apply, so it refuses instead")
    ap.add_argument("--qwen", action="store_true",
                    help="a tiny Qwen3.8-Flash-Next text fixture: packed-MoE "
                         "WEXP banks, GDN/QSA/HC names, 16 on-disk PLE heads")
    args = ap.parse_args()
    if args.index_bits == 6:
        # The engine validates index_bits 6 only as 4 stages of 64 entries
        # (src/model.c), so there is no other combination to write. Both
        # constants are read by the writers below; reassigning them here is
        # the whole of the switch.
        global STAGES, CB_ENTRIES, PACKED, INDEX_BITS
        STAGES, CB_ENTRIES, PACKED, INDEX_BITS = 4, 64, True, 6
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.qwen:
        return write_qwen_container(args, rng)

    cfg = dict(CFG)
    if args.glm:
        if args.rope:
            ap.error("--glm and --rope are different models")
        cfg.update(GLM)
        if args.index_topk:
            cfg["index_topk"] = args.index_topk
    elif args.index_topk:
        ap.error("--index-topk needs --glm")
    if args.vision and not args.glm:
        ap.error("--vision is GLM's tower; K3's is not generated here")
    if args.rope:
        # Dropping linear_attn_config is what makes every layer MLA, so the
        # rotation is exercised at depth rather than in the one full-attention
        # layer the Kimi mix leaves. It also makes the container readable by
        # tools/deepseek_ref.py, which — unlike kimi_ref.py — does not index
        # linear_attn_config and does apply the rotary.
        del cfg["mla_use_nope"], cfg["linear_attn_config"]
        cfg["model_type"] = "deepseek_v3"
        cfg["architectures"] = ["DeepseekV3ForCausalLM"]
        cfg.update(V3_ROPE)
        if args.nope is not None:
            cfg["mla_use_nope"] = json.loads(args.nope)
        if args.rope_type:
            cfg["rope_scaling"] = dict(cfg["rope_scaling"], type=args.rope_type)
        if args.mscale is not None:
            cfg["rope_scaling"] = dict(cfg["rope_scaling"], mscale=args.mscale)
        if args.rope_scaling == "null":
            cfg["rope_scaling"] = None
        elif args.rope_scaling == "empty":
            cfg["rope_scaling"] = {}
        elif args.rope_scaling == "drop":
            del cfg["rope_scaling"]
        elif args.rope_scaling == "notype":
            cfg["rope_scaling"] = {"factor": 40.0, "beta_fast": 1.0, "beta_slow": 1.0}
    if args.qk_rope:
        cfg["qk_rope_head_dim"] = args.qk_rope
    if args.tokenizer:
        # Every special has to be a real row of the embedding table and the
        # head: a container whose vocab_size stops short of its own specials
        # is one where tokenizing a chat template indexes out of the model.
        vocab = write_tokenizer(args.out)
        cfg["vocab_size"] = vocab
        cfg["bos_token_id"] = vocab - len(SPECIALS)
        cfg["eos_token_id"] = vocab - len(SPECIALS) + 2   # <|end_of_msg|>
    hid = cfg["hidden_size"]
    nh, vh = cfg["num_attention_heads"], cfg["v_head_dim"]
    qd = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    kvl, rope = cfg["kv_lora_rank"], cfg["qk_rope_head_dim"]
    moe, dense = cfg["moe_intermediate_size"], cfg["intermediate_size"]
    kda = {l - 1 for l in cfg.get("linear_attn_config", {}).get("kda_layers", [])}

    hc = cfg.get("hc_mult", 0)
    q_lora = cfg.get("q_lora_rank") or 0

    t = Trunk(rng, args.prefix)
    t.quant("model.embed_tokens.weight", [cfg["vocab_size"], hid])
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        t.f32(p + "input_layernorm.weight", [hid])
        t.f32(p + "post_attention_layernorm.weight", [hid])
        for site in ("attn", "ffn") if hc else ():
            # (2 + H) * H mixing logits from the H * hidden flattened
            # streams: pre, post, and the H x H combine matrix.
            t.quant(p + f"hc_{site}_fn", [(2 + hc) * hc, hc * hid], bits=8)
            # base is what the logits are offset by and scale is the three
            # gains on them; both are tiny, so f32 as the converter leaves
            # anything under 65536 elements.
            t.f32(p + f"hc_{site}_base", [(2 + hc) * hc])
            t.f32(p + f"hc_{site}_scale", [3])
        if L in kda:
            a = p + "self_attn."
            for w in ("q_proj", "k_proj", "v_proj"):
                t.quant(a + w + ".weight", [C_KDA, hid])
            t.quant(a + "b_proj.weight", [H_KDA, hid])
            t.quant(a + "f_a_proj.weight", [D_KDA, hid])
            t.quant(a + "f_b_proj.weight", [C_KDA, D_KDA])
            t.quant(a + "g_a_proj.weight", [D_KDA, hid])
            t.quant(a + "g_b_proj.weight", [C_KDA, D_KDA])
            t.quant(a + "o_proj.weight", [hid, C_KDA])
            for w in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t.f32(a + w + ".weight", [C_KDA, 1, 4])
            t.f32(a + "A_log", [1, 1, H_KDA, 1])
            t.f32(a + "dt_bias", [C_KDA])
            t.f32(a + "o_norm.weight", [D_KDA])
        else:
            a = p + "self_attn."
            if q_lora:
                t.quant(a + "q_a_proj.weight", [q_lora, hid])
                t.f32(a + "q_a_layernorm.weight", [q_lora])
                t.quant(a + "q_b_proj.weight", [nh * qd, q_lora])
            else:
                t.quant(a + "q_proj.weight", [nh * qd, hid])
            t.quant(a + "kv_a_proj_with_mqa.weight", [kvl + rope, hid])
            t.f32(a + "kv_a_layernorm.weight", [kvl])
            t.quant(a + "kv_b_proj.weight", [nh * (cfg["qk_nope_head_dim"] + vh), kvl])
            t.quant(a + "o_proj.weight", [hid, nh * vh])
            if cfg.get("index_topk"):
                ix, idm = a + "indexer.", cfg["index_head_dim"]
                ih, kp = cfg["index_n_heads"], cfg["index_kpool"]
                t.quant(ix + "wq_b.weight", [ih * idm, q_lora])
                t.quant(ix + "wk.weight", [idm, hid])
                # a LayerNorm, so it has a bias as well as a gain
                t.f32(ix + "k_norm.weight", [idm])
                t.f32(ix + "k_norm.bias", [idm])
                t.quant(ix + "weights_proj.weight", [ih, hid])
                t.quant(ix + "index_kpool_compress_gate", [idm, hid])
                t.f32(ix + "index_kpool_compress_ape", [kp, idm])
        if L < cfg["first_k_dense_replace"]:
            t.quant(p + "mlp.gate_proj.weight", [dense, hid])
            t.quant(p + "mlp.up_proj.weight", [dense, hid])
            t.quant(p + "mlp.down_proj.weight", [hid, dense])
        else:
            m = p + "block_sparse_moe."
            t.quant(m + "gate.weight", [cfg["num_experts"], hid])
            t.f32(m + "gate.e_score_correction_bias", [cfg["num_experts"]])
            sh = moe * cfg["num_shared_experts"]
            t.quant(m + "shared_experts.gate_proj.weight", [sh, hid])
            t.quant(m + "shared_experts.up_proj.weight", [sh, hid])
            t.quant(m + "shared_experts.down_proj.weight", [hid, sh])
    if args.vision:
        # A GLM tower at test scale: the same 25 tensor kinds the release
        # has, at dimensions that make an encode take milliseconds. The
        # shapes are what the engine branches on — a bias where the release
        # has one, a per-head q/k norm, a Conv3d patch embed flattened to
        # two dimensions as the converter flattens it — and the weights are
        # noise, exactly as everywhere else here.
        v = VISION
        vd, vh_, vi = v["hidden_size"], v["num_heads"], v["intermediate_size"]
        vo, vp = v["out_hidden_size"], v["projection_intermediate_size"]
        npix = 3 * v["temporal_patch_size"] * v["patch_size"] ** 2
        t.f32("vision_tower.patch_embed.proj.weight", [vd, npix], prefixed=False)
        t.f32("vision_tower.patch_embed.proj.bias", [vd], prefixed=False)
        for b in range(v["depth"]):
            a = f"vision_tower.blocks.{b}."
            t.f32(a + "norm1.weight", [vd], prefixed=False)
            t.f32(a + "norm2.weight", [vd], prefixed=False)
            t.f32(a + "attn.qkv.weight", [3 * vd, vd], prefixed=False)
            t.f32(a + "attn.qkv.bias", [3 * vd], prefixed=False)
            t.f32(a + "attn.q_norm.weight", [vd // vh_], prefixed=False)
            t.f32(a + "attn.k_norm.weight", [vd // vh_], prefixed=False)
            t.f32(a + "attn.proj.weight", [vd, vd], prefixed=False)
            t.f32(a + "attn.proj.bias", [vd], prefixed=False)
            for w in ("gate_proj", "up_proj"):
                t.f32(a + f"mlp.{w}.weight", [vi, vd], prefixed=False)
                t.f32(a + f"mlp.{w}.bias", [vi], prefixed=False)
            t.f32(a + "mlp.down_proj.weight", [vd, vi], prefixed=False)
            t.f32(a + "mlp.down_proj.bias", [vd], prefixed=False)
        t.f32("vision_tower.post_layernorm.weight", [vd], prefixed=False)
        m2 = v["spatial_merge_size"]
        t.f32("vision_tower.downsample.weight", [vo, vd * m2 * m2], prefixed=False)
        t.f32("vision_tower.downsample.bias", [vo], prefixed=False)
        t.f32("vision_tower.merger.proj.weight", [vo, vo], prefixed=False)
        t.f32("vision_tower.merger.post_projection_norm.weight", [vo], prefixed=False)
        t.f32("vision_tower.merger.post_projection_norm.bias", [vo], prefixed=False)
        t.f32("vision_tower.merger.gate_proj.weight", [vp, vo], prefixed=False)
        t.f32("vision_tower.merger.up_proj.weight", [vp, vo], prefixed=False)
        t.f32("vision_tower.merger.down_proj.weight", [vo, vp], prefixed=False)
    t.f32("model.norm.weight", [hid])
    t.quant("lm_head.weight", [cfg["vocab_size"], hid])
    if args.prefix:
        # One tensor outside the prefix, which is what the loader declines
        # to load: it sets on_disk and continues before `group` is
        # assigned, and the row-scratch sizing then divided by that zero.
        # arm64's sdiv answers 0 and x86's idiv raises #DE, so K3 was an
        # instant SIGFPE on every x86 build while this suite stayed green
        # — a prefix-less container has no tensor that takes the skip.
        # Naming it vision_tower is K3's case; any name outside the prefix
        # reproduces it. issue #10.
        t.quant("vision_tower.encoder.layers.0.fc0.weight", [128, hid],
                prefixed=False)
    with open(os.path.join(args.out, "trunk.bin"), "wb") as f:
        f.write(t.buf)

    # one codebook per (kind, stage) per MoE layer
    shapes = [(moe, hid), (moe, hid), (hid, moe)]      # gate, up, down
    moe_layers = [L for L in range(cfg["num_hidden_layers"])
                  if L >= cfg["first_k_dense_replace"]]
    layers, cb_base = {}, 0
    with open(os.path.join(args.out, "codebooks.bin"), "wb") as cf:
        for L in moe_layers:
            for ki in range(len(KINDS)):
                for si in range(STAGES):
                    cid = cb_base + ki * STAGES + si
                    cf.write(struct.pack("<IHBBII", MAGIC_CODEBOOK,
                                         cid & 0xFFFF, FMT_VQ3R, VEC_DIM,
                                         CB_ENTRIES, 0))
                    cf.write(f16([rng.uniform(-0.3, 0.3)
                                  for _ in range(CB_ENTRIES * VEC_DIM)]))
            name = f"experts-L{L}.bin"
            with open(os.path.join(args.out, name), "wb") as bf:
                total = sum(write_expert(bf, L, e, cb_base, shapes, rng)
                            for e in range(cfg["num_experts"]))
            layers[str(L)] = {"file": name, "experts": cfg["num_experts"],
                              "bytes": total, "codebook_base": cb_base}
            cb_base += len(KINDS) * STAGES

    # index_bits is absent from every container written before VQ4P and the
    # engine reads the absence as 8 (src/model.c), so the default container
    # keeps omitting the key — adding it would change manifest bytes that
    # the rotary fixture hashes. Same for fmt and bits_per_weight, which at
    # index_bits 8 stay exactly what this script wrote before the flag
    # existed.
    eq = {"fmt": "VQ3R", "stages": STAGES, "vec_dim": VEC_DIM,
          "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
          "bits_per_weight": STAGES}
    if PACKED:
        eq["fmt"] = "VQ4P"
        eq["index_bits"] = INDEX_BITS
        # stages*index_bits/vec_dim = 3.00; 4*6/8, same as convert.py
        eq["bits_per_weight"] = STAGES * INDEX_BITS / VEC_DIM

    manifest = {
        "format_version": 0,
        "arch": cfg["model_type"],
        "tensor_prefix": args.prefix,
        "config": cfg,
        "expert_quant": eq,
        "layers": layers,
        "trunk": t.index,
    }
    # See the note beside specials.json above: this container is hashed by
    # tests/run.sh to date the rotary fixture, so a CRLF manifest reads as
    # "regenerate me" on Windows against a fixture that is perfectly good.
    with open(os.path.join(args.out, "manifest.json"), "w",
              newline="\n") as f:
        json.dump(manifest, f, indent=1)

    if args.vision:
        vj = dict(VISION)
        vj["tower"] = "glm5-next"
        vj["media_placeholder_token_id"] = cfg["vocab_size"] - 1
        vj["max_patches"] = 64
        vj["min_image_tokens"] = 1
        vj["image_mean"] = [0.48145466, 0.4578275, 0.40821073]
        vj["image_std"] = [0.26862954, 0.26130258, 0.27577711]
        with open(os.path.join(args.out, "vision.json"), "w",
                  newline="\n") as f:
            json.dump(vj, f, indent=1)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"wrote {args.out}: {cfg['num_hidden_layers']} layers, "
          f"{cfg['num_experts']} experts, {total / (1 << 20):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
