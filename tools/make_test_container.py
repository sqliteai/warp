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
FMT_F32, FMT_Q8G, FMT_Q4G, FMT_VQ3R = 0, 2, 3, 4
VEC_DIM, CB_ENTRIES, STAGES, IDX_BLOCK = 8, 256, 3, 64
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


def f32(vals):
    return struct.pack("<%df" % len(vals), *vals)


def f16(vals):
    return struct.pack("<%de" % len(vals), *vals)


class Trunk:
    """Accumulates trunk.bin and its index, in the layout model.c reads:
    F32 verbatim, Q8G as int8 rows followed by one fp16 scale per group,
    Q4G the same with two weights per byte.

    Which format goes where follows tools/convert.py rather than being a
    free choice: it writes Q8G for embed_tokens and lm_head and Q4G for
    every layer weight, because --trunk-bits defaults to 4. A synthetic
    container that is Q8G throughout does not exercise the width a real
    one is almost entirely made of — which is how a Q4G trunk that could
    not be dequantized at load reached a release with CI green."""

    def __init__(self, rng):
        self.buf = bytearray()
        self.index = []
        self.rng = rng

    def f32(self, name, shape):
        n = 1
        for s in shape:
            n *= s
        off = len(self.buf)
        self.buf += f32([self.rng.uniform(-0.5, 0.5) for _ in range(n)])
        self.index.append({"name": name, "fmt": FMT_F32, "off": off,
                           "shape": list(shape), "bytes": len(self.buf) - off})

    def q8g(self, name, shape):
        rows, N = 1, shape[-1]
        for s in shape[:-1]:
            rows *= s
        ng = (N + GROUP - 1) // GROUP
        off = len(self.buf)
        # int8 payload is padded out to whole groups, as the converter does
        self.buf += bytes((self.rng.randrange(-127, 128) & 0xFF)
                          for _ in range(rows * ng * GROUP))
        soff = len(self.buf)
        self.buf += f16([self.rng.uniform(0.002, 0.02)
                         for _ in range(rows * ng)])
        self.index.append({"name": name, "fmt": FMT_Q8G, "off": off,
                           "shape": list(shape), "group": GROUP,
                           "scale_off": soff, "bytes": len(self.buf) - off})

    def q4g(self, name, shape):
        """Two weights per byte, zero point 8, one fp16 scale per group —
        the layout model.c unpacks as
            (i & 1) ? (byte >> 4) - 8 : (byte & 0x0F) - 8
        so the low nibble is the even element and the high nibble the odd
        one. Stored values are v + 8, i.e. 0..15 for v in -8..7."""
        rows, N = 1, shape[-1]
        for s in shape[:-1]:
            rows *= s
        ng = (N + GROUP - 1) // GROUP
        off = len(self.buf)
        # Padded out to whole groups, as the converter does. GROUP is even,
        # so the pairing below never runs off the end.
        vals = [self.rng.randrange(0, 16) for _ in range(rows * ng * GROUP)]
        self.buf += bytes(vals[i] | (vals[i + 1] << 4)
                          for i in range(0, len(vals), 2))
        soff = len(self.buf)
        self.buf += f16([self.rng.uniform(0.002, 0.02)
                         for _ in range(rows * ng)])
        self.index.append({"name": name, "fmt": FMT_Q4G, "off": off,
                           "shape": list(shape), "group": GROUP,
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


def write_expert(f, layer, eid, cb_base, shapes, rng):
    hdr_size = 48
    off, offsets, body = hdr_size, [], bytearray()
    for (M, N) in shapes:
        nvec = M * N // VEC_DIM
        idx = [[rng.randrange(CB_ENTRIES) for _ in range(nvec)]
               for _ in range(STAGES)]
        offsets.append(off)
        b = block_indices(idx, M, N)
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
                      MAGIC_EXPERT, layer, eid, FMT_VQ3R, 0, cb_base, 0, 0,
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
    with open(os.path.join(outdir, "specials.json"), "w") as f:
        json.dump(specials, f, indent=1)

    return base + len(SPECIALS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed, byte-identical container")
    ap.add_argument("--tokenizer", action="store_true",
                    help="also write a small tiktoken-style vocabulary with "
                         "K3's XTML specials, so the container can be driven "
                         "with text instead of raw ids")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    cfg = dict(CFG)
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
    kda = {l - 1 for l in cfg["linear_attn_config"]["kda_layers"]}

    t = Trunk(rng)
    t.q8g("model.embed_tokens.weight", [cfg["vocab_size"], hid])
    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        t.f32(p + "input_layernorm.weight", [hid])
        t.f32(p + "post_attention_layernorm.weight", [hid])
        if L in kda:
            a = p + "self_attn."
            for w in ("q_proj", "k_proj", "v_proj"):
                t.q4g(a + w + ".weight", [C_KDA, hid])
            t.q4g(a + "b_proj.weight", [H_KDA, hid])
            t.q4g(a + "f_a_proj.weight", [D_KDA, hid])
            t.q4g(a + "f_b_proj.weight", [C_KDA, D_KDA])
            t.q4g(a + "g_a_proj.weight", [D_KDA, hid])
            t.q4g(a + "g_b_proj.weight", [C_KDA, D_KDA])
            t.q4g(a + "o_proj.weight", [hid, C_KDA])
            for w in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t.f32(a + w + ".weight", [C_KDA, 1, 4])
            t.f32(a + "A_log", [1, 1, H_KDA, 1])
            t.f32(a + "dt_bias", [C_KDA])
            t.f32(a + "o_norm.weight", [D_KDA])
        else:
            a = p + "self_attn."
            t.q4g(a + "q_proj.weight", [nh * qd, hid])
            t.q4g(a + "kv_a_proj_with_mqa.weight", [kvl + rope, hid])
            t.f32(a + "kv_a_layernorm.weight", [kvl])
            t.q4g(a + "kv_b_proj.weight", [nh * (cfg["qk_nope_head_dim"] + vh), kvl])
            t.q4g(a + "o_proj.weight", [hid, nh * vh])
        if L < cfg["first_k_dense_replace"]:
            t.q4g(p + "mlp.gate_proj.weight", [dense, hid])
            t.q4g(p + "mlp.up_proj.weight", [dense, hid])
            t.q4g(p + "mlp.down_proj.weight", [hid, dense])
        else:
            m = p + "block_sparse_moe."
            t.q4g(m + "gate.weight", [cfg["num_experts"], hid])
            t.f32(m + "gate.e_score_correction_bias", [cfg["num_experts"]])
            sh = moe * cfg["num_shared_experts"]
            t.q4g(m + "shared_experts.gate_proj.weight", [sh, hid])
            t.q4g(m + "shared_experts.up_proj.weight", [sh, hid])
            t.q4g(m + "shared_experts.down_proj.weight", [hid, sh])
    t.f32("model.norm.weight", [hid])
    t.q8g("lm_head.weight", [cfg["vocab_size"], hid])
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

    manifest = {
        "format_version": 0,
        "arch": cfg["model_type"],
        "tensor_prefix": "",
        "config": cfg,
        "expert_quant": {"fmt": "VQ3R", "stages": STAGES, "vec_dim": VEC_DIM,
                         "entries": CB_ENTRIES, "index_block": IDX_BLOCK,
                         "bits_per_weight": STAGES},
        "layers": layers,
        "trunk": t.index,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"wrote {args.out}: {cfg['num_hidden_layers']} layers, "
          f"{cfg['num_experts']} experts, {total / (1 << 20):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
