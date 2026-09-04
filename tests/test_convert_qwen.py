#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""test_convert_qwen.py — Qwen3.8-Flash-Next conversion helpers.

The packed expert tensors, the nested text_config, the language_model
prefix, and the 128-to-16 PLE split are the reasons convert.py cannot
treat this checkpoint as another Kimi family member. This file pins those
rules without torch and without a 360 GB conversion.

  python3 tests/test_convert_qwen.py
"""
import json
import os
import shutil
import struct
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

# Import after the resume stubs so convert.py can load without torch.
import test_convert_resume as H          # noqa: E402
CONV = H.CONV

fails = 0


def ck(cond, what):
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def qwen_cfg():
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 640,
            "shared_expert_intermediate_size": 640,
            "vocab_size": 248320,
            "ngram_vocab_size_base": 20000000,
            "split_ngram_parts": 128,
            "heads_per_ngram": 8,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
        },
        "vision_config": {"depth": 27, "hidden_size": 1152},
    }


def main():
    print("detect and flatten qwen4_exp_text")
    prefix, src_pfx, cfg = CONV.source_prefixes(qwen_cfg())
    ck(CONV.is_qwen(cfg), "is_qwen recognises the flattened config")
    ck(CONV.is_qwen(qwen_cfg()), "is_qwen recognises the raw config too")
    ck(not CONV.is_qwen({"model_type": "kimi_linear"}), "Kimi is not Qwen")
    ck(src_pfx == "model.language_model.",
       f"experts are found under model.language_model. ({src_pfx!r})")
    ck(cfg.get("model_type") == "qwen4_exp_text",
       f"flattened model_type is qwen4_exp_text ({cfg.get('model_type')!r})")
    ck(cfg.get("num_hidden_layers") == 48, "text_config fields are promoted")
    ck("vision_config" in (cfg.get("_outer") or {}),
       "vision_config stays under _outer, not in the text config")
    ck(prefix == "", f"Qwen writes tensors under model.*, prefix={prefix!r}")

    print("normalise num_experts_per_tok")
    n = CONV.normalise_cfg(cfg)
    ck(n.get("num_experts_per_token") == 10,
       f"num_experts_per_tok becomes num_experts_per_token ({n.get('num_experts_per_token')})")
    ck(n.get("num_experts") == 512, "num_experts is already canonical")

    print("strip model.language_model. to model.")
    ck(CONV.qwen_rename("model.language_model.layers.0.linear_attn.A_log")
       == "model.layers.0.linear_attn.A_log",
       "layer tensors lose language_model")
    ck(CONV.qwen_rename("model.language_model.embed_tokens.weight")
       == "model.embed_tokens.weight",
       "embeddings lose language_model")
    ck(CONV.qwen_rename("lm_head.weight") == "lm_head.weight",
       "lm_head is already unprefixed")

    print("text-only exclusions")
    ck(CONV.qwen_skip_tensor("mtp.layers.0.mlp.gate.weight"), "skip mtp.*")
    ck(CONV.qwen_skip_tensor("model.visual.blocks.0.attn.qkv.weight"),
       "skip model.visual.*")
    ck(not CONV.qwen_skip_tensor(
        "model.language_model.layers.0.mlp.gate.weight"),
       "keep text tensors")
    ck(CONV.qwen_skip_tensor(
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_0.weight") is False,
       "PLE ngram shards are not a skip — they have their own consumer")

    print("what build_trunk drops")
    drop = CONV.qwen_drop_trunk()
    ck(drop("mtp.layers.0.mlp.gate.weight"), "the MTP layer has no reader")
    ck(drop("model.visual.blocks.0.attn.qkv.weight"), "the tower is not carried")
    ck(drop("model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight"),
       "n-gram shards are build_ple's, not build_trunk's")
    ck(drop("model.language_model.layers.1.ple.ple_embedding."
            "ngram_heads_offsets"),
       "the i64 tables go to the manifest, not the trunk")
    ck(not drop("model.language_model.layers.0.self_attn.q_proj.weight"),
       "ordinary text tensors are kept")

    print("packed expert shapes")
    # The invariant, not the release's numbers: gate_up [E, 2I, H] beside
    # down [E, H, I]. A fixture at E=2, I=8, H=16 converts on the same path.
    ck(CONV.packed_shapes_ok((512, 1280, 2560), (512, 2560, 640), 512),
       "the pinned Flash-Next pair is a valid packed layout")
    ck(CONV.packed_shapes_ok((2, 16, 16), (2, 16, 8), 2),
       "a tiny fixture with the same layout is accepted")
    ck(not CONV.packed_shapes_ok((512, 1281, 2560), (512, 2560, 640), 512),
       "an odd 2I cannot be split into gate and up")
    ck(not CONV.packed_shapes_ok((512, 1280, 2560), (512, 2560, 641), 512),
       "down's I must be gate_up's I")
    ck(not CONV.packed_shapes_ok((512, 1280, 2560), (511, 2560, 640), 512),
       "both tensors must hold every expert")
    ck(not CONV.packed_shapes_ok((512, 1280), (512, 2560, 640), 512),
       "a two-dimensional gate_up is not a packed layer")
    gate, up = CONV.split_packed_gate_up_shape((512, 1280, 2560), 640)
    ck(gate == (512, 640, 2560) and up == (512, 640, 2560),
       f"gate/up split along dim 1 → {gate} {up}")

    print("packed expert source names")
    g, d = CONV.qwen_packed_names("model.language_model.", 0)
    ck(g == "model.language_model.layers.0.mlp.experts.gate_up_proj",
       f"gate_up name {g}")
    ck(d == "model.language_model.layers.0.mlp.experts.down_proj",
       f"down name {d}")

    print("PLE 128 shards → 16 heads")
    offsets = [0, 20000003, 40000026]
    sizes = [20000003, 20000023, 20000033]
    slices = CONV.ple_head_slices(offsets, sizes)
    ck(slices[0] == (0, 20000003), f"head 0 slice {slices[0]}")
    ck(slices[1] == (20000003, 20000023), f"head 1 slice {slices[1]}")
    ck(CONV.ple_source_loc(0, 2500012) == (0, 0), "row 0 is shard 0 local 0")
    ck(CONV.ple_source_loc(2500012, 2500012) == (1, 0),
       "row 2500012 is shard 1 local 0")
    ck(CONV.ple_source_loc(20000002, 2500012) == (7, 2499918),
       "last row of head 0 lands in shard 7")
    ck(CONV.PLE_HEADS == 16 and CONV.PLE_HEAD_WIDTH == 160,
       "16 logical heads of width 160")
    ck(sizes[0] == 20000003, "head 0 rows are the first prime after 20M")

    print("ShardDebt consumers")

    def consumer(name):
        return CONV.ShardDebt.consumer(name, CONV.qwen_skip_tensor)

    ck(consumer(
        "model.language_model.layers.3.mlp.experts.gate_up_proj")
       == ("layer", 3),
       "packed experts belong to their layer")
    ck(consumer(
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_4.weight")
       == CONV.ShardDebt.PLE,
       "PLE ngram shards are a separate consumer")
    ck(consumer(
        "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets")
       == CONV.ShardDebt.PLE,
       "I64 head offsets are a PLE consumer — build_ple reads them after trunk")
    ck(consumer(
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_heads_vocab_sizes")
       == CONV.ShardDebt.PLE,
       "I64 vocab sizes are a PLE consumer")
    ck(consumer(
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers")
       == CONV.ShardDebt.PLE,
       "I64 layer_multipliers are a PLE consumer")
    ck(consumer(
        "model.language_model.layers.1.ple.key_proj.weight")
       == CONV.ShardDebt.TRUNK,
       "PLE projections stay on the trunk")
    ck(consumer("mtp.layers.0.mlp.experts.gate_up_proj")
       == CONV.ShardDebt.SKIP,
       "mtp experts are not layer 0")
    ck(consumer("model.visual.patch_embed.proj.weight")
       == CONV.ShardDebt.SKIP,
       "vision tensors do not hold a text consumer")
    ck(consumer("model.language_model.embed_tokens.weight")
       == CONV.ShardDebt.TRUNK,
       "embeddings are trunk")

    print("a checkpoint that carries its tower keeps it")
    # `model.visual.` is also how GLM spells its tower, and GLM does carry
    # it. Without a skip predicate nothing is a SKIP.
    ck(CONV.ShardDebt.consumer("model.visual.patch_embed.proj.weight")
       == CONV.ShardDebt.TRUNK,
       "with no skip predicate the tower is a trunk tensor")

    print("Kimi prefixes are unchanged")
    k3 = {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {"model_type": "kimi_linear", "num_hidden_layers": 2},
    }
    prefix, src_pfx, cfg = CONV.source_prefixes(k3)
    ck(not CONV.is_qwen(cfg), "K3 is not Qwen")
    ck(prefix == "language_model.", f"K3 prefix stays language_model. ({prefix!r})")
    ck(src_pfx == "language_model.model.", f"K3 src prefix stays ({src_pfx!r})")

    print("--reclaim dry keeps a PLE shard until PLE finishes")
    tmp = tempfile.mkdtemp(prefix="qwen-debt-")
    try:
        wm = {
            "model.language_model.layers.0.mlp.gate.weight": "shard-mix.safetensors",
            "model.language_model.layers.0.mlp.experts.gate_up_proj":
                "shard-mix.safetensors",
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight": "shard-mix.safetensors",
            "mtp.fc_hidden.weight": "shard-mtp.safetensors",
            "model.visual.pos_embed.weight": "shard-vis.safetensors",
        }
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        for shard in ("shard-mix.safetensors", "shard-mtp.safetensors",
                      "shard-vis.safetensors"):
            open(os.path.join(src, shard), "wb").write(b"\0" * 64)
        debt = CONV.ShardDebt(wm, src, CONV.qwen_skip_tensor)
        CONV.reclaim(debt, "dry", CONV.ShardDebt.SKIP, "excluded")
        CONV.reclaim(debt, "dry", CONV.ShardDebt.TRUNK, "trunk")
        CONV.reclaim(debt, "dry", ("layer", 0), "layer 0")
        still = sorted(debt.owed)
        ck(still == ["shard-mix.safetensors"],
           f"mix shard still held after trunk+layer, skip shards gone ({still})")
        CONV.reclaim(debt, "dry", CONV.ShardDebt.PLE, "ple")
        ck(list(debt.owed) == [], "PLE was the last consumer of the mix shard")
        ck(os.path.exists(os.path.join(src, "shard-mix.safetensors")),
           "dry deletes nothing")

        # Offsets live in their own shard on some layouts. build_ple reads
        # them after the trunk reclaim, so TRUNK must not be their consumer.
        open(os.path.join(src, "shard-i64.safetensors"), "wb").write(b"\0" * 64)
        open(os.path.join(src, "shard-gate.safetensors"), "wb").write(b"\0" * 64)
        debt2 = CONV.ShardDebt({
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_heads_offsets": "shard-i64.safetensors",
            "model.language_model.layers.0.mlp.gate.weight":
                "shard-gate.safetensors",
        }, src, CONV.qwen_skip_tensor)
        CONV.reclaim(debt2, "dry", CONV.ShardDebt.TRUNK, "trunk")
        ck("shard-i64.safetensors" in debt2.owed,
           f"I64 shard still held after trunk ({sorted(debt2.owed)})")
        CONV.reclaim(debt2, "dry", CONV.ShardDebt.PLE, "ple")
        ck("shard-i64.safetensors" not in debt2.owed,
           "PLE release frees the I64 shard")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("PLE reclaim needs 16 ngram_head tensors in the trunk index")
    def head(h):
        return {"name": f"model.layers.1.ple.ple_embedding.ngram_head.{h}.weight"}
    ck(not CONV.ple_heads_written(None), "absent index is incomplete")
    ck(not CONV.ple_heads_written([]), "empty index is incomplete")
    ck(not CONV.ple_heads_written([head(h) for h in range(15)]),
       "15 heads are incomplete")
    ck(CONV.ple_heads_written([head(h) for h in range(16)]),
       "16 heads are complete")
    ck(not CONV.ple_heads_written([head(0)] * 16),
       "sixteen copies of head 0 are not 16 heads")

    tmp = tempfile.mkdtemp(prefix="qwen-ple-reclaim-")
    try:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        open(os.path.join(src, "shard-ple.safetensors"), "wb").write(b"\0" * 64)
        wm = {
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight": "shard-ple.safetensors",
        }
        debt = CONV.ShardDebt(wm, src, CONV.qwen_skip_tensor)
        CONV.reclaim_ple_if_complete(debt, "dry", [])
        ck("shard-ple.safetensors" in debt.owed,
           "skip-trunk with no heads retains the PLE shard")
        ck(os.path.exists(os.path.join(src, "shard-ple.safetensors")),
           "incomplete dry deletes nothing")
        CONV.reclaim_ple_if_complete(debt, "dry", [head(h) for h in range(15)])
        ck("shard-ple.safetensors" in debt.owed,
           "15 published heads still retain the PLE shard")
        CONV.reclaim_ple_if_complete(debt, "dry", [head(h) for h in range(16)])
        ck("shard-ple.safetensors" not in debt.owed,
           "16 published heads permit dry PLE reclaim")
        ck(os.path.exists(os.path.join(src, "shard-ple.safetensors")),
           "complete dry still deletes nothing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("Qwen packed default jobs is 1 unless --jobs is set")
    ck(CONV.resolve_jobs(True, None) == 1, "omitted --jobs is 1 for Qwen")
    ck(CONV.resolve_jobs(False, None) == 3, "omitted --jobs stays 3 otherwise")
    ck(CONV.resolve_jobs(True, 4) == 4, "explicit --jobs 4 wins for Qwen")
    ck(CONV.resolve_jobs(False, 1) == 1, "explicit --jobs 1 wins for Kimi")

    raw = os.environ.get("QWEN_SRC",
                         "/Users/admin/mnt/llm/qwen38-flash-next/raw")
    idxp = os.path.join(raw, "model.safetensors.index.json")
    if not os.path.isfile(idxp):
        print("real source sample SKIP (no pinned checkpoint)")
    else:
        print("real source sample (headers + I64 + reclaim class, no 360 GB)")
        wm = json.load(open(idxp))["weight_map"]
        g, d = CONV.qwen_packed_names("model.language_model.", 0)

        def meta(name):
            fn = wm[name]
            with open(os.path.join(raw, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                return json.loads(f.read(hlen))[name]

        def i64(name):
            fn = wm[name]
            with open(os.path.join(raw, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
                base = 8 + hlen
                beg, end = hdr[name]["data_offsets"]
                f.seek(base + beg)
                buf = f.read(end - beg)
            n = len(buf) // 8
            return list(struct.unpack("<%dq" % n, buf))

        gm, dm = meta(g), meta(d)
        ck(gm["dtype"] == "BF16" and tuple(gm["shape"]) == CONV.QWEN_PACKED_GATE_UP,
           f"L0 gate_up {gm['dtype']} {gm['shape']}")
        ck(dm["dtype"] == "BF16" and tuple(dm["shape"]) == CONV.QWEN_PACKED_DOWN,
           f"L0 down {dm['dtype']} {dm['shape']}")

        def bf16s_at(name, byte_off, n):
            fn = wm[name]
            with open(os.path.join(raw, fn), "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
                base = 8 + hlen
                beg, _end = hdr[name]["data_offsets"]
                f.seek(base + beg + byte_off)
                buf = f.read(n * 2)
            vals = struct.unpack("<%dH" % n, buf)
            return [struct.unpack("<f", struct.pack("<I", v << 16))[0]
                    for v in vals]

        # Packed layout is [E, 2I, H]; gate is dim1 [0:I], up is dim1 [I:2I].
        hid = CONV.QWEN_PACKED_GATE_UP[2]
        inter = CONV.QWEN_PACKED_GATE_UP[1] // 2
        gate0 = bf16s_at(g, 0, 4)
        up0 = bf16s_at(g, inter * hid * 2, 4)
        ck(all(abs(x) < 1e6 for x in gate0 + up0),
           f"L0 e0 gate/up split BF16 are finite gate={gate0} up={up0}")
        down0 = bf16s_at(d, 0, 4)
        ck(all(abs(x) < 1e6 for x in down0),
           f"L0 e0 down first 4 BF16 are finite {down0}")
        off = i64("model.language_model.layers.1.ple.ple_embedding."
                  "ngram_heads_offsets")
        sz = i64("model.language_model.layers.1.ple.ple_embedding."
                 "ngram_heads_vocab_sizes")
        ck(off[0] == 0 and sz[0] == 20000003,
           f"head 0 offset/size {off[0]}/{sz[0]}")
        ck(off[1] == 20000003, f"head 1 starts at {off[1]}")
        ck(CONV.ple_source_loc(20000002, 2500012) == (7, 2499918),
           "head 0 last source row is shard 7 local 2499918")
        ck(sum(sz) == 320001446, f"16 heads sum {sum(sz)}")
        ngram0 = meta("model.language_model.layers.1.ple.ple_embedding."
                      "ngram_embedding.shard_0.weight")
        ck(tuple(ngram0["shape"]) == (2500012, 160),
           f"PLE shard 0 {ngram0['shape']}")
        mix = {consumer(n)
               for n, sh in wm.items() if sh == "model-00005-of-00131.safetensors"}
        ck(("layer", 1) in mix and CONV.ShardDebt.PLE in mix
           and CONV.ShardDebt.TRUNK in mix,
           f"model-00005 mixed consumers {mix}")
        ck(consumer("mtp.layers.0.mlp.experts.gate_up_proj")
           == CONV.ShardDebt.SKIP,
           "real mtp packed experts are SKIP, not layer 0")

    print("PLE write must stay inside RAM")
    # CONV.raw_bytes is stubbed in test_convert_resume; read the file.
    conv_py = open(os.path.join(REPO, "tools", "convert.py"),
                   encoding="utf-8").read()
    ck(".tolist" not in conv_py.split("def raw_bytes", 1)[1].split("\ndef ", 1)[0],
       "raw_bytes must not turn every byte into a Python int "
       "(a PLE head is ~3 GiB of int8; list() of that swaps)")
    ck("torch.cat(chunks" not in conv_py.split("def build_ple", 1)[1].split("\ndef ", 1)[0],
       "build_ple must not cat a whole 20M-row head before Q8G")

    print("QWEN CONVERT FAILED" if fails else "QWEN CONVERT OK")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
