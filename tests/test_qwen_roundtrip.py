#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""Qwen conversion round-trip on a tiny packed+PLE source.

Needs torch. Exit 77 if it is missing, matching tests/run.sh.

  python3 tests/test_qwen_roundtrip.py
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed")
    raise SystemExit(77)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tensor_bytes(t):
    """Raw little-endian payload. uv's `--with torch` env has no NumPy."""
    t = t.detach().cpu().contiguous()
    n = t.numel() * t.element_size()
    beg = t.storage_offset() * t.element_size()
    return bytes(t.untyped_storage())[beg:beg + n]


def write_st(path, tensors):
    """tensors: name -> (dtype_str, tensor)."""
    header, payload = {}, bytearray()
    for name, (dtype, t) in tensors.items():
        raw = t.detach().cpu().contiguous()
        if dtype == "BF16" and raw.dtype != torch.bfloat16:
            raise TypeError(f"{name} labeled BF16 but tensor is {raw.dtype}")
        blob = tensor_bytes(raw)
        start = len(payload)
        payload.extend(blob)
        header[name] = {"dtype": dtype, "shape": list(raw.shape),
                        "data_offsets": [start, len(payload)]}
    hb = json.dumps(header, separators=(",", ":")).encode()
    while (8 + len(hb)) % 8:
        hb += b" "
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(payload)


def main():
    tmp = tempfile.mkdtemp(prefix="qwen-rt-")
    src, out = os.path.join(tmp, "src"), os.path.join(tmp, "out")
    os.makedirs(src)
    try:
        E, I, H = 2, 8, 16
        torch.manual_seed(0)
        gate_up = torch.randn(E, 2 * I, H)
        down = torch.randn(E, H, I)
        embed = torch.randn(32, H)
        gate = torch.randn(E, H)
        offsets = torch.tensor([i * 8 for i in range(16)], dtype=torch.int64)
        sizes = torch.tensor([8] * 16, dtype=torch.int64)
        mult = torch.tensor([3, 5, 7], dtype=torch.int64)
        shards = {i: torch.randn(8, 160) for i in range(16)}

        mix = {
            "model.language_model.layers.0.mlp.experts.down_proj": ("BF16", down.to(torch.bfloat16)),
            "model.language_model.layers.0.mlp.gate.weight": ("BF16", gate.to(torch.bfloat16)),
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.shard_0.weight": ("BF16", shards[0].to(torch.bfloat16)),
            "model.language_model.embed_tokens.weight": ("BF16", embed.to(torch.bfloat16)),
        }
        exp = {
            "model.language_model.layers.0.mlp.experts.gate_up_proj":
                ("BF16", gate_up.to(torch.bfloat16)),
        }
        ple_rest = {}
        for i in range(1, 16):
            ple_rest[f"model.language_model.layers.1.ple.ple_embedding."
                     f"ngram_embedding.shard_{i}.weight"] = (
                         "BF16", shards[i].to(torch.bfloat16))
        ple_rest["model.language_model.layers.1.ple.ple_embedding."
                 "ngram_heads_offsets"] = ("I64", offsets)
        ple_rest["model.language_model.layers.1.ple.ple_embedding."
                 "ngram_heads_vocab_sizes"] = ("I64", sizes)
        ple_rest["model.language_model.layers.1.ple.ple_embedding."
                 "layer_multipliers"] = ("I64", mult)
        skip = {
            "mtp.fc_hidden.weight": ("BF16", torch.randn(H, H).to(torch.bfloat16)),
            "model.visual.pos_embed.weight": ("BF16", torch.randn(4, H).to(torch.bfloat16)),
        }
        write_st(os.path.join(src, "shard-mix.safetensors"), mix)
        write_st(os.path.join(src, "shard-exp.safetensors"), exp)
        write_st(os.path.join(src, "shard-ple.safetensors"), ple_rest)
        write_st(os.path.join(src, "shard-skip.safetensors"), skip)
        wm = {}
        for group, fn in ((mix, "shard-mix.safetensors"),
                          (exp, "shard-exp.safetensors"),
                          (ple_rest, "shard-ple.safetensors"),
                          (skip, "shard-skip.safetensors")):
            for n in group:
                wm[n] = fn
        json.dump({"weight_map": wm},
                  open(os.path.join(src, "model.safetensors.index.json"), "w"))
        json.dump({
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 1,
                "num_experts": E,
                "num_experts_per_tok": 1,
                "hidden_size": H,
                "moe_intermediate_size": I,
                "vocab_size": 32,
            },
            "vision_config": {"depth": 1},
        }, open(os.path.join(src, "config.json"), "w"))
        open(os.path.join(src, "chat_template.jinja"), "w").write("TOOLS")

        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "1"
        cmd = [sys.executable, os.path.join(REPO, "tools", "convert.py"),
               "--src", src, "--out", out, "--jobs", "1", "--layers", "0",
               "--cb-sample", "2", "--entries", "16", "--device", "cpu",
               "--reclaim", "dry"]
        print("RUN", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            print("FAIL convert returned", r.returncode)
            return 1
        man = json.load(open(os.path.join(out, "manifest.json")))
        assert man["format_version"] == 0, man["format_version"]
        assert man["arch"] == "qwen4_exp_text", man["arch"]
        assert man["tensor_prefix"] == "", man["tensor_prefix"]
        names = [t["name"] for t in man["trunk"]]
        assert "model.embed_tokens.weight" in names, names
        assert not any(n.startswith("mtp.") or "visual" in n for n in names), names
        assert not any("language_model" in n for n in names), names
        heads = [n for n in names if "ngram_head." in n]
        assert len(heads) == 16, heads
        magic = struct.unpack_from("<I", open(os.path.join(out, "experts-L0.bin"), "rb").read(4))[0]
        assert magic == 0x50584557, hex(magic)
        cfg = man["config"]
        assert cfg["num_experts_per_token"] == 1, cfg.get("num_experts_per_token")
        assert cfg["ple_head_offsets"] == [i * 8 for i in range(16)]
        assert os.path.exists(os.path.join(src, "shard-mix.safetensors")), "dry deleted mix"
        assert os.path.exists(os.path.join(src, "shard-skip.safetensors")), "dry deleted skip"
        assert not os.path.exists(os.path.join(out, "vision.json")), "vision.json must not be installed"
        # The template is metadata, not a code path: the engine has no
        # Jinja renderer and nothing reads it. Carried like every other
        # release's, so `waste info` on the container can show what the
        # model was trained to be addressed with.
        assert os.path.exists(os.path.join(out, "chat_template.jinja")), \
            "chat_template.jinja is carried like any other release's"
        if "would free" not in r.stdout:
            print("FAIL reclaim dry printed no 'would free'")
            return 1
        print("ROUNDTRIP OK")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
