#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""qwen_ref.py — official qwen4_exp dumps for the C engine.

Pins huggingface/transformers git commit fc5c5bde8e656dad91cbf34e61940d984b1c7b91
(the merge of PR 48337). Does not use mutable main.

Tiny path: builds a random Qwen4ExpForCausalLM so the 360 GB checkpoint is
never loaded. That is not a common-weight match.

Common-weight path (--src): streams only the tensors one official decoder
layer needs. It must not call from_pretrained (that would map all shards).

  uv run --python 3.12 --with torch \\
      --with 'transformers @ git+https://github.com/huggingface/transformers@fc5c5bde8e656dad91cbf34e61940d984b1c7b91' \\
      --no-project python tools/qwen_ref.py --src DIR --ids 3,7,11,5 --dump out.bin
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import struct
import sys
import traceback

TRANSFORMERS_PIN = "fc5c5bde8e656dad91cbf34e61940d984b1c7b91"
# The pinned checkpoint, wherever it was downloaded to. No default: this
# reference reads 335 GB of source shards, and a wrong guess at the path is
# a SKIP that reads like a missing feature.
PINNED_SRC = os.environ.get("WASTE_QWEN_SRC", "")
SRC_PREFIX = "model.language_model."


def skip_prereq(msg):
    print(f"SKIP_PREREQ: {msg}")
    return 77


def blocker(msg):
    print("\nBLOCKER:")
    print("  ", msg)
    return 1


class StreamingExperts:
    """Official expert apply without a [512,1280,2560] Parameter."""

    def __init__(self, config, st=None, layer=0):
        self.num_experts = getattr(config, "num_experts", 0)
        self.hidden_dim = getattr(config, "hidden_size", 0)
        self.intermediate_dim = getattr(config, "moe_intermediate_size", 0)
        self.st = st
        self.layer = layer
        self.gate_up_proj = None
        self.down_proj = None

    def parameters(self):
        return iter(())


class StreamingNGram:
    """Official n-gram ids without the 102 GiB embedding table."""

    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0,
                 st=None):
        self.st = st
        self.layer_idx = layer_idx
        self.ple_layer_index = ple_layer_index
        self.ngram_size = getattr(config, "ngram_size", 3)
        self.heads_per_ngram = getattr(config, "heads_per_ngram", 8)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.embedding_dim = embedding_dim
        eos = getattr(config, "eos_token_id", 0)
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.gate_up_proj = None

    def parameters(self):
        return iter(())


def import_official():
    try:
        import transformers
        from transformers.models.qwen4_exp.configuration_qwen4_exp import (
            Qwen4ExpTextConfig,
        )
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpForCausalLM,
            Qwen4ExpTextQSAIndexer,
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )
    except ModuleNotFoundError as e:
        name = str(e)
        if "qwen4_exp" in name:
            return None, blocker(
                f"pinned transformers has no qwen4_exp ({e}). "
                f"Need git commit {TRANSFORMERS_PIN}."
            )
        return None, skip_prereq(f"import failed: {e}")
    except Exception as e:
        return None, blocker(f"official import failed: {type(e).__name__}: {e}")
    return {
        "ver": getattr(transformers, "__version__", "?"),
        "cfg": Qwen4ExpTextConfig,
        "lm": Qwen4ExpForCausalLM,
        "indexer": Qwen4ExpTextQSAIndexer,
        "chunk": torch_chunk_gated_delta_rule,
        "recurrent": torch_recurrent_gated_delta_rule,
    }, None


def tiny_config(Cls):
    return Cls(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        hc_count=4,
        hc_lowrank=8,
        num_experts=8,
        num_experts_per_tok=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=["linear_attention", "full_attention"],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=32,
        indexer_compress_ratio=4,
        ple_layer_ids=[],
        eos_token_id=2,
        bos_token_id=1,
        max_position_embeddings=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [2, 2, 2],
            "mrope_interleaved": True,
        },
    )


def text_config_from_src(Cls, src):
    outer = json.load(open(os.path.join(src, "config.json")))
    text = dict(outer.get("text_config") or outer)
    if "num_experts_per_tok" not in text and "num_experts_per_token" in text:
        text["num_experts_per_tok"] = text["num_experts_per_token"]
    gen = os.path.join(src, "generation_config.json")
    if os.path.isfile(gen):
        g = json.load(open(gen))
        text.setdefault("eos_token_id", g.get("eos_token_id"))
        text.setdefault("bos_token_id", g.get("bos_token_id"))
        text.setdefault("pad_token_id", g.get("pad_token_id"))
    try:
        return Cls.from_dict(text)
    except Exception:
        import dataclasses
        names = {f.name for f in dataclasses.fields(Cls)}
        return Cls(**{k: v for k, v in text.items() if k in names})


def _gather_rows(st, name, ids):
    import torch
    hdr, base = st._header(st.wm[name])
    meta = hdr[name]
    shape = meta["shape"]
    rows, width = int(shape[0]), int(shape[1])
    dt = {"BF16": torch.bfloat16, "F16": torch.float16,
          "F32": torch.float32}[meta["dtype"]]
    es = torch.tensor([], dtype=dt).element_size()
    path = os.path.join(st.dir, st.wm[name])
    out = torch.empty(len(ids), width, dtype=dt)
    with open(path, "rb") as f:
        for i, rid in enumerate(ids):
            if rid < 0 or rid >= rows:
                out[i].zero_()
                continue
            f.seek(base + meta["data_offsets"][0] + rid * width * es)
            buf = bytearray(f.read(width * es))
            out[i].copy_(torch.frombuffer(buf, dtype=dt))
    return out


def _load_into(module, st, prefix, skip_sub):
    import torch
    sd = {}
    for name, p in module.named_parameters():
        if any(name.startswith(s) or s in name for s in skip_sub):
            continue
        src = prefix + name
        if st.have(src):
            t = st.raw(src)
            if tuple(t.shape) != tuple(p.shape):
                print(f"  skip {src} shape {tuple(t.shape)} vs {tuple(p.shape)}")
                continue
            sd[name] = t.to(dtype=p.dtype)
        elif st.have(src.replace(".weight", "")):
            alt = src.replace(".weight", "")
            t = st.raw(alt)
            if tuple(t.shape) == tuple(p.shape):
                sd[name] = t.to(dtype=p.dtype)
    for name, b in module.named_buffers():
        src = prefix + name
        if st.have(src) and name not in sd:
            t = st.raw(src)
            if tuple(t.shape) == tuple(b.shape):
                sd[name] = t.to(dtype=b.dtype)
    missing, unexpected = module.load_state_dict(sd, strict=False)
    miss = [m for m in missing if not any(s in m for s in skip_sub)]
    return miss, unexpected


def run_streaming(src, ids, dump_path, hidden_path, max_layers=0):
    """Official modules, one layer of source tensors at a time.

    must not call from_pretrained
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        ACT2FN,
        Qwen4ExpTextDecoderLayer,
        Qwen4ExpTextExperts,
        Qwen4ExpTextGatedResidual,
        Qwen4ExpTextNGramEmbedding,
        Qwen4ExpTextRotaryEmbedding,
        Qwen4ExpTextConfig,
    )
    try:
        from transformers.masking_utils import (
            create_causal_mask, create_recurrent_attention_mask)
    except Exception:
        create_causal_mask = create_recurrent_attention_mask = None

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import convert as C
    from mxfp4 import ST

    cfg = text_config_from_src(Qwen4ExpTextConfig, src)
    st = ST(src)
    n_layers = cfg.num_hidden_layers if max_layers <= 0 else min(
        max_layers, cfg.num_hidden_layers)
    print(f"  streaming src={src} layers={n_layers}/{cfg.num_hidden_layers} "
          f"ids={ids}")
    print(f"  config hidden={cfg.hidden_size} experts={cfg.num_experts} "
          f"top_k={cfg.num_experts_per_tok} ple={cfg.ple_layer_ids}")

    embeds = _gather_rows(st, SRC_PREFIX + "embed_tokens.weight", ids).float()
    hidden = embeds.unsqueeze(0)  # [1, T, H]
    hidden = hidden.repeat(1, 1, cfg.hc_count)
    T = hidden.shape[1]

    rotary = Qwen4ExpTextRotaryEmbedding(config=cfg)
    pos = torch.arange(T).view(1, 1, T).expand(4, 1, T)
    text_position_ids = pos[0]
    rope_pos = pos[1:]
    position_embeddings = rotary(hidden, rope_pos)

    dummy = hidden[..., :cfg.hidden_size]
    if create_causal_mask is not None:
        mask_kwargs = {
            "config": cfg,
            "inputs_embeds": dummy,
            "attention_mask": None,
            "past_key_values": None,
            "position_ids": text_position_ids,
            "allow_is_causal_skip": False,
        }
        attn_mask = create_causal_mask(**mask_kwargs)
        conv_mask = create_recurrent_attention_mask(**mask_kwargs)
    else:
        attn_mask = None
        conv_mask = None
    # Official QSA indexer requires a 4D mask (bool True = visible). The
    # masking helper can still return None; a None here is AttributeError
    # on layer 3, not a skip.
    if attn_mask is None:
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)

    class _StreamExperts(nn.Module):
        def __init__(self, config, layer):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.act_fn = ACT2FN[config.hidden_act]
            gname, dname = C.qwen_packed_names(layer)
            print(f"    L{layer} load packed experts", flush=True)
            self.gate_up = st.raw(gname)
            self.down = st.raw(dname)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(
                    top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                wgu = self.gate_up[int(expert_idx)].float()
                wd = self.down[int(expert_idx)].float()
                gate, up = F.linear(current_state, wgu).chunk(2, dim=-1)
                cur = self.act_fn(gate) * up
                cur = F.linear(cur, wd)
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, cur.to(final_hidden_states.dtype))
            return final_hidden_states

    class _StreamNGram(Qwen4ExpTextNGramEmbedding):
        def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
            nn.Module.__init__(self)
            self.layer_idx = layer_idx
            self.ngram_size = config.ngram_size
            self.context_len = self.ngram_size - 1
            self.heads_per_ngram = config.heads_per_ngram
            self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
            self.ple_layer_index = ple_layer_index
            eos = config.eos_token_id
            self.eos_token_id = eos[0] if isinstance(eos, list) else eos
            meta = C.qwen_ple_config(st, list(st.wm))
            off = meta["ple_head_offsets"]
            sz = meta["ple_head_vocab_sizes"]
            self.register_buffer(
                "layer_multipliers",
                torch.tensor(meta["ple_layer_multipliers"], dtype=torch.long))
            self.register_buffer(
                "ngram_heads_vocab_sizes",
                torch.tensor(sz, dtype=torch.long))
            self.register_buffer(
                "ngram_heads_offsets",
                torch.tensor(off, dtype=torch.long))
            self.head_dim = embedding_dim // self.ngram_heads
            self._shards = C.ple_shard_map(list(st.wm))
            sample = st.raw(self._shards[min(self._shards)])
            self._shard_rows = int(sample.shape[0])
            del sample
            self.ngram_embedding = None

        def forward(self, input_ids, past_key_values):
            input_ids = input_ids.long()
            previous_context = input_ids.new_full(
                (input_ids.shape[0], self.context_len), self.eos_token_id)
            token_history = torch.cat([previous_context, input_ids], dim=-1)
            shifted_tokens = [
                self._shift_right_ignore_eos(token_history, shift)
                for shift in range(self.ngram_size)]
            blocks = []
            for ngram in range(2, self.ngram_size + 1):
                start_idx = (ngram - 2) * self.heads_per_ngram
                end_idx = start_idx + self.heads_per_ngram
                mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
                for position in range(1, ngram):
                    mixed_ids = torch.bitwise_xor(
                        mixed_ids,
                        shifted_tokens[position] * self.layer_multipliers[position])
                head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
                head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
                ngram_ids = torch.remainder(
                    mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
                blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
            ngram_ids = torch.cat(blocks, dim=-1)[:, -input_ids.shape[1]:]
            B, TT, Hn = ngram_ids.shape
            out = torch.empty(B, TT, Hn, self.head_dim, dtype=torch.float32)
            for b in range(B):
                for t in range(TT):
                    for h in range(Hn):
                        gid = int(ngram_ids[b, t, h])
                        si, loc = C.ple_source_loc(gid, self._shard_rows)
                        row = st.raw(self._shards[si])[loc].float()
                        out[b, t, h] = row[:self.head_dim]
            return out.flatten(-2)

    hidden_chunks = []
    ids_t = torch.tensor([ids], dtype=torch.long)
    skip = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj",
            "ple.ple_embedding.ngram_embedding")

    for L in range(n_layers):
        print(f"  layer {L} {cfg.layer_types[L]}", flush=True)
        orig_e, orig_n = Qwen4ExpTextExperts, Qwen4ExpTextNGramEmbedding

        def _make_e(config, _L=L):
            return _StreamExperts(config, _L)

        def _make_n(config, embedding_dim, layer_idx, ple_layer_index=0):
            return _StreamNGram(config, embedding_dim, layer_idx, ple_layer_index)

        import transformers.models.qwen4_exp.modeling_qwen4_exp as mm
        mm.Qwen4ExpTextExperts = _make_e
        mm.Qwen4ExpTextNGramEmbedding = _make_n
        try:
            layer = Qwen4ExpTextDecoderLayer(cfg, L)
        finally:
            mm.Qwen4ExpTextExperts = orig_e
            mm.Qwen4ExpTextNGramEmbedding = orig_n
        layer.eval()
        miss, _ = _load_into(layer, st, f"{SRC_PREFIX}layers.{L}.", skip)
        if miss:
            print(f"    missing {len(miss)}: {miss[:8]}")
        with torch.no_grad():
            hidden = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                conv_mask=conv_mask,
                past_key_values=None,
                ple_input_ids=ids_t,
            )
        hidden_chunks.append(hidden[0, -1].float().contiguous().cpu())
        del layer
        gc.collect()

    mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    mixer.eval()
    _load_into(mixer, st, SRC_PREFIX + "hyper_connection_mixer.", ())
    with torch.no_grad():
        mixed = mixer(hidden)
    lm_w = st.raw("lm_head.weight").float()
    with torch.no_grad():
        logits = F.linear(mixed[0, -1].float(), lm_w)
    del lm_w, mixer
    gc.collect()

    if hidden_path:
        with open(hidden_path, "wb") as f:
            for h in hidden_chunks:
                f.write(struct.pack(f"<{h.numel()}f", *h.tolist()))
        print(f"  wrote hidden {hidden_path} ({len(hidden_chunks)} layers)")
    if dump_path:
        arr = logits.float().cpu().contiguous()
        with open(dump_path, "wb") as f:
            f.write(struct.pack(f"<{arr.numel()}f", *arr.tolist()))
        print(f"  wrote logits {dump_path} shape {tuple(arr.shape)} "
              f"argmax {int(arr.argmax())}")
    print("PASS official streaming common-weight oracle")
    return 0, logits, hidden_chunks


def tiny_oracle(mods, ids, c_bin):
    import torch
    print(f"  transformers: OK {mods['ver']} has qwen4_exp")
    try:
        cfg = tiny_config(mods["cfg"])
        model = mods["lm"](cfg)
        model.eval()
        t = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model(t)
        logits = out.logits[0, -1].float()
        print(f"  tiny model: OK logits {tuple(logits.shape)} "
              f"argmax {int(logits.argmax())}")
    except Exception as e:
        traceback.print_exc()
        return blocker(f"tiny Qwen4ExpForCausalLM failed: {type(e).__name__}: {e}")

    try:
        T, B, Hk, Hv, Dk, Dv = 4, 1, 16, 48, 128, 128
        torch.manual_seed(0)
        q = torch.randn(B, T, Hv, Dk)
        k = torch.randn(B, T, Hv, Dk)
        v = torch.randn(B, T, Hv, Dv)
        g = torch.randn(B, T, Hv)
        beta = torch.rand(B, T, Hv)
        rec, _ = mods["recurrent"](
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        chunk, _ = mods["chunk"](
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        d = (rec.float() - chunk.float()).abs().max().item()
        print(f"  GDN official 16/48/128: recurrent vs chunk max|diff| {d:.3e}")
        if d > 5e-4:
            return blocker(f"official recurrent vs chunk differ by {d}")
    except Exception as e:
        traceback.print_exc()
        return blocker(f"official GDN kernels failed: {type(e).__name__}: {e}")

    try:
        indexer = mods["indexer"](cfg, layer_idx=1)
        indexer.eval()
        rot = cfg.indexer_head_dim
        for T, want in ((3, [0, 1, 2]), (4, [0, 1, 2, 3])):
            h = torch.randn(1, T, cfg.hidden_size)
            cos = torch.ones(1, T, rot)
            sin = torch.zeros(1, T, rot)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
            with torch.no_grad():
                sel_mask = indexer(h, (cos, sin), mask, None)
            last = sel_mask[0, 0, T - 1]
            if last.dtype != torch.bool:
                last = last == 0
            got = torch.where(last)[0].tolist()
            if got != want:
                return blocker(
                    f"official QSAIndexer T={T} selected {got}, want {want}")
        print("  QSA official indexer: T=3 tail [0,1,2], T=4 block [0,1,2,3]")
    except Exception as e:
        traceback.print_exc()
        return blocker(f"official QSAIndexer failed: {type(e).__name__}: {e}")

    try:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextTopKRouter,
        )
        rcfg = tiny_config(mods["cfg"])
        rcfg.num_experts = 16
        rcfg.num_experts_per_tok = 10
        router = Qwen4ExpTextTopKRouter(rcfg)
        with torch.no_grad():
            h = torch.randn(1, rcfg.hidden_size)
            _, scores, indices = router(h)
        if indices.shape[-1] != 10:
            return blocker(f"official router top_k is {indices.shape[-1]}, not 10")
        print("  MoE official top-10: OK")
    except Exception as e:
        traceback.print_exc()
        return blocker(f"official router failed: {type(e).__name__}: {e}")

    if c_bin:
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        import qwenparts_ref
        rc = qwenparts_ref.main(c_bin)
        if rc != 0:
            return blocker("C isolated dump does not match official equations")
        print("  C dump vs equations: OK")

    print("\nPASS official tiny-model oracle")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="1,2,3,4")
    ap.add_argument("--c-bin", default="")
    ap.add_argument("--src", default="",
                    help="pinned checkpoint; streams one official layer at a time")
    ap.add_argument("--dump", default="", help="last-token logits as f32")
    ap.add_argument("--hidden", default="", help="last-token hyper-state per layer")
    ap.add_argument("--max-layers", type=int, default=0)
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x]
    print("Qwen4Exp official oracle")
    print(f"  transformers pin: {TRANSFORMERS_PIN}")

    mods, err = import_official()
    if err is not None:
        return err

    if args.src:
        try:
            rc, _, _ = run_streaming(
                args.src, ids, args.dump, args.hidden, args.max_layers)
            return rc
        except Exception as e:
            traceback.print_exc()
            return blocker(f"streaming oracle failed: {type(e).__name__}: {e}")

    return tiny_oracle(mods, ids, args.c_bin)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
