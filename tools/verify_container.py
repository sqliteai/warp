#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SQLite Cloud, Inc.
"""
verify_container.py — read a WASTE container back and check it against the
source weights. This is the test that the format actually works: it parses
records with the exact byte layout of src/waste_format.h, so a mismatch in
header size, offsets, alignment or codebook indexing shows up here.

  uv run --with torch python tools/verify_container.py \
      --container /path/model.waste --src /Volumes/WasteDisk/kimi-linear
"""

import argparse
import json
import os
import struct
import sys
import zlib

import torch

MAGIC_EXPERT = 0x50584557
MAGIC_CODEBOOK = 0x4B424357
ALIGN = 4096
VEC_DIM = 8
CB_ENTRIES = 256
IDX_BLOCK = 64
HDR = "<IHHBBHHHIIIIIIII"        # must match waste_expert_hdr
HDR_SIZE = 48
KINDS = (("gate", "w1"), ("up", "w3"), ("down", "w2"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mxfp4 import ST                                              # noqa: E402
from convert import (                                             # noqa: E402
    qwen_packed_names, ple_source_loc, ple_shard_map)

# Where the Qwen checkpoint puts its layers. The container's tensor_prefix
# is "" on this family (see convert.source_prefixes), so it cannot be used
# to build a *source* name.
QWEN_SRC_PFX = "model.language_model."


def dequant_q8g(payload, scales, shape, group=128):
    """A whole Q8G tensor. The reference dequant_q8g_row is checked against."""
    rows, N = 1, shape[-1]
    for s in shape[:-1]:
        rows *= s
    ng = (N + group - 1) // group
    q = torch.frombuffer(bytearray(payload), dtype=torch.int8).reshape(rows, ng, group)
    sc = torch.frombuffer(bytearray(scales), dtype=torch.float16).float().reshape(rows, ng, 1)
    return (q.float() * sc).reshape(rows, ng * group)[:, :N].reshape(*shape)


def dequant_q8g_row(q_row, sc_row, width, group=128):
    """One Q8G row without materializing the whole matrix.

    A PLE head is ~20 M rows. Slurping trunk.bin (81 GiB) or dequantizing
    the head as f32 (~12 GiB) swaps a 48 GB machine.
    """
    ng = (width + group - 1) // group
    pad = ng * group
    q = torch.frombuffer(bytearray(q_row), dtype=torch.int8)[:pad].view(ng, group)
    sc = torch.frombuffer(bytearray(sc_row), dtype=torch.float16).float().view(ng, 1)
    return (q.float() * sc).reshape(pad)[:width]


def packed_expert_src(eid, kind, cache):
    """One expert's gate / up / down out of the layer's packed pair."""
    gate_up, down, inter = cache
    if kind == "gate":
        return gate_up[eid, :inter]
    if kind == "up":
        return gate_up[eid, inter:]
    return down[eid]


def load_codebooks(path):
    """codebooks.bin -> list of [CB_ENTRIES, VEC_DIM] tensors, in file order."""
    books, data = [], open(path, "rb").read()
    rec = 16 + CB_ENTRIES * VEC_DIM * 2
    for off in range(0, len(data), rec):
        magic, _cid, _fmt, vdim, n, _r = struct.unpack("<IHBBII", data[off:off + 16])
        assert magic == MAGIC_CODEBOOK, f"bad codebook magic at {off}"
        assert vdim == VEC_DIM and n == CB_ENTRIES
        t = torch.frombuffer(bytearray(data[off + 16:off + rec]),
                             dtype=torch.float16).view(n, vdim).float()
        books.append(t)
    return books


def read_expert(bank_bytes, rec_off, books, cb_base, stages, shapes, block=0):
    h = struct.unpack(HDR, bank_bytes[rec_off:rec_off + HDR_SIZE])
    (magic, layer, eid, fmt, flags, cb_id, lowrank_id, _r0,
     blocks, g_off, u_off, d_off, corr_off, crc, _r1, _r2) = h
    assert magic == MAGIC_EXPERT, f"bad expert magic at {rec_off:#x}"
    assert lowrank_id == 0, "v0 requires lowrank_id == 0"
    assert cb_id == cb_base, f"codebook base mismatch {cb_id} != {cb_base}"

    end = rec_off + blocks * ALIGN
    body = bank_bytes[rec_off + HDR_SIZE:end]
    # crc covers the body up to the padding
    payload_len = corr_off - HDR_SIZE + sum(s[0] for s in shapes) * 2
    assert zlib.crc32(bytes(body[:payload_len])) & 0xFFFFFFFF == crc, "CRC mismatch"

    out, scale_cursor = {}, corr_off - HDR_SIZE
    offs = {"gate": g_off, "up": u_off, "down": d_off}
    for i, (kind, _tag) in enumerate(KINDS):
        M, N = shapes[i]
        nvec = M * N // VEC_DIM
        beg = offs[kind] - HDR_SIZE
        raw = torch.frombuffer(bytearray(body[beg:beg + nvec * stages]),
                               dtype=torch.uint8)
        if block:                       # [M/B][nvr][B][stage] -> [nvec][stage]
            nvr = N // VEC_DIM
            nb = (M + block - 1) // block
            idx = (raw.view(nb, nvr, block, stages).permute(0, 2, 1, 3)
                      .reshape(nb * block, nvr, stages)[:M]
                      .reshape(nvec, stages).long())
        else:
            idx = raw.view(nvec, stages).long()
        recon = torch.zeros(nvec, VEC_DIM)
        for s in range(stages):
            recon += books[cb_base + i * stages + s][idx[:, s]]
        sc = torch.frombuffer(bytearray(body[scale_cursor:scale_cursor + M * 2]),
                              dtype=torch.float16).float().view(M, 1)
        scale_cursor += M * 2
        out[kind] = recon.view(M, N) * sc
    return out, blocks, eid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", required=True)
    ap.add_argument("--src", default="/Volumes/WasteDisk/kimi-linear")
    ap.add_argument("--experts", type=int, default=4, help="how many to check")
    ap.add_argument("--layers", default="",
                    help="comma list of layer ids; default = all")
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.container, "manifest.json")))
    stages = man["expert_quant"]["stages"]
    books = load_codebooks(os.path.join(args.container, "codebooks.bin"))
    print(f"container: {man['expert_quant']['fmt']}, {len(books)} codebooks, "
          f"layers {list(man['layers'])}")

    sr = ST(args.src)
    prefix = man.get("tensor_prefix", "")
    want = None
    if args.layers.strip():
        want = {int(x) for x in args.layers.split(",") if x.strip()}
    ok = True
    for lstr, meta in man["layers"].items():
        L = int(lstr)
        if want is not None and L not in want:
            continue
        bank = open(os.path.join(args.container, meta["file"]), "rb").read()
        assert len(bank) == meta["bytes"]
        shapes = []
        moe_segment, src_kinds = "block_sparse_moe", KINDS

        # Qwen packs a whole layer's experts into two tensors, so there is
        # no per-expert tensor to read a shape from and no per-expert slice
        # to compare against — both come out of the pair. Probed on the
        # source, like the DeepSeek naming below.
        gname, dname = qwen_packed_names(QWEN_SRC_PFX, L)
        packed_cache = None
        if sr.have(gname):
            gate_up, down = sr.tensor(gname), sr.tensor(dname)
            inter = int(gate_up.shape[1]) // 2
            hid = int(gate_up.shape[2])
            packed_cache = (gate_up, down, inter)
            shapes = [(inter, hid), (inter, hid), (hid, inter)]
        elif sr.have(f"{prefix}model.layers.{L}.mlp.experts.0.gate_proj.weight"):
            src_kinds = (
                ("gate", "gate_proj"),
                ("up", "up_proj"),
                ("down", "down_proj"),
            )
            moe_segment = "mlp"

        if packed_cache is None:
            for _kind, tag in src_kinds:
                t = sr.tensor(
                    f"{prefix}model.layers.{L}.{moe_segment}.experts.0.{tag}.weight"
                )
                shapes.append(tuple(t.shape))

        off, checked = 0, 0
        while off < len(bank) and checked < args.experts:
            rec, blocks, eid = read_expert(bank, off, books,
                                           meta["codebook_base"], stages, shapes,
                                           man["expert_quant"].get("index_block", 0))
            assert off % ALIGN == 0, f"record {eid} not 4 KiB aligned"
            for i, (kind, tag) in enumerate(src_kinds):
                if packed_cache is not None:
                    W = packed_expert_src(eid, kind, packed_cache)
                else:
                    W = sr.tensor(
                        f"{prefix}model.layers.{L}.{moe_segment}.experts.{eid}.{tag}.weight"
                    )
                err = (W - rec[kind]).norm() / W.norm()
                flag = "ok " if err < 0.30 else "BAD"
                if err >= 0.30:
                    ok = False
                print(f"  L{L} e{eid:<3} {kind:<5} {tuple(W.shape)} "
                      f"rel err {err:>6.2%}  {flag}")
            off += blocks * ALIGN
            checked += 1
        print(f"  layer {L}: {len(bank)//ALIGN} blocks, "
              f"{meta['experts']} experts, {len(bank)/2**20:.1f} MB, "
              f"{len(bank)/meta['experts']/2**20:.2f} MB/expert")
        del bank, packed_cache

    cfg = man.get("config") or {}
    offsets, sizes = cfg.get("ple_head_offsets"), cfg.get("ple_head_vocab_sizes")
    shards = ple_shard_map(sr.wm)
    if offsets and sizes and shards:
        print("PLE rows")
        trunk_path = os.path.join(args.container, "trunk.bin")
        heads = [t for t in man["trunk"]
                 if "ngram_head." in t.get("name", "")]
        sample = sr.raw(shards[min(shards)])
        shard_rows = int(sample.shape[0])
        del sample
        group = 128
        with open(trunk_path, "rb") as tf:
            for t in heads:
                h = int(t["name"].split("ngram_head.")[1].split(".")[0])
                n_rows = int(sizes[h])
                start = int(offsets[h])
                width = int(t["shape"][-1])
                ng = (width + group - 1) // group
                pad = ng * group
                for local in (0, n_rows // 2, n_rows - 1):
                    gi = start + local
                    si, lr = ple_source_loc(gi, shard_rows)
                    src = sr.raw(shards[si])[lr].float()
                    tf.seek(t["off"] + local * pad)
                    q = tf.read(pad)
                    tf.seek(t["scale_off"] + local * ng * 2)
                    sc = tf.read(ng * 2)
                    recon = dequant_q8g_row(q, sc, width, group)
                    err = (src - recon).norm() / src.norm().clamp(min=1e-8)
                    flag = "ok " if err < 0.05 else "BAD"
                    if err >= 0.05:
                        ok = False
                    print(f"  head {h} row {local} (src shard {si}[{lr}]) "
                          f"rel err {err:>6.2%}  {flag}")
                    del src

    print("\nPASS — container round-trips" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
