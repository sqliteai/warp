# Qwen3.8-Flash-Next

Text inference for `qwen4_exp` / `qwen4_exp_text`. Four architectural
pieces that no other model here has, and a container format that does not
change to hold them.

Pinned for every measurement below:

    Qwen/Qwen3.8-Flash-Next
    revision de4b8e4d43b917e7706784d8bb445c9af86a3540

131 shards, 360,013,002,208 bytes on the hub, index SHA-256
`99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
Fetch the revision, not `main` — this repository's `main` has moved since,
and a container built from a different one is not the container these
numbers describe.

## What is different about it

| | Kimi / GLM | Qwen3.8-Flash-Next |
|---|---|---|
| recurrent layers | Kimi Delta Attention | **Gated DeltaNet** — per-head scalar decay, 16 QK heads repeated onto 48 V heads |
| attention layers | MLA over a compressed latent | **Qwen Sparse Attention** — an indexer scores 4-key mean-pooled blocks, keeps the best 512 plus the token tail, and attention runs over the *original* K/V |
| residual | one stream (four on GLM's mHC) | **HyperConnection** — four streams mixed through a rank-320 bottleneck and recombined through a per-branch gate |
| extra embedding | none | **PLE** — an n-gram lookup at one layer, 16 hashed head tables of ~20 M rows each |
| router | sigmoid plus a learned bias | **softmax over all 512, then top-k** |
| experts on disk | one tensor per expert per matrix | **two packed tensors per layer** |

Layer mix is 36 `linear_attention` and 12 `full_attention`, stated by
`layer_types` and read from it. A container that does not say is refused:
"every layer is GDN" is a plausible-looking default that would be wrong in
every token and visible in none.

Shape: hidden 2560, 48 layers, vocab 248320, 512 experts with top-10
routed plus one gated shared expert of width 640, GDN 16×128 QK and 48×128
V, QSA 24 query and 2 KV heads at dim 256 with 64 rotary dims, indexer MQA
at dim 128 with a 2048 budget, `hc_count` 4 / `hc_lowrank` 320, n-gram
size 3 over 16 heads, `ple_layer_ids: [2]`.

`ple_layer_ids` is 1-based and the PLE tensors live on
`model.language_model.layers.1`. Two spellings of the same layer; the
container carries the 0-based one.

## Converting

```bash
tools/fetch_weights.sh --repo Qwen/Qwen3.8-Flash-Next \
    --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
    --dest /path/to/raw

uv run --with torch --no-project python tools/convert.py \
    --src /path/to/raw --out /path/to/qwen38-flash-next.waste
```

No Qwen-specific flags. Three things happen that do not happen for any
other family:

* **Packed experts.** `experts.gate_up_proj` [E, 2I, H] and
  `experts.down_proj` [E, H, I] hold a whole layer's routed experts. They
  are split into ordinary WEXP records — one expert per 4 KiB-aligned
  record, format v0, nothing about the read path changes. The layout is
  validated against its invariant rather than against Flash-Next's
  dimensions, so a larger family member with the same packing converts on
  the same code.

* **PLE.** 128 source shards become 16 Q8G head tensors on the trunk. A
  head is ~12 GiB as f32, so it is quantized 64 Ki rows at a time with one
  source shard resident; Q8G groups along the last dimension, which is
  what makes the concatenation of the batches equal to quantizing the head
  whole. The i64 offset and vocabulary-size tables go into the manifest as
  integers — they are primes near 2×10⁷ and do not survive a float.

* **`--jobs` defaults to 1.** A worker holds a whole layer's packed pair,
  3.3 GiB of BF16 before the f32 it dequantizes into. Three at once is
  what turns a conversion into a swap storm. An explicit `--jobs` wins.

`--reclaim` works, with the n-gram shards as a consumer of their own:
`build_ple` runs after the trunk pass, so a shard holding both an expert
and an n-gram slice is not released until the 16 heads are on the trunk.
The vision tower and the MTP layer have no consumer at all here and are
released first.

`tools/verify_container.py` reads packed sources and checks PLE rows
against their source shard, three rows per head, without dequantizing a
head.

## Running

```bash
./waste run /path/to/qwen38-flash-next.waste "The capital of France is" \
    --budget 8G
```

Measured on an Apple silicon laptop, 12 logical CPUs (8 performance, 4
efficiency), 48 GiB of RAM, container on the internal SSD, at the commit
that ships this file:

| | |
|---|---:|
| container | 123.9 GiB |
| ├ `trunk.bin` | 80.51 GiB |
| ├ expert banks | 42.5 GiB |
| resident trunk | 2.60 GB |
| memory floor | 3.11 GB |
| one token's expert working set | 888.7 MB |
| parameters | 176.94 B total, 57.87 B active per token |

The gap between an 80.51 GiB trunk file and a 2.60 GB resident trunk is
the point of the PLE design: the 16 n-gram heads are 78 GiB of the file
and are read one row per head per token, never held. `waste_plan_memory`
excludes them from the resident set for the same reason it excludes the
embedding table.

Throughput, 48 tokens greedy from the same prompt:

| expert cache | tok/s | hit rate |
|---:|---:|---:|
| 4 GiB | 3.20 | 8% |
| 8 GiB | 4.97 | 64% |
| 16 GiB | 4.92 | 88% |

**8 GiB is the knee and there is nothing above it.** Below one token's
working set the hit rate collapses rather than degrades — the 4 GiB row is
that, not a gentle slope — and above 8 GiB a better hit rate buys no time
at all: the reads it saves were already overlapping the arithmetic. That
is the whole reason the default budget is not "as much as the machine
has".

Threads, at 8 GiB: 5.04 tok/s at the default (one per logical CPU), 5.38
at `--threads 8`, 5.14 at 6, 5.09 at 12. Eight — the performance-core
count on this machine — is worth about 7%, which is real but is a property
of this machine and not of the architecture, so it is left to `--threads`
rather than compiled in as a default. LEARNED §47 has the same finding
inverting between two other models.

Routed experts go through the existing expert-parallel path and its
existing per-layer decision: one task per expert when the records are
already resident, one task per row range when holding a batch would
barrier the read-ahead. `WASTE_XPAR=0/1` still forces it.

## Correctness

* **Components.** `tests/test_qwenparts.c` dumps every intermediate of the
  five kernels and `tools/qwenparts_ref.py` recomputes them in PyTorch
  from the published equations — at the official geometry as well as at
  toy sizes. Largest disagreement across the dump: 2.4e-7 absolute.

* **Whole forward pass.** `tools/qwen_container_ref.py` implements the
  same forward pass in PyTorch reading the same container, so the
  comparison is against an independent decode of identical weights. On the
  synthetic fixture the routed expert ids and weights match **exactly** at
  every layer and the logits argmax matches. The worst hidden-state
  difference is 2.4e-7 absolute and the final-logit difference is 1.9e-6.

* **Tokenizer.** Qwen's pre-tokenization pattern splits every digit into
  its own piece where Kimi's and GLM's take up to three. The engine is
  told which through `tokenizer_digit_run`; `tools/hf_tokenizer.py`
  refuses a pattern it does not recognise rather than approximating one.
  Against the release's own tokenizer the C encoder matches on every
  string in `tests/test_qwen_tok.py`, numbers included. The rank file this
  path writes is byte-identical to one built from `vocab.json` through the
  GPT-2 byte map.

* **End to end.** On the pinned checkpoint at an 8 GiB budget, "The
  capital of France is" completes to "**Paris**".

## Not supported

Text only. Not converted, not executed, and refused rather than half-done:

* the **vision tower** (`model.visual.*`, 333 tensors) and everything that
  configures it — no `vision.json` is written, so a container cannot be
  handed an image;
* **video and audio**;
* the **MTP layer** (`mtp.*`, 31 tensors) and speculative decoding;
* **native Qwen serving** — `serve/` has no Qwen chat format, so the
  OpenAI-compatible server falls back to `chatfmt.py`, which refuses
  tools, thinking and images by name rather than dropping them.

The chat template is carried into the container as metadata, as every
other release's is. Nothing reads it.
