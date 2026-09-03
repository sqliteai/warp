# GLM-5.3-Flash on WASTE

`zai-org/GLM-5.3-Flash` — 313 B parameters, 328 GB of fp8 as published,
`Glm5NextForConditionalGeneration`. It is close enough to Kimi K3 that most
of this engine already ran it: the same KDA recurrence, the same MLA with
`kv_b_proj` absorbed, the same sigmoid / top-k router over per-layer expert
banks, the same expert-per-record container. Three things are new, and this
document is mostly about those and about the three places the release states
something the engine already states, differently.

## The shape

| | |
|---|---:|
| layers | 45 (34 KDA, 11 MLA — one in four) |
| hidden | 4096 |
| routed experts | 288, top-8, 2048 wide |
| shared experts | 1 |
| dense layers | the first 3 |
| KDA | 64 heads x 128, conv 4, `gate_lower_bound` -5.0 |
| MLA | 64 heads, `q_lora` 1536, `kv_lora` 512, qk_nope 256, v 256 |
| rope | none at all: `qk_rope_head_dim` 0 and `mla_use_nope` |
| context | 1,048,576 |
| vocab | 154,880 |

The layer mix, the KDA parameterization and the router are Kimi K3's, field
for field. What follows is what is not.

### mHC — Manifold-Constrained Hyper-Connections

The residual stream is not one vector but `hc_mult` = 4 parallel ones.
Before each sublayer a learned mapping reads all four at once — a
`24 x 16384` projection of the flattened, RMS-normalized streams — and
produces three things:

    pre  [4]     collapse weights: the one vector the sublayer runs on
    post [4]     where the sublayer's output lands, per stream
    comb [4][4]  how the streams mix into each other

`comb` is then projected onto the doubly-stochastic manifold by
Sinkhorn-Knopp, twenty alternating row/column normalizations, which is the
constrained half of the name and what keeps the four streams' norms from
diverging over 45 layers. After the sublayer,

    x[i] <- post[i] * y + sum_k comb[k][i] * x[k]

and at the end of the stack the four are collapsed by an unweighted mean.

Two sites per layer, ninety in all. The arithmetic is negligible — a 4x4
Sinkhorn is nothing — but the projection that feeds it is 393216 weights
per site, 35 B in total, and it lives in the resident trunk.

This is the same *kind* of mechanism as K3's Attention Residuals: both let a
layer see more than the one before it. They are mutually exclusive in
practice, and `src/model.c` treats them as three alternatives — plain
residual, AttnRes, mHC — at the same two points in the layer loop.

### The clamped SwiGLU

`swiglu_limit` is 10.0: the gate half is clamped above, the up half on both
sides, **before** the SiLU-and-multiply. It fires on real activations, so
dropping it produces a model that looks right and drifts.

The family now has three activations — plain, K3's SiTU, and this — at
seven call sites in `model.c`, which is exactly the shape in which a fourth
gets forgotten at one of them. They are one function,
`waste_act_pair_range`.

### DeepSeek Sparse Attention, k-pool flavour

An MLA layer does not attend over the whole context. A small indexer, with
projections of its own, scores *pools* of `index_kpool` = 4 adjacent cached
tokens and keeps the best `index_topk / index_kpool` = 512 of them; the
query attends over those pools' tokens plus the tail of the context that has
not filled a pool yet.

Two things make it cheap to keep resident:

- A pool's compressed key is one 128-wide vector per four tokens — 32x less
  than the raw keys the scores would need — and it is computed once, when
  the pool's last token arrives. The per-step state is a rolling buffer of
  the four (key, gate) pairs of the pool being filled, plus an append-only
  array of finished pool keys. At 128 B per token per full-attention layer
  it is a fourteenth of what the latents already cost.

- **Below 2048 tokens of context this is exactly dense attention.** With no
  more complete pools than the query is allowed to keep, the selection is
  every visible token; `dsa_select` says so by returning -1 and the head
  loop takes its ordinary path. That is not an approximation for short
  prompts — it is what the arithmetic reduces to, and it is why the
  selection cost only appears where the saving does.

The one place this is unspecified: a pool whose every head scored negative
lands at exactly 0 after the relu, and so do its neighbours. Which of those
the top-k keeps is ordered by index upstream and by heap order here. They
contribute the same nothing either way.

## The three silent differences

All three are converter-side, all three are invisible to a forward-pass
diff against an oracle that reads the same manifest — it would be wrong the
same way — and all three are covered by `tests/test_convert_glm.py`.

**The text model is nested the other way round from K3.**

    K3    language_model.model.layers.N.…    language_model.lm_head.weight
    GLM   model.language_model.layers.N.…    lm_head.weight

Same two components, opposite order, and `lm_head` inside the wrapper on
one and outside it on the other. The engine looks up
`{tensor_prefix}model.layers.N.…`, so K3's spelling is a prefix away from
it and GLM's is not: nothing is found, every tensor reads as absent, and
the load refuses a container that in fact holds every weight. GLM's
container is therefore prefix-less and the wrapper is unnested at
conversion (`glm_rename`); there is nothing left for a prefix to
disambiguate, since the vision tower is not carried.

This one was caught before the first conversion rather than after it, by
checking every name and shape the engine will demand against the
checkpoint's own index — 1246 tensors, 0 missing, 0 mismatched — which is
an hour of conversion cheaper than finding out at load.

**`linear_attn_config.kda_layers` is 0-based on GLM and 1-based in a WASTE
manifest.** Copied through, it puts KDA on the wrong layers and MLA on the
rest. Every tensor is still found, every shape still checks out, and the
model answers noise. `convert.py` rebuilds it from `layer_types`, which is
unambiguous, rather than shifting it by one and hoping.

**`eos_token_id` is a list of three** — end-of-text plus two turn markers.
The engine's config holds one id; a list read as an integer stops on
nothing. The first is taken, because the turn markers belong to the chat
format rather than to the model config.

## The tokenizer

GLM ships `tokenizer.json` (the `tokenizers` library's own JSON) where both
Kimi releases ship a tiktoken rank file. Same byte-level BPE, different
container, so `tools/hf_tokenizer.py` re-encodes it: undo GPT-2's
bytes-to-unicode escape, emit `base64(bytes) rank`, and `src/tokenizer.c`
reads it exactly as it reads Kimi's. It refuses rather than approximates
when the merge list is not ordered by the id of what it produces — that
ordering is what makes merge-by-rank and merge-by-list-position the same
encoder.

One difference is real and was found by checking rather than by reading.
**Kimi's pre-tokenization pattern gives Han its own `[\p{Han}]+` branch and
GLM's does not**, so on GLM a Han run and the Latin run it touches are one
pre-token rather than two. Sixteen tokens in this vocabulary span that
boundary, and they are the frequent kind:

| | Kimi's pattern | GLM's, and the release |
|---|---|---|
| `A股` | `32 98963` | `111321` |
| `维生素C` | `103261 34` | `121569` |
| `C罗` | `34 99209` | `126152` |
| `QQ音乐` | `47724 99908` | `126724` |

So the container states which, in `tokenizer_han_split`, and the engine
passes it to `waste_tok_set_han_split`. The default is the Kimi pattern: a
default that has to be set to keep working is a default that gets missed.
`tools/tokdiff.py` picks its reference from what the source ships and
compares 21 strings, the last two of which cross that boundary.

## Converting

```bash
uv run --with torch python tools/convert.py \
    --src /path/to/GLM-5.3-Flash --out glm.waste
```

Nothing GLM-specific on the command line: the fp8 reader, the DeepSeek MoE
tensor naming and the nested `text_config` were all already there for K2 and
K3. The conversion drops two families of tensor the container has no reader
for — the MTP layer at index 45 (`num_nextn_predict_layers`, a
speculative-decoding head) and the vision tower — because the trunk is
resident for the life of the process and a tensor nothing reads is RAM taken
from the expert cache.

The mHC projections are kept at 8 bits rather than the trunk's usual 4. Its
24 outputs decide how the residual streams mix for a whole layer, so an
error there is not one weight's worth; the two widths are 34 MB and 8 MB.

### What it comes to

Measured, on the conversion this document now describes (LEARNED.md §68):

| | | K3, for scale |
|---|---:|---:|
| download | 306 GiB, ~2 h | 1.42 TB |
| conversion, `--jobs 3` | ~45 min | 4.7 h |
| trunk, resident | **5022 MB** | 27.3 GiB |
| experts at VQ3R | 42 x 2598 MB = **109 GB** | 962.8 GB |
| container | **112 GB** | 982 GB |
| one token's working set | **3.17 GB** | 17.19 GB |
| floor | **5.14 GB** | 29.19 GB |

`waste info` reports **313.33 B parameters, 16.74 B active per token**.

Which is a very different proposition from K3. On this 64 GB machine the
automatic budget resolves **46.37 GB with a 41.36 GB expert cache** —
thirteen working sets, 36% of the whole expert set resident — where K3 gets
one and a half.

### What it does

    $ waste run ~/models/glm53.waste "The capital of Italy is" -n 20
    waste: no --budget, using 46.37 GB of 64.00 GB (expert cache 41.36 GB)
    The capital of Italy is Rome. Rome is the largest city in Italy and is
    known for its rich history, iconic landmarks such
    [20 tokens, 5.07 s, 3.94 tok/s | experts 5816 hit / 904 miss = 87%]

| `waste bench` | tok/s | hit | read |
|---|---:|---:|---:|
| 64 tokens | 2.40 | 87.7% | 72.7 GB |
| 200 tokens | **3.09** | 89.3% | 160.8 GB |

Five times K3's 0.45-0.62 tok/s. The expert cache buys traffic rather than
throughput on this NVMe — the same 200 tokens at the pre-§66 budget run at
3.16 tok/s and read **481 GB** — which is a three-fold difference on a disk
slow enough for it to show.

## The chat format

GLM's format needs three things the four-string `chat.json` did not have,
and each is there because the format cannot be written without it:

```json
{"prelude": "[gMASK]<sop>",
 "system": ["<|system|>", ""],
 "user": ["<|user|>", ""],
 "assistant": ["<|assistant|>", ""],
 "open": "<|assistant|>",
 "think": ["<think>", "</think>"],
 "stop": "<|user|>",
 "image": "<|begin_of_image|><|image|><|end_of_image|>"}
```

`image` is the block one picture expands into, holding exactly one
placeholder the engine repeats into as many positions as the tower
produced. `prelude` opens every conversation and belongs to no role. `stop` exists
because a GLM turn ends when the *next role marker* begins — there is no
suffix to close it with, so the history must not carry one, and without it
the CLI ran on and kept answering questions nobody asked. `think` is the
reasoning channel: the generation prompt opens it, the model closes it
itself, and naming the pair is what lets a reader separate the reasoning
from the answer instead of returning the model's scratch work as the reply.

`examples/chat-glm53.json` is that file and the converter installs it.
Over HTTP the two come back separately:

```json
"content": "The capital of Italy is Rome.",
"reasoning_content": "The user is asking a simple factual question..."
```

streaming included. `reasoning_effort` maps to GLM's own spelling of it —
a system turn, `<|system|>Reasoning Effort: High` — through the optional
`effort` field. `thinking: false` is refused: GLM's template has no path
that leaves the channel closed, and answering with it closed puts a stray
`</think>` in the reply.

## The vision tower

GLM's is not K3's, so `src/vision.c` holds two — dispatched on `tower` in
`vision.json`, and a file without the key is K3's, which is every container
written before this one.

| | K3 | GLM-5.3-Flash |
|---|---|---|
| blocks | 27 x 1024, 12 heads | 24 x 1024, 16 heads |
| position | learned 64x64 grid, bilinearly resized | 2D rope only |
| q/k | as projected | RMSNormed per head, then rotated |
| biases | none | on every projection |
| MLP | GELU (tanh) | clamped SwiGLU |
| patch | Conv2d over 3x14x14 | Conv3d over 3x2x14x14, the temporal slot a copy |
| merge | 2x2 reshape, two-layer projector | Conv2d downsample, gated merger with a LayerNorm and an exact GELU |

**Against `tools/glm_vision_ref.py`: rel L2 3.3e-5** on the real 563.6 M
tower, 7.7e-7 on the synthetic one `tests/run.sh` builds.

```
$ waste run glm53.waste "What does this image look like?" --image x.png
[x.png: 40 image tokens]
This image displays a vibrant, abstract pattern of diagonal stripes in
various colors like green, blue, purple, and pink, overlaid with fine
vertical lines.
```

The tower is 282 MB at 4 bits and is loaded only when a caller asks for
images; `waste plan` prices it at 805 MB more with `--image`.

### Two things the preprocessing had to get exactly right

**The patch order is block-major over merge blocks** — block row, block
column, then the 2x2 inside it — which is what lets the downsample be a
reshape and what the rotary indices assume. Raster order, the obvious
reading and what K3 uses, rotates every patch by someone else's position:
the output stays finite and the model describes a different picture.
Checked against the reference's own reshape on an image already aligned to
`patch * merge`, so only the ordering is under test: **max abs difference
exactly 0**.

**The resize is antialiased bicubic**, because bilinear is not close
enough. Measured on the tower's input:

| the engine's pixels vs | rel |
|---|---:|
| torch bilinear | 0.0000 |
| torch bicubic + antialias (the release's) | **0.0773** |

The convention was already exact — half-pixel centres, matching torch's
bilinear to the bit — and the *kernel* was 7.7% off, which is a different
image rather than a noisier one. So the kernel is implemented: separable,
cubic with a = -0.5, support widened by the scale factor on a downsample.
Against torch's antialiased bicubic: **max abs 4e-5**.

The grid rule is transcribed from `smart_resize`, including the two things
in it that look like accidents: the frame count rounds to a multiple of the
temporal factor before it multiplies the budget, and the binary search
advances `low` to `content_height + 1` rather than to the aligned height.
- **Chunked prefill**, and measured not to matter. `waste_model_prefill`
  routes a GLM container through the per-token path, because the chunked
  one implements neither mHC nor the indexer's per-token pool bookkeeping.
  On the model that *does* have that path, warm cache, it is worth nothing
  at either length tried — 4063 ms against 4075 for 64 tokens, 27067
  against 27231 for 512 — because 82.8% of a chunked prefill is the VQ
  apply and a chunk expands nearly the whole bank. Prefill runs at decode
  speed here (266 tokens in 69 s) and chunking is not what would change
  that. LEARNED §69.
- **Cross-layer top-k sharing.** `indexer_types` is all `"full"` on this
  release. A container converted from a release that shares a selection
  across layers is refused rather than produced.
- **MTP.** The extra prediction layer is dropped, as above.
- **Tools.** GLM's template carries a full tool-call protocol, and the
  declarative `chat.json` cannot express one — so the server renders it
  from the tokenizer instead. GLM's specials carry the whole XML grammar
  (`<tool_call>`, `<arg_key>`, `<arg_value>` and the response/observation
  markers) as single tokens, `serve/glmtools.py` renders and reads it back
  the way the release's own `chat_template.jinja` spells it, and
  `tests/serve/test_glm_upstream.py` diffs that rendering against the
  template with `GLM_DIR` naming the release. The raw `.jinja` stays in the
  container for a host that does interpret Jinja.

## What is checked

`tests/run.sh` builds its own GLM-shaped container — a few megabytes, seed
0, byte-reproducible — because none of the checks that run on a Kimi reach
any of the three new things, and all three fail quietly. mHC produces
weight-shaped logits from any mixing matrix; a clamp that never fires looks
like a clamp that works; and an indexer that selects every pool is
indistinguishable from one that selects the right ones until the context
outgrows `index_topk`.

- **against the oracle.** `tools/kimi_ref.py` grew mHC, the clamped SwiGLU
  and the indexer, so one reference covers the family. Sixteen tokens with
  `index_topk` 8 over pools of 4: four complete pools of which the indexer
  may keep two, so the sparse branch is really reached. At twelve tokens or
  fewer every pool is kept and the check would pass with the selection
  deleted. Measured max abs 5.7e-6 on the last token's logits, against a
  1e-3 threshold — the same order as the Kimi baseline.

  **And on the real container**: 4 tokens of the converted 313 B model
  against the same reference, **rel L2 2.41e-5**, max abs 2.39e-4, argmax
  and top-10 identical in the same order.

- **the sparse selection, on real weights.** The run above exercises the
  dense branch — `index_topk` is 2048 and the prompt is four tokens. To
  reach the other one without an hours-long oracle run, the container is
  cloned with symlinks and a manifest whose `index_topk` is 16: the same
  112 GB of weights, four pools kept instead of 512, and the branch reached
  at 24 tokens. **The selections are identical, 55 of 55** — the pools
  themselves, compared through a `WASTE_DUMP_DSA` trace against the same
  line format from the reference, not inferred from a logit difference.

  The logits in that run differ by 1.95e-3 rather than 2.41e-5, and the
  control says why: the same 24 tokens through the dense path give
  1.74e-3. It is the KDA recurrence over 24 steps, not the selection.

  At the real setting, 2100 tokens: the branch fires on all 11 MLA layers
  for positions 2051-2099, keeping 512 pools of 513 — 2048 pooled positions
  plus the tail, of 2052 cached. Against the same prompt with the selection
  off, rel L2 0.0909 with argmax and top-10 unchanged.
- **that the indexer selects.** The same weights with `index_topk` raised
  above the prompt length, which is one number in the config and nothing
  else. The two must differ; measured 0.56 max abs.
- **session state.** The four streams and the indexer's pool history are
  both session state, and both are written to and read from a saved state.
- **the chunked fallback.** Bit-identical to decoding the same prompt.
- **the converter's config handling**, `tests/test_convert_glm.py`, for the
  two silent differences above.
- **the tokenizer**, `tools/tokdiff.py` against the `tokenizers` library on
  the real `tokenizer.json`: 21 of 21 identical, including the Han/Latin
  boundary.

The synthetic container remains the only place any of this runs in CI:
1/32 scale, shapes the engine branches on, weights that are noise. What the
real conversion adds is that the same code, on 313 B real parameters, makes
the same selections as an independent PyTorch implementation, agrees with
it on the logits, and answers questions in English.
