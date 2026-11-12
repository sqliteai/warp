# Changelog

Every release since 0.6.1. Numbers here were measured on the commit they
ship with, the same rule the rest of the documentation follows — where a
change was measured and *not* adopted, that is recorded too, because the
measurement is the useful part.

`docs/LEARNED.md` carries the full reasoning; this file carries what
changed. Each entry names the section to read for the numbers behind it.

## 0.8.2 — 2026-11-12

**The server can hold a swappable set of containers** — contributed by
@hendrikras in #74 — and **the trunk kernel is a property of a model
rather than of the process**, by @mfethe1 in #75, which closes #68.

Both were re-measured on this commit rather than taken on trust. With a
Kimi-Linear container: 110 passed, 0 failed, 6 skipped, the serve suite at
417 checks (from 319), and `make asan` clean with no sanitizer report,
100 passed against 0.8.1's 96. Line for line, against 0.8.1 both runs
differ by the same four added checks — #75's two and the two version
greps below — and the serve suite's own count. Nothing else moved.

The registry was additionally driven over a real socket against real
weights — two containers behind one port, and a second pair of two
different architectures — for the swap ordering, the refusals, and the
per-container facts a swap re-derives.

No ABI move: `src/waste.h` changes only its version macros.

### Added

- **A model registry behind a stable route name**, so the container a
  route name resolves to can change without reconfiguring clients.
  `--models PATH[=ID]` (repeatable) registers what may be switched to;
  `POST /v1/models/load` switches; `GET /v1/models` lists the registry
  with a `loaded` flag per entry. `loaded` means **residency**, the same
  set the load response reports: with `--keep-previous` a model the server
  has swapped away from but still holds an open `waste_ctx` for answers
  `loaded: true`, and switching back to it is a slot move rather than a
  reopen of a multi-gigabyte container.

  A generation naming a model the registry does not know is a **404**, and
  one it knows but has not loaded is a **409** pointing at the load
  endpoint — not a surprise multi-gigabyte swap in the middle of a
  conversation. That strictness follows the operator's opt-in (`--models`,
  or `--auto-register`) and not the number of ids the registry happens to
  hold, which with `--auto-register` is something any client can change.
  **A server started without either flag is unchanged**: absent, the
  container's own id, or any other string all serve, including the fixed
  name most OpenAI clients are configured with and the `"model": "waste"`
  example in `serve --help`.

  A swap opens the new container *before* closing the old one, which is
  the opposite of what a tight-RAM machine would prefer and is the point:
  a failed open leaves the server serving the model it was serving. It
  happens under the current engine's lock, so a generation in flight
  finishes first, and a request that queued behind it finds the slot moved
  and answers 409 rather than generating on a closed container. Everything
  a container decides — reply format, markers, stop tokens, the thinking
  default, the shape `GET /v1/models` reports — moves in a single
  assignment, so no request can see one model's `waste_ctx` with another
  model's format.

  `--auto-register` lets `POST /v1/models/load` register a container by
  pathname it has not seen before. Off by default, for the reason
  `--allow-local-images` is: a client that can reach the port could
  otherwise make the server open an arbitrary file on it. `--usage` cannot
  be combined with either flag — a learned hotlist is specific to one
  container. Request log lines name the model that served or was refused,
  and `--no-log-requests` silences them. See docs/SERVE.md, "Swapping
  models".

- **A swap has to prove it fits, and a load that would not is a 507.** A
  swap opens the new container before closing the old one — the order that
  makes a failed open leave the server serving what it was serving — so
  two contexts are resident at once. Documenting that and leaving the
  operator to size `--budget` for it would have been worse than "the sum
  of the two": with `--budget 0` each context sizes itself to as much as
  3/4 of `waste_usable_ram()`, so a swap would run at ~1.5x what the
  process may use — a paging run, not a slow one.

  So `--models` requires an explicit `--budget`, and the server refuses
  to start unless `2 x budget` fits, naming the largest budget that does;
  the same lines are printed under the startup banner and by `--plan`,
  which is where a budget gets chosen. At runtime the resident set is
  counted at every load — each engine's budget, or the floor and expert
  cache `waste_memory_used` reports for one that chose its own — and a
  load that would not fit is refused with **507**
  (`insufficient_memory`) *before* the container is opened: nothing is
  closed, nothing is half-loaded, the previous model keeps serving, and
  the message says what it needed next to what was already held. 507
  rather than 503 because asking again cannot help.

  That is also what bounds `--keep-previous`, whose resident set grows
  with every model ever switched to and which no startup check can price
  in advance: the cap is derived from the budgets rather than from a
  separate `--max-resident` count that could disagree with them, and a
  slot move to a model that is already resident is never refused because
  it allocates nothing. Evicting a resident model to make room was the
  alternative and is not done — changing what is resident behind a
  client's back is the failure `--keep-previous` exists to prevent.

  A load that arrives with no `--budget` at all — the single-container
  default, and the path an `--auto-register`ed container takes — is priced
  from the container's own plan at its **floor**, the least it can run
  with, rather than at `recommended_bytes`: the engine sizes its expert
  cache down on a tighter machine instead of refusing to open, so pricing
  at the recommended figure would have refused a load it would have taken
  fine.

  `serve/` goes to 179 server tests plus a new `tests/serve/test_main.py`
  of 14, for the arithmetic on its own.

- **`tests/run.sh` checks that the version is spelled consistently, and
  that the changelog was written in the same commit.** `src/waste.h`
  spells the version twice — three numeric macros and
  `WASTE_VERSION_STRING`, which `src/version.c` is what every banner
  prints — and nothing compared them. Bumping the macros for this release
  and leaving the string behind made `waste --version`, the server's
  banner and the header of the suite's own output all say 0.8.1 on a 0.8.2
  build, with every check still green, because nothing in the suite read a
  version at all. Two greps now: the string against the macros, and a
  `## <version>` section in `CHANGELOG.md` against the same figure, which
  is the rule CLAUDE.md states and the one this release had to go back and
  apply by hand.

### Fixed

- **A Qwen load no longer moves a model already open onto i8mm** (#68,
  #75). `trunk_kern` and `sdot4_sg` were file-statics that
  `matvec_t_inner`, `matvec_t_batch` and `prequant_ok` read for every
  4-bit trunk tensor in the process, and 0.8.1 made a *container load*
  write them: the right default for Qwen (LEARNED §83: 29% for a
  difference the text cannot see) but applied process-wide. So a host that
  opened a Kimi or GLM container and then a Qwen one moved the first
  container's trunk onto i8mm as well, mid-generation, and the same prompt
  through the same container gave different logits depending on what had
  been opened earlier in the same process. i8mm is explicitly not the
  exact arithmetic — that is what `WASTE_TRUNK_KERNEL=0` is for — so this
  was a numerical change and nothing reported it.

  Measured here before the fix, on two synthetic fixtures in one process:
  **253 of 256 logits** of the already-open non-Qwen model moved, max abs
  4.95e-05. The two controls that give that number meaning: a *second
  non-Qwen* container as the thing loaded second moves 0 of 256, and so
  does a Qwen container with `WASTE_TRUNK_KERNEL` pinned. After the fix,
  0 of 256 in both load orders, while a Qwen container's own logits stay
  **bit-identical** to `WASTE_TRUNK_KERNEL=2` — the isolation is not
  bought by denying Qwen its kernel — and a real Kimi-Linear container is
  byte-identical to 0.8.1. `tests/test_kernel_isolation.c` holds the
  check the issue asked for, and SKIPs loudly rather than passing on a
  CPU without i8mm or with the kernel pinned.

  `waste_model_set_sdot4()` still sets the process default a fresh load
  inherits; the new `waste_model_set_kernel(m, mode, sg)` retunes one
  model. `tests/sweep.c` and `tests/kernel_kl.c` use it, which they have
  to: both flip the kernel inside one process, and after the change the
  process-wide setter no longer reaches a model already loaded, so
  `kernel_kl` would have compared a kernel against itself and reported
  `KL mean 0.00e+00` — a passing-looking number from a tool that had
  stopped measuring. `tests/run.sh` now asserts that KL is nonzero for
  exactly that reason.

  This is also what **#70** was waiting for: the i8mm trunk kernel as
  DeepSeek-V4.1's default, measured in 0.8.1 at 3.87–3.97 → 4.52–4.56
  tok/s and held back only because a per-model default could not be kept
  to its own model. It can be now.

## 0.8.1 — 2026-09-18

**Qwen3.8-Flash-Next runs**, text only — contributed by @Skibisky in #63.
It is a fifth architecture: Gated DeltaNet rather than KDA, Qwen Sparse
Attention over the original K/V rather than a latent, four HyperConnection
residual streams, a softmax router with a gated shared expert, and a
per-layer n-gram embedding whose 16 tables live on the trunk and are read a
row per head per token. That last part is what lets a model of 176.94 B
parameters open with a 3.11 GB floor. The throughput figures in
[docs/QWEN.md](docs/QWEN.md) and LEARNED §80–93 are the contributor's,
measured on a 48 GiB Apple silicon laptop. The checkpoint is not on the
machine this release was cut on, so they have not been re-measured here.
What was measured on this commit: against the container-native oracle, the
routes match exactly at every layer, and the logits agree to 1.9e-06 on
the synthetic fixture and to 2.4e-06 on one with a shared expert 64x the
hidden width.

For anyone already on 0.8.0: **a DeepSeek-V4.1 conversion without
`--reclaim` produced a container that could not be opened** (#71). The
Engram index was never written, and nothing said so. That is fixed, and
there is a repair for containers already converted.

No ABI move: `src/waste.h` changes only its version macros.

### Added

- **The Qwen3.8-Flash-Next forward pass** (`qwen4_exp_text`): GDN, QSA,
  HyperConnection, PLE and the softmax top-k router, each checked against
  an independent PyTorch reference (`tests/test_qwenparts.c`,
  `tools/qwenparts_ref.py`) and all of them together against
  `tools/qwen_container_ref.py`. Chunked prefill is bit-identical to the
  sequential path, and so are the row split, the fixed batches and the
  per-expert staged schedule of `qwen_moe_layer`.
- **Its converter.** Nested `text_config`, packed `gate_up_proj` /
  `down_proj` split into ordinary WEXP records (the format stays v0), 128
  PLE source shards written as 16 Q8G heads a row batch at a time, and a
  `--reclaim` ledger that treats the n-gram shards as a consumer of their
  own. `--jobs` defaults to 1 for Qwen, since one worker holds a whole
  layer's packed pair. `tools/verify_container.py` reads packed sources and
  spot-checks PLE rows.
- **`tokenizer_digit_run`** — Qwen splits every digit into its own
  pre-token, where Kimi and GLM take up to three.
- **`cfg_sane` bounds every Qwen dimension that sizes a buffer or a loop.**
  The shared expert's width is bounded the way its siblings are, and
  `m->ff`, `m->xq` and `m->xs` are sized from it. `tests/run.sh`'s
  refusals now pass only when `cfg_sane` itself refused — not on any
  non-zero exit, which a crash also produces — and
  `make_test_container.py --qwen-shared N` builds a wide shared expert that
  runs under `make asan` and matches the oracle.
- **`tests/kernel_kl.c`** — two trunk kernels over a long prompt, compared
  at every position: KL, argmax, routes, and perplexity on the real text.
- **`WASTE_PROFILE=decode`** leaves the prompt steps out of the profile.
  On a Qwen container the profile also prints a per-phase tree and a trunk
  matvec table broken down by tensor role.
- **A CI guard** that checks, block by block, that every block of
  `convert.py` writing Engram tables also writes the index (#72). It fails
  on the tree before that fix, and so does a merge that puts the call in
  the wrong place.
- **A strict CI job for the K2 tool protocol**, the one GLM has had and
  which `ci.yml` used to have to exempt K2 from in so many words: "the same
  ground-truth rule the K2 template check in tests/run.sh applies, except
  this one may not skip". Now neither may. `CI_K2_ORACLE_STRICT=1` turns
  every skip path — missing template, missing jinja2, unresolved markers —
  into a failure, and the job asserts all **seven** checks ran, so it
  cannot be green by having done nothing. Vendoring the template is what
  made it possible: no download, no weights, no machine-local `~/models`.

### Changed

- **A Qwen load selects the i8mm trunk kernel for the whole process**,
  containers already open included, unless `WASTE_TRUNK_KERNEL` pins it.
  That is not exact arithmetic. For Qwen it is worth 29%, and the
  contributor measured next-token perplexity of 3.698 against 3.712 for
  f32. The side effect on other containers is #68, which tracks moving the
  choice onto each model.
- **QSA forms every product-and-sum through one definition**,
  `waste_qwen_qsa_mac`: `fmaf` where the target can fuse, two statements
  where it cannot. Left to the compiler, the four-wide loop and its
  reference rounded differently on gcc for arm64 at `-O1` and `-O2`,
  whatever the contraction flag said, and on clang at `-O1`. On arm64, selections of 32 tokens or more
  now round once per element where the branch's measurements rounded
  twice. The last bits of those numbers move. `kernel_kl` on the real
  checkpoint has been asked for.
- **`convert.py` writes tensor bytes in one copy.** `raw_bytes` used to go
  through a Python list with one int per byte, for every family's
  conversion.

### Fixed

- **DeepSeek-V4.1 without `--reclaim`: the Engram index was never
  written** (#71, #72 — found and fixed by @helenkwok converting the
  release on an EPYC 7713). Only the `--reclaim` path called
  `build_engram_meta()`, and it called it twice. The default path wrote
  110 GB of Engram tables and no `engram.json`, and `engram_open()` —
  the one silent `-2` in the load path — turned that into a bare
  `open: malformed container`. Both paths now write the index, and the
  loader names the missing file. #72 has the script that repairs a
  container converted without it.
- **The server refused connections under load.** `ChatServer` kept
  socketserver's listen backlog of 5, so eight concurrent connects had
  their surplus reset. On macOS 27 `TestConcurrency` failed 11 of 20 runs;
  with `request_queue_size = 128` it failed 0 of 30.
- **The K2 tool-grammar check defaulted to a path on an external volume**,
  which is a description of one machine rather than a default: everywhere
  else — CI, a fresh clone, this machine with the disk unplugged — it read
  as "no template" and skipped, and it is the *only* check of that grammar.
  Kimi-Linear's tokenizer carries K2's five tool-call tokens and its own
  release ships no `chat_template` at all, so without it the rendering is
  checked against a parser that reads back what the renderer wrote.

  The template is vendored at `tests/serve/k2_upstream/` with its
  provenance, upstream revision, file hash and licence, the way
  `glm_upstream/` already was; `K2_DIR` still points a real release over
  it. The suite goes to 85 passed, 0 failed, 2 skipped with nothing set.

### Measured and not adopted

- **The i8mm trunk kernel as DeepSeek-V4.1's default.** 3.87–3.97 tok/s
  becomes 4.52–4.56. Of 2,080 routing decisions it flips one, a tie the
  reference itself cannot resolve, and the 201 others that differ all
  follow from that one. It waits for #68, so that a
  per-model default does not reach other models in the same process
  (#70). Along the way it showed that the DS41 oracle gate's fixture is
  53x quieter than the real model on the same change: 6 layers against 40.
  LEARNED §94, §96.
- **`WASTE_VQ8=1` on DeepSeek-V4.1.** A further +21%, and 574 of 880
  routing decisions change, the first of them a real disagreement. That is
  a different model, not a faster one. LEARNED §96.
- **The Qwen branch's batched trunk matvec and staged expert schedule,
  ported to DeepSeek-V4.1.** Both are bit-identical, and neither makes any
  difference. Both are about work units too small to be worth a dispatch,
  and DS41's projections are 512 rows or more and its experts 2,304.
  The code is on `perf/mvb-ds41` and `perf/xpar-staged-ds41`. LEARNED §94,
  §95.
- **`WASTE_METAL_MOE=1` and the thread count on DeepSeek-V4.1.** Metal is a
  wash, and 6 to 18 threads is flat. LEARNED §96.

## 0.8.0 — 2026-09-15

**DeepSeek-V4.1-Flash runs.** 552 B backbone plus 197 B of n-gram memory,
510 GB as published, converted to a 299 GiB container and decoding at
**3.77 tok/s over 64 tokens** on a 64 GB laptop — faster than
GLM-5.3-Flash on a bank twice the size, and above the 1.5–2.5 that
[docs/DS41.md](docs/DS41.md) projected before the download started. Against
a PyTorch oracle reading the same container: 0.0025% relative L2, top-5
identical.

It is a fourth architecture rather than a variant of the three already
here — CSA2, Engram, single-pass mHC, a third router score function, a
third pre-tokenizer and a third prompt format — and bringing it up found
defects in all of them. The one that matters outside this release is the
JSON reader: `specials.json` had been read without decoding JSON escapes
since 0.6.0, which no ASCII-marked container could notice.

The other theme is a continuation of 0.7.2's. That release was about tests
that compare a thing to itself; this one is about tests that are wrong
about a thing that is right. The suite reported four failures against a
correct engine, three of which were the check — see LEARNED §79 — and each
had been green since 0.6.0 because three models happened to share an
assumption nothing had written down.

No ABI move: `src/waste.h` changes only its version macros.

### Added

- **`tools/hf_peek.py`** — read individual tensors out of a HuggingFace
  safetensors repo over HTTP range requests, dequantizing E2M1/E4M3 against
  an E8M0 scale stream. Gate 8 needed 24 experts out of 48 shards and cost
  **190 MB of 510 GB**. `tools/quant_lab.py` grew `--npy` to take the
  result. LEARNED §74.
- **The DeepSeek-V4.1 pre-tokenizer** in `src/tokenizer.c`, selected by
  `tokenizer_pattern` in the manifest and `waste_tok_set_pattern`. Three
  isolating Splits in sequence rather than one pattern, punctuation and
  symbols as classes of their own, and no contraction branch.
  `tools/hf_tokenizer.py` recognizes it and refuses anything else, as
  before.
- **`tools/gen_unicode.py` and `src/unicode_classes.h`** — `\p{L}`,
  `\p{M}`, `\p{N}`, `\p{P}`, `\p{S}` as generated range tables instead of
  hand-written blocks. 30 KB of rodata, binary searched.
- **`tools/tokdiff.py --wide N`** — a randomized corpus over the whole
  codepoint space plus multi-byte whitespace runs, and a `tests/run.sh`
  check that runs it.
- **The DeepSeek-V4.1 container format and converter.** `waste_config`
  gains CSA2 (`head_dim`, `o_groups`, `o_lora_rank`, `sliding_window`,
  `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`,
  `candidate_*`, `compress_rope_theta`), single-pass mHC, a third router
  score function (`sqrtsoftplus`) with a second selection bias for image
  spans, and Engram. `cfg_sane` bounds all of it, including the invariant
  that a compressing layer's ratio matches the last KV source's — a
  mismatch is not a shape error anywhere, it just divides the position by
  the wrong number.
- **Two rope schedules.** The window-only layers rotate at `rope_theta`
  with YaRN off and the compressed ones at `compress_rope_theta` with it
  on. This release states no `mscale`, so the shared `rope_init` would
  refuse it — and writing the two keys in to get past that check would put
  a 1.63x on the attention scale the model was not trained with.
- **`tools/ds41_engram.py`** — the compressed token map, the bucket primes
  and the hash multipliers, none of which is in the checkpoint. Checked two
  ways against what the release states: the map came out 99,092 ids against
  a stated 99,092, and the primes summed to 384,006,168 and 384,016,682
  against the two stated row counts.
- **`engram-L{n}.bin`**, streamed a chunk of rows at a time, because the
  two tables are 98 GB each and 40% of the download. `--engram-bits` is 4
  (110 GB) or 8 (209 GB); neither changes what a token reads.
- **`make_test_container.py --ds41`** and `tests/test_convert_ds41.py`.
  The container opens, and one missing `attn_sink` is refused by name —
  a per-head temperature whose absence a forward-pass diff would show only
  as drift.

- **The DeepSeek-V4.1 forward pass**, text only. CSA2 (a sliding window of
  raw KV and up to `index_topk` compressed positions in one softmax, an
  attention sink per head, the query's rotation removed from the output,
  and an output projection that is low-rank *and* block-diagonal over
  `o_groups`), single-pass mHC, Engram at two layers, and the
  sqrt-softplus router. **0.000018% relative L2** against
  `tools/ds41_ref.py`, which reads the same container — so that is
  arithmetic and not quantization — and the same on the residual stream
  after every layer. Chunked prefill is bit-identical to the sequential
  path, as on GLM.
- **`tools/ds41_ref.py`**, the oracle, and a `tests/run.sh` check that runs
  it over twelve tokens. Twelve because the test container's window is four
  slots: below five tokens the ring never wraps, no compressed cache fills,
  and the candidate filter has nothing to choose between. LEARNED §76 is
  about the two bugs that found — one in the engine, one in the oracle.

- **`serve/dsml.py`** — DeepSeek-V4.1's prompt format and its reply reader,
  wired into the server ahead of the `chat.json` fallback and behind XTML.
  A numeric reasoning effort (1–100, with low/high/max mapping onto
  50/75/100), `<think>` channels, `<｜DSML｜ calls>` tool markup,
  mid-conversation system turns, `<tool_result>` blocks in place of a
  `tool` role, and the six internal task tokens.

  Diffed against `encoding/encoding.py`, the release's five checked-in
  golden outputs included, by `tests/serve/test_dsml_upstream.py` when
  `DS41_DIR` names a release. `tests/serve/test_dsml.py` holds what a
  string diff cannot see: **which segments are markup**. `｜DSML｜` is the
  control token and the tag name is not, so `<｜DSML｜ calls>` is three
  segments and a tool result containing that literal cannot open a block.
- **Session state for DeepSeek-V4.1** — `waste_state_save`/`_load`, which
  segfaulted on this container because the shared path wrote a latent
  cache it does not have. CSA2 saves the window ring whole and, for the
  four KV source layers, the compressed latents, the index keys and the
  partly-filled pooling group; Engram saves its n-gram history, whose ids
  are *compressed* by a map only the container has, so a caller holding
  the original prompt cannot reconstruct it. `head_dim`, `sliding_window`
  and the Engram row count join the header fields a mismatched state file
  is refused on — restoring a window at the wrong width is a session that
  resumes attending to the wrong tokens, not a short read — and the size
  is checked before the first byte is read.
- **`Engine.marker_ids_for`**, so a format can state its own markers rather
  than sharing K3's four. All of them resolve or the format is refused: one
  that half-resolves is the failure the probe exists to prevent.

- **DeepSeek-V4.1's vision tower**, a third one. 32 blocks, no learned
  position grid and no q/k norms, a fused `w1` for gate and up, and a
  projector that is a 3x3 pixel-unshuffle into two dense layers.

  Two things in it are not a variant of anything already here. The rotation
  is **split-halves** — each head's dims halved and the first half rotated
  against the second — where every other rotation in this engine pairs
  adjacent elements. And the span is not the image: the LLM sees
  `[start] ([image] * n_w [newline]) * n_h [end]`, so the tower emits the
  three learned delimiters itself and the engine's media queue stays one
  row per placeholder.

  Preprocessing too: the image is contained and grey-padded rather than
  stretched, and the grid is budgeted in LLM tokens rather than in patches.
  `waste_image_plan_ds41` is that geometry on its own so an oracle can be
  asked the same question.

  6e-7 relative L2 against `tools/ds41_vision_ref.py` on five patch grids,
  three of them not multiples of the downsample; the geometry agrees on
  seven source sizes including both collapse cases.

- **`tools/pipeline.sh` takes a `MODEL`**, so the unattended
  download → probe → round-trip → convert → run → oracle path is no longer
  K3's alone. `MODEL=ds41` and `MODEL=glm` join it; everything that differs
  between the three — the repo, the default paths, the free space demanded,
  which oracle can read the container and what it needs installed — is one
  table at the top, and an unknown name is refused rather than defaulted.
  `SRC`, `OUT` and `MIN_FREE_GB` still win if set, so a profile is a
  default and not a constraint. Verified end to end on DeepSeek-V4.1: all
  six stages, `rel 2.409e-05`, argmax match, top-10 identical.

  Running it that way found a bug older than the change: the free-space
  check compared the container's full size against *free* space alone, so
  a resumed run on a finished 299 GiB container was refused for wanting
  310 GiB on a volume with 197 left. It counts what the container already
  occupies now, which is what "room for the finished container" means and
  what every resumable stage below it assumed.
- **`tools/spec_window.py`** — what a speculative batch of K tokens costs
  an engine whose budget is bytes read per token, from a real
  `WASTE_DUMP_ROUTE` trace. Gate 9 ran it on three containers: a window of
  five consecutive decode tokens touches **3.45x** the expert records one
  token does, on Kimi-Linear, GLM-5.3-Flash and K3 alike — two orders of
  magnitude of scale and two different top-k, agreeing to a tenth of a
  point at every K. So DSpark needs 3.45 of its 5 drafts accepted to break
  even on bytes. LEARNED §77.

- **Measured throughput**, replacing the projection docs/DS41.md carried
  (which is kept as written beside it). **3.77 tok/s over 64 tokens and
  3.71 over 200**, against 1.5–2.5 projected — faster than GLM-5.3-Flash
  on a bank twice the size, because the working set is GLM's to within 1%.
  It is the first container here that is *not* faster over the longer run:
  the cache is at 93% by the 64th token and there is nothing left to fill.
  docs/DS41.md has the cache-size curve, which is flat from 9.6 GB and
  declines slightly after, and the phase profile at both ends of it —
  expert I/O is 58.7% of a step at the 152 MB floor and 10.2% at 17 GB, so
  this engine is disk-bound starved and compute-bound fed.
- **`waste bench` now says when it measured a prefill.** It divides by the
  tokens actually generated, and a model that ends its turn on the bare
  continuation prompt it uses leaves that at 1: DeepSeek-V4.1 reported
  "0.19 tok/s" for one decode step behind an 18-token prefill, with
  nothing to say so. The rate was true; what it was a rate of was missing.

Not implemented: DSpark, deferred by gate 9. A container's MTP weights are
dropped at conversion.

### Fixed

**Two tokenizer defects that affect every existing container**, found by
the wide corpus and invisible to the twenty-one curated strings, which
scored 21/21 before and after. On 24021 strings, Kimi-Linear went from
22937 identical to 24017 and GLM-5.3-Flash from 22914 to 24020 — so
**about 4.5% of strings used to encode differently from the release**.
LEARNED §75.

- `\s+(?!\S)` backed off one **byte** where it must back off one
  character, cutting a U+00A0 before a word into two replacement bytes.
- `\p{N}` and `\s` were ASCII-only. Both patterns mean the Unicode
  classes; which characters are in `\s` was probed against both releases
  rather than assumed.
- `tests/run.sh` tested the tokenizer with `grep -q identical`, and
  `"22914/24021 identical"` contains that word. It now reads the counts.
- **`tools/fetch_weights.sh` reported a vanished destination as a finished
  download.** A USB enclosure dropped off the bus 184 GB into a 475 GB
  pull; `$STATE` went with it, `wc -l` produced nothing, `[ "" -lt 48 ]` is
  an error that `test` reports as false, and the run printed
  `ALL SHARDS COMPLETE` with rc=0 over a directory that no longer existed.
  The next thing that would have happened is `convert.py` writing a
  container out of 39% of a model. It now asks whether the destination is
  still there before believing anything counted from it, and refuses a
  count that is not a number. Same shape as #35 one level up.
- **A JSON reader that did not decode JSON.** `specials.json` is written
  by `json.dump`, which escapes non-ASCII by default, so
  DeepSeek-V4.1's control tokens reached the container as
  `"<\uff5cUser\uff5c>"` — and both readers, `js_str` in `src/json.h` and
  `load_specials` in `src/tokenizer.c`, copied the bytes between the
  quotes. Every marker the release has is non-ASCII, so **none of them
  resolved**: `waste_tokenize_markup` returned the same ids as
  `waste_tokenize`, which is the security boundary in §"Prompt safety"
  collapsed to nothing, DSML could not be served, and the CLI printed
  `<\uff5cend\u2581of\u2581sentence\uff5c>` where the model had emitted EOS.
  Every container before this one had ASCII-only markup — `<|open|>`,
  `<|endoftext|>` — which is why a `memcpy` where a decoder belonged
  shipped in 0.6.0 and survived every release since.

  Both readers now share one `js_unescape`, surrogate pairs included, and
  the converters write UTF-8 (`ensure_ascii=False`) so the file says what
  it holds. A container written either way loads: the escaped one on disk
  is what the regression check in `tests/run.sh` builds, since a reader
  that only works against our own writer is the same bug waiting.
- **`tools/verify_container.py` kept a second copy of how a checkpoint
  names its experts** — an inline probe for DeepSeek-V3's
  `mlp/gate_proj` with Mixtral's `block_sparse_moe/w1` as the fallback,
  while `convert.py` had the same fact in `MOE_LAYOUTS`. DeepSeek-V4.1 is
  neither (`layers.0.ffn.experts.0.w1.weight`, and no `model.` in front of
  it), so the round-trip on the newest converter was the one that could
  not run: a `KeyError` with its stderr swallowed by `tests/run.sh`. It
  imports `moe_layout` and `source_prefixes` now. On the 299 GiB
  container it passes across all 40 layers at **19.5–20.4% per-expert
  error**, which is gate 8's projected 19.95–20.74% confirmed on the
  weights rather than on 190 MB of range requests.
- **The learned-hotlist check now skips where no hotlist can hit.** It
  guarded only on the container's floor fitting under its 5G budget.
  DeepSeek-V4.1's floor is 4.86 GB, so it opens — and leaves 310 MB of
  expert cache against a 3.19 GB working set, under a tenth of one token,
  where docs/ENGINE.md section 3 says the hit rate is zero and not low.
  The check answered 284 misses -> 286 on one run and fewer on the next,
  which is a verdict decided by noise. A missing prerequisite is a SKIP.
- `mxfp4.ST` read an `int8` tensor with a `.scale` companion as if the
  int8 were the values. That is DeepSeek-V4.1's spelling for packed fp4,
  and no shape disagrees; it now refuses an int8 tensor with no scale
  beside it rather than guess which of the two it is.

## 0.7.2 — 2026-08-28

Six pull requests and two issues, and one theme running through all of
them: **a test that compares a thing to itself is not a weak oracle, it is
not an oracle.** Three of the defects this release fixes were sitting under
green boards, and each one was found by diffing against something outside
the code rather than by a check inside it.

No ABI move — `src/waste.h` is untouched — and no measured number in the
docs changes. The engine gains one optional capability, bank striping, and
two refusals.

### Added

- **Bank striping across N devices** (#53). `WASTE_BANK_SHARDS=/mnt/a,/mnt/b`
  reopens each layer's bank as N shard files; expert `e` lives on shard
  `e % N` at offset `(e / N) * rec_bytes`. Round-robin rather than
  per-layer because the per-token demand is k experts of *one* layer, so
  layer-granularity placement would leave every read of a token on one
  drive. `tools/split_banks.py` produces and byte-verifies the shard sets.
  Unset is the N=1 case of the same code path, and no container format
  changes. Striped and unstriped logits are byte-identical, under
  ASan/UBSan too.

  **No bandwidth number**, deliberately: that needs a multi-drive rig with
  a real container, and this is the mechanism. The 1.8-2x it targets is a
  prediction, not a measurement, and is recorded as one.

- **Kimi K2 native tool calling for containers served from `chat.json`**
  (#50), in `serve/kimitools.py` — its own module beside `xtml.py`,
  because it is neither of the two formats `serve/` renders: it is carried
  in a container's *tokenizer* while that container's `chat.json` says
  nothing about it. Kimi-Linear carries the five markers; GLM does not and
  is still refused by name. The split is by subject: whether a container
  can do tools is a fact about its `chat.json`, so `chatfmt.py` decides it;
  how a call is spelled is a fact about the protocol, so `kimitools.py`
  owns it and raises its own `KimiToolError`, the same shape `xtml.py` has.

- **`tests/serve/test_chatfmt_upstream.py`**, the oracle that made the
  above trustworthy (`docs/LEARNED.md` §73). The 438 lines of tests that
  came with #50 all passed and the rendering was still wrong: they fed the
  renderer's output to the parser, which agrees with itself whatever the
  format is, and one of them *pinned* the defect. Diffed against K2's
  published `chat_template.jinja`, everything matched byte for byte except
  the turn a tool result opens. `K2_DIR` names the release; the check needs
  no weights and no container, which makes it the cheapest oracle here.

- **A container the suite can reach at `index_bits 6`** (#48).
  `vq_rows_p6` had none — the synthetic container was always `index_bits 8`
  and a real conversion costs hours — so every green run covered VQ3R and
  nothing covered VQ4P, on any platform, which `CLAUDE.md` warns about by
  name. Four checks, and the arm is not decorative: swapping `T1`/`T2` in
  `vq_rows_p6_neon` fails it, naming the specific verdict.

- **`tests/test_vq4p_packing.py`** (#56), which runs
  `make_test_container.py`'s VQ4P packing against `convert.py`'s. The arm
  above cannot: it compares the engine against itself, so a generator that
  packed differently would produce a container both backends decode the
  same wrong way and all four checks would pass. Verified not to be
  decoration by breaking the generator two ways.

### Fixed

- **A `python3` that exists and is not an interpreter said the engine was
  broken** (#51, #52). On Windows the name on PATH is usually the Microsoft
  Store App Execution Alias: a zero-byte reparse point that `command -v`
  finds, that exits 49, and that prints an advert instead of running
  anything. Every guard in the suite was written for an *absent*
  interpreter. Measured on a stock Windows box: **7 passed / 7 failed / 16
  skipped before, 44 / 0 / 13 after**, with the Store's advertisement text
  appearing verbatim inside failure messages. The suite shims PATH; the
  Makefile and the two shell tools resolve a name.

- **The other half of that** (#54): seven checks still reported the engine,
  the budget, the converter and the release's own preprocessing config
  broken when *no* interpreter answered. 14 failures before #51, 7 after,
  **0 now**, with every skip naming the interpreter. 77 rather than a new
  number — `tests/check_budget.sh` already meant "could not measure, skip
  loudly" by it.

- **The turn a tool result opens is named for the tool**, `name or "tool"`,
  never `system`. K2's own template renders
  `<|im_system|>{{ message.get('name') or message['role'] }}<|im_middle|>`.

- **Two claims about shards that were not true.** #53's comment promised
  that a short or long shard fails the load; `bank_open`'s `rec_bytes`
  parameter is `(void)`-ed and nothing measured anything. A shard short by
  one record **loaded, ran, and produced correct logits** — correct because
  the missing expert simply was not routed to in an 8-token prompt, which
  is the hazard rather than the reprieve. #55 then added the size check at
  open, which is what should have been written in the first place:
  `waste_file_size` has been in `platform.h` all along.

  Its own justification is corrected too. A misplaced record cannot be
  served as another expert's — every record carries the expert it belongs
  to and `record_check` reads it, and two shards padded to the right sizes
  but carrying another split's records refuse on the first read. What the
  check buys is *when*: a long shard loaded happily and generated correct
  output, and a short one waited for the router to reach the missing
  expert.

- **`WASTE_BANK_SHARDS` truncated at 1024 bytes was ignored**, silently
  dropping shards. A shard count that disagrees with the split that
  produced them is a different layout, not a smaller one. Refused.

- **The `O_DIRECT` probe used a stack buffer**, which broke every container
  load on `linux-arm64` — 25 passed, 29 failed. `O_DIRECT` rejects a
  transfer whose *buffer* is unaligned, not only its offset and length.
  It passed on macOS, whose `F_NOCACHE` has no such requirement, and on the
  x86_64 runner, whose filesystem declines `O_DIRECT` and falls back to
  buffered: **56 passed / 0 failed on the same code and the same OS as the
  job that failed**. Two platforms agreeing is not coverage when only the
  third exercises the path. `main` was red for one commit.

- **A refusal check had two outcomes where it needed three.**
  `rope_refused` called anything that was not the expected message "loaded
  instead of being refused", so a `test_forward` that was killed — which
  says nothing — was reported as a container loading that should have been
  rejected.

### Also

CI's SPDX glob gained `tests/*.py`, `tests/*.sh`, `tests/serve/*.py` and
`serve/*.py`. Every file in those directories already carried the header;
the glob had simply never looked, three separate times this release.

`route_verdict` takes its reference as a parameter, so the three arms that
ask it — VQ3R, rotated, VQ4P — share one copy instead of open-coding it.

## 0.7.1 — 2026-08-27

**0.7.0 did not build on x86_64 Linux, on Windows, or under ASan.** That is
the reason this release exists; if you are on any of those, 0.7.0 is not
installable and this is the fix. Apple Silicon was unaffected, which is why
it shipped.

The rest is the test suite learning to tell a tie from a defect. Four checks
were red or blind across the three models, and every one of them turned out
to be measuring a discrete choice by the distance between logits — a
question that measurement cannot answer. None was an engine bug. The engine
changes here are two: the i8mm guard above, and one printf format.

### Fixed

- **The i8mm call site is guarded on the same predicate as its source file**
  (`docs/LEARNED.md` §72). `model.c` declared and called
  `waste_mvq4_rows_i8mm` unguarded, but the Makefile adds `src/simd_i8mm.c`
  to `SRC` only for `arm%|aarch64%`, so on every other platform the symbol
  does not exist:

  ```
  model.c:805: undefined reference to `waste_mvq4_rows_i8mm'
  ```

  Three of five CI jobs failed at the link, and one of them is `asan +
  fuzz` — so the sanitizers had never run against anything 0.7.0 added.
  `simd_i8mm.c` was already careful, defining the symbol twice so it exists
  in every ARM build even where `-march` did not take; what was missing is
  that **a dispatcher and the list of files that satisfies it have to be
  guarded on the same predicate**. `TK_I8MM` is already unreachable off ARM,
  so this is dead code being compiled, not behaviour. Verified rather than
  assumed: compiling `model.c` for x86_64 leaves the symbol undefined at
  0.7.0 and unreferenced now, a full x86_64 build links, and the arm64
  object still calls it.

- **The two K3 checks were measuring the router, not the arithmetic**
  (`docs/LEARNED.md` §71). Chunked prefill and the CPU backend each differed
  from the default path by max-abs 0.2858 against a 1e-3 threshold, and
  0.7.0 shipped them red on the explanation that this was the i8mm/SMLAL
  trunk kernels. **It was not** — `trunk_kern` defaults to `TK_F32`, so
  neither kernel was in the run.

  It is one routing tie. Of 1472 routing decisions in a 16-token K3 prefill,
  the earliest the paths disagree on is token 12, layer 56, experts 889 and
  712, at a **relative margin of 7.311e-07 — the minimum over all 1472**.
  The 1st percentile is 4.4e-05 and the median 7.4e-03; the 47 differences
  that follow average 5e-03, because they are computed on a hidden state
  that has already moved. Two independent paths produce byte-identical route
  traces and differ from the default in the same 48 places: NEON summation
  order against scalar, disagreeing by 1e-08 on a decision that needed
  1e-07.

  A top-K router makes an arbitrarily small arithmetic difference discrete,
  so past the first flipped expert the logit distance measures how much the
  model cares which of two indistinguishable experts it ran. No threshold on
  it works: 1e-3 fails on a tie forever, and the 0.3 that would pass could
  not catch a broken kernel.

  `tests/route_diff.py` asks the question the threshold stood in for, over
  the route and score traces `WASTE_DUMP_ROUTE` / `WASTE_DUMP_SCORES`
  already wrote: **identical**, **tie** (the first disagreement is under a
  relative margin of `--eps`, default 1e-5) or **diverged**. The default
  sits in the empty decade between the tie that flipped and the tightest
  call that held. The three chunked/backend checks now use it, and the
  result is a stricter suite, not a looser one — the argmax became a hard
  failure on every path rather than a clause on one, a route that flips on a
  resolvable margin fails while naming the token and layer, and a threshold
  miss with the routing *unchanged* — the case that is a real arithmetic
  defect — is its own verdict instead of being pooled with the tie.

- **`WASTE_DUMP_SCORES` prints `%.9g`.** The dump exists to say how close a
  ranking decision was, and six digits cannot resolve one: the two scores
  above both printed as `0.112161` while selecting differently, which reads
  as a selection bug. Nine digits round-trip a float.

- **The GLM oracle check was failing on both Linux jobs, on an exact tie**
  (`docs/LEARNED.md` §72). It had been red since 0.7.0, saying only "the GLM
  path diverges from the oracle". linux-x86_64 and linux-arm64 report rel L2
  0.00783997 and 0.00783999 — **two different ISAs agreeing to five
  significant figures is not a platform difference** — and at the worst
  logit macOS, both Linux runs and the shipped fixture all print -7.3257,
  while only the freshly generated Linux oracle prints -7.43141.

  At layer 2, token 15 the four visible pools score 0, 0, 0 and 0.00164 with
  `keep=2`: pool 3 wins outright and the second slot is an **exact
  three-way tie**, margin 0.000e+00. The engine takes pool 1;
  `torch.topk`, called in `kimi_ref.py` with `sorted=False` and so free to
  answer in any order, takes pool 2 on Linux and pool 1 on macOS. One pool
  of four tokens attended differently is worth rel L2 0.0078 in the logits,
  and it is a defect in neither.

  `kimi_ref.py` already wrote the trace that says this, under the same
  `WASTE_DUMP_DSA` the engine uses, and its own comment says why: so the two
  selections "can be diffed directly rather than inferred from a logit
  difference". **Nothing was diffing them.** `tests/dsa_diff.py` now does,
  reporting the same three answers `route_diff.py` does over the pool
  ranking instead of the expert ranking.

  Recorded because the measurement is the useful part, this is what it was
  **not**: not a stale fixture (the engine matches it to 5.72e-06), not
  macOS skipping the real comparison, not torch's version (`uv` resolves the
  same 2.13.0, fla-core 0.5.2 and einops 0.8.2 on both), not
  `-ffp-contract`, not thread count, not UB under ASan/UBSan, and **not a
  router tie** — the tightest top-2-of-8 boundary gap in that prefill is
  9.4e-03.

- **A refusal check has three outcomes, not two.** `rope_refused` grepped
  `test_forward`'s output for the expected refusal and called anything else
  "loaded instead of being refused". A `test_forward` that is killed says
  nothing, and saying nothing is not the same as loading a container it
  should have rejected; the check could not tell them apart and reported the
  more alarming one. It now separates them and prints what came back either
  way.

- **Checks say what they saw.** Every comparison above now reports magnitude
  and location on failure rather than a bare verdict. Two of these were
  investigated for hours against a bare FAIL, and the answer each time was
  four numbers the check already had in hand.

## 0.7.0 — 2026-08-27

A third model, and the two things that had to change for it to be worth
running. **GLM-5.3-Flash** — 313 B parameters, text and images — converts
and runs at 3.86 tok/s on a 64 GB laptop and 2.99 on the 12 GB a 16 GB
machine resolves, against K3's 0.45-0.62. Most of the engine already ran
it; the three things that did not are mHC, a clamped SwiGLU and DeepSeek
Sparse Attention, and each is behind a config key absent everywhere else.

The two engine changes are not GLM's. The automatic budget stopped three
working sets short of the machine, and the thread pool spent 54 us waking
efficiency cores for jobs too small to hide it — 150 times a token.
Kimi-Linear went 12.67 to 17.22 tok/s over 200 tokens on the two together;
K3 is byte-for-byte unchanged by both, because at 465 GB read per 20 tokens
neither residency nor dispatch is where its time goes.

This release also carries the `exp1` kernel work: the i8mm and SMLAL trunk
matvecs, the register-held int8 VQ3R table, the Metal VQ3R apply, and a
thread pool that knows which of its cores are fast.

**ABI move.** `waste_memplan` gained `bank_bytes` at the end of the struct,
so anything that allocates one — `serve/engine.py` mirrors it field for
field — has to be rebuilt against the new header.

**Two K3 checks fail on this release and failed identically before it.**
Chunked prefill and the CPU backend each differ from the default path by
rel L2 0.0176 (max abs 0.2858, argmax unchanged) against a 1e-3 threshold.
That is the i8mm/SMLAL trunk kernels, whose K3 accuracy `CLAUDE.md` already
records; it is not a regression introduced here and it is not resolved
here either.

> **Corrected after release.** The failures are real and the numbers above
> are right, but the cause named here is not: `trunk_kern` defaults to
> `TK_F32`, so neither kernel was in the run. It is a single routing tie at
> a relative margin of 7.3e-07, and the checks were measuring the router
> rather than the arithmetic. See 0.7.1, above, and `docs/LEARNED.md` §71.

### Added

- **A thread pool that knows which cores are fast**
  (`docs/LEARNED.md` §47, `docs/EXP1.md`). `waste_perf_cpu_count()` answers
  6 here from `hw.perflevel0.logicalcpu`, big.LITTLE `cpu_capacity` on
  Linux, and 0 where the question has no answer — which makes the whole
  thing a no-op on a uniform machine. `waste_parallel_for_fast()` cuts work
  for that group and `WASTE_WIDE` selects the call sites; the default is the
  VQ apply and the LUT build. Bit-identical, and worth **1.39x on
  Kimi-Linear** and nothing on K3, whose applies are 4.7x larger and use
  every core.

  The first version measured exactly nothing — 1.181 against 1.182 tok/s —
  because one condvar meant a fast job still broadcast to all eighteen
  threads and the twelve efficiency-core ones woke, took no chunk, and
  decremented the counter the barrier waits on. Two start signals fixed it.

- **`WASTE_TRUNK_KERNEL`: i8mm, SMLAL and SDOT for the Q4G trunk matvec**
  (`docs/EXP1.md` §2c). i8mm is **1.18x end to end on K3 at top-8** — the
  trunk matvec goes 56 to 105 GB/s — at a per-matvec relative error of
  1.4e-08, four orders of magnitude under SDOT's and below f32
  re-association noise. `src/simd_i8mm.c` is its own translation unit with
  its own `-march`, like `simd_avx2.c` on x86: written inside `model.c` the
  kernel compiled to an empty function, the runtime feature bit said yes
  anyway, and the logits came out 109% off with nothing reporting a problem.

- **`WASTE_VQ8=1`: the VQ3R apply through a register-held int8 table.**
  `vqtbl4q` against an int8 shadow instead of walking a 1.34 MB fp32 one —
  four lookups over c, c-64, c-128, c-192 added together, since `vqtbl4q`
  answers 0 past index 63. **1.76x on the kernel**, 1.19x on the bucket,
  1.05x end to end. Off by default: route set agreement is 89-97% against
  the fp32 table.

- **`WASTE_METAL_MOE=1`: the VQ3R apply on the GPU** (`docs/EXP1.md` §5b).
  One command buffer with a concurrent compute encoder per batch, each
  dispatch binding that expert's own page-aligned cache slot. Correct to
  rel L2 3.6e-07, worth 1.14x on Kimi-Linear and a wash on K3. Concurrency
  inside one command buffer is the design and it is measured: eight
  separate buffers reach 41 GB/s of index, eight concurrent dispatches in
  one buffer reach 126.

- **Instruments for ranking kernels**, because the obvious metric says the
  most accurate one is the second worst. `WASTE_PROFILE` gained a
  trunk-matvec bucket — 28.5 GB of Q4G per K3 token, three quarters of what
  a decode step reads, had been counted inside `kda`; `WASTE_TRUNK_CHECK=1`
  runs the f32 reference beside the selected kernel on real activations;
  `tests/sweep.c` reports route agreement as *set* and *rank* separately,
  because a rank swap between two near-tied scores is invisible in the
  output and counting it as a mismatch turns 88% into 56%.

- **`WASTE_DUMP_DSA`**, the sparse-attention selection: which pools won and
  on what scores. A logit diff can say two implementations disagree; only
  this can say whether they disagree about the ranking or about a tie.

- **GLM-5.3-Flash, text-only** (`docs/GLM.md`, `docs/LEARNED.md` §65).
  `zai-org/GLM-5.3-Flash` — 313 B parameters, 328 GB of fp8, published
  2026-08-26 — converts and runs. Most of it was already here: the KDA
  recurrence, the absorbed MLA, the sigmoid/top-k router, the fp8 reader and
  the nested `text_config` layout are K3's, and `qk_rope_head_dim` 0 with
  `mla_use_nope` means the rotary is not reached at all. Three things are
  new, each behind a config key absent on every other container:

  - **mHC** (`hc_mult`, `hc_sinkhorn_iters`, `hc_eps`): four parallel
    residual streams instead of one, collapsed into the sublayer by learned
    weights and scattered back through a Sinkhorn-projected doubly-stochastic
    matrix, twice a layer. The final collapse is an unweighted mean.
  - **A clamped SwiGLU** (`swiglu_limit`): gate clamped above, up on both
    sides, before the SiLU. The family's three activations are now one
    function, `waste_act_pair_range`, rather than an if/else repeated at
    seven call sites.
  - **DeepSeek Sparse Attention, k-pool** (`index_topk`, `index_kpool`,
    `index_n_heads`, `index_head_dim`): a full-attention layer scores pools
    of four cached tokens and attends over the best `index_topk/index_kpool`
    of them plus the unfilled tail. A pool's compressed key is one vector per
    four tokens, computed once — 128 B per token per layer, a fourteenth of
    the latents. Below `index_topk` tokens of context the selection is every
    visible token, so short prompts take the dense path exactly rather than
    approximately.

  Measured against a PyTorch oracle on a GLM-shaped synthetic container:
  **5.7e-6 max abs** on the last token's logits, against the suite's 1e-3
  threshold and the same order as the Kimi baseline. The sparse branch is
  really reached — the same weights with `index_topk` raised above the prompt
  give logits 0.56 apart.

- **GLM-5.3-Flash converted and run for real** (`docs/GLM.md`,
  `docs/LEARNED.md` §68). 306 GiB of shards down, ~45 minutes of conversion
  at `--jobs 3`, and a **112 GB container** — 5022 MB of trunk and 42 expert
  banks of 2598 MB. `waste info` reports 313.33 B parameters, 16.74 B active
  per token.

  Against the PyTorch oracle on the real container: **rel L2 2.41e-5**, max
  abs 2.39e-4, argmax and top-10 identical in the same order. VQ3R holds on
  GLM's expert distribution without retuning — 0.1951 relative error on a
  real `gate_proj`, against the 19.59-19.77% `docs/K3.md` records for K3.

      $ waste run ~/models/glm53.waste "The capital of Italy is" -n 20
      The capital of Italy is Rome. Rome is the largest city in Italy and is
      known for its rich history, iconic landmarks such
      [20 tokens, 5.07 s, 3.94 tok/s | experts 5816 hit / 904 miss = 87%]

  `waste bench`: 2.40 tok/s over 64 tokens, **3.09 over 200** at 89.3% hit,
  on a 64 GB machine — five times K3's 0.45-0.62. The floor is 5.14 GB and
  §66's ladder resolves a 41.36 GB expert cache, thirteen working sets.

- **`chat.json` gained `prelude`, `think` and `stop`, and GLM is now usable
  as an instruct model** (`docs/GLM.md`, `docs/LEARNED.md` §69). Each field
  exists because GLM's format cannot be written without it: it opens every
  conversation `[gMASK]<sop>` (belonging to no role), its turns end when the
  *next role marker* begins rather than with a suffix, and its generation
  prompt always opens a reasoning channel the model closes itself. Written
  without `stop`, the format answers correctly and then runs on into the
  next turn, answering questions nobody asked — the container's
  `eos_token_id` is `<|endoftext|>`, which is right for a raw continuation
  and is not what ends a chat turn.

  `examples/chat-glm53.json` is that file and the converter installs it.
  Both Kimi formats are unchanged: `stop` defaults to the assistant suffix,
  which is what they have always used.

- **The server serves a reasoning channel.** `serve/chatfmt.py` said a
  chat.json container has no think markup and refused a request that asked
  for one — true of both Kimi containers, false of GLM. `PlainParser` now
  splits on the channel's close marker and `/v1/chat/completions` returns
  `content` and `reasoning_content` separately, streaming included.
  `reasoning_effort` maps to GLM's own spelling of it, a system turn, via
  an optional `effort` field; a format that does not say how still refuses
  rather than dropping it. `thinking: false` is refused for GLM, whose
  template has no path that leaves the channel closed.

- **GLM-5.3-Flash's vision tower** (`docs/GLM.md`, `docs/LEARNED.md` §70).
  `src/vision.c` holds two towers now, dispatched on `tower` in
  `vision.json`; a file without the key is K3's. GLM's is 24 blocks with 2D
  rope instead of a learned grid, per-head RMSNorms on q and k, biases
  everywhere, a clamped SwiGLU, a two-slot temporal patch and a gated
  merger. **Against a new `tools/glm_vision_ref.py`: rel L2 3.3e-5** on the
  real 563.6 M tower and 7.7e-7 on the synthetic one `tests/run.sh` now
  builds, so the tower is in CI.

  Its preprocessing needed two things exactly right. The patch order is
  block-major over merge blocks, not raster — raster rotates every patch by
  someone else's position and leaves the output plausible — and is checked
  against the reference's own reshape at **max abs difference 0**. And the
  resize is antialiased bicubic: a bilinear sample matched torch's bilinear
  to the bit and the release's kernel to only **7.7% relative**, which is a
  different image rather than a noisier one, so the kernel is implemented
  (separable, cubic a = -0.5, support widened on a downsample) and lands at
  max abs 4e-5 against torch.

  `chat.json` gained an `image` block, so `waste run --image` and
  `/v1/chat/completions` with an image part both work. The tower is 282 MB
  at 4 bits and loads only when a caller asks for images — which needed its
  own fix: the skip was written as "outside `tensor_prefix`", the same set
  as "the tower" on K3 and the empty set on GLM, so its 282 MB were
  resident on every text-only open and counted in the floor.

- **The DSA sparse branch, verified on real weights.** The oracle run in
  §68 exercised the dense branch — `index_topk` is 2048 and the prompt was
  four tokens. Cloning the container with symlinks and an `index_topk` of
  16 reaches the sparse one at 24 tokens on the same 112 GB of weights:
  **the selections are identical, 55 of 55**, compared directly through a
  new `WASTE_DUMP_DSA` trace rather than inferred from a logit difference.
  At the real setting and 2100 tokens the branch fires on all 11 MLA layers
  for positions 2051-2099, keeping 512 pools of 513.

- **`tools/hf_tokenizer.py`**, re-encoding a `tokenizers` `tokenizer.json`
  into the tiktoken rank file a container carries. GLM ships no rank file.
  It refuses rather than approximates when the pre-tokenization pattern is
  not the one `src/tokenizer.c` implements, or when the merge list is not
  ordered by the id of what it produces — that ordering is what makes
  merge-by-rank and merge-by-list-position the same encoder.

- **`tokenizer_han_split`**, and `waste_tok_set_han_split` behind it. Kimi's
  pre-tokenization pattern gives Han its own `[\p{Han}]+` branch and GLM's
  does not, so a Han run touching a Latin one is two pre-tokens there and one
  here. Sixteen tokens in GLM's vocabulary cross that boundary and they are
  ordinary words: `A股` is `111321` on GLM and `32 98963` with the Han
  branch. Default is the Kimi pattern; `tools/tokdiff.py` now picks its
  reference from what the source ships (tiktoken or `tokenizers`) and covers
  the boundary. 21 of 21 identical against the real GLM vocabulary, 21 of 21
  against Kimi-Linear's.

- **The automatic budget climbs to the whole expert set** (`docs/ENGINE.md`
  §3, `docs/LEARNED.md` §66). The ladder's top rung was `floor + 3x` a
  token's working set, which is what `recommended_bytes` means and the wrong
  ceiling once the machine is known: on a 64 GB laptop Kimi-Linear's entire
  17.17 GB expert set fits and the default gave it 1.65 GB of cache. It now
  starts at however many working sets cover the container's bank — the point
  past which a cache cannot be improved by making it larger — and walks
  down, capped at the bank. `waste plan` reports that budget as **fully
  resident** and `bank_bytes` is published in the plan.

  Kimi-Linear, `waste bench`, defaults, 200 tokens: **12.67 -> 14.81 tok/s**,
  hit rate 70.7% -> 96.3%, **66.27 GB -> 17.75 GB read**. K3 is byte-for-byte
  unchanged (0.3251 tok/s, 465 GB) — its bank is 962.83 GB and no rung above
  `floor + 1x` fits under the cap, which is still three quarters of what the
  process may use.

- **`moe_layer` picks its parallel path from the cache, not from a flag.**
  §44 measured one task per routed expert at 1.18x on Kimi-Linear and a
  regression on K3, and the reason it gave was the answer: holding K records
  before doing any arithmetic is a barrier against the read-ahead — free
  parallelism when they are resident, a stall when they are not. The layer
  now asks (`waste_ecache_resident_all`, top_k hash lookups). Kimi-Linear
  12.18 / 14.51 / **14.25** for XPAR=0 / XPAR=1 / automatic; K3 0.3251 /
  0.1729 / **0.3251**. `WASTE_XPAR=0/1` still forces it.

  The two paths are bit-identical and `tests/run.sh` now asserts it: a
  scheduling choice made from cache state that also moved the numbers would
  make an answer depend on how warm the cache happened to be.

- **A background fill for a cache that can hold everything.** With the bank
  resident, 200 tokens still discovered 13.2 GB of it as 3443 separate
  demand misses. A thread now walks the banks in file order and fills empty
  slots (`waste_ecache_admit`: never evicts, never counts a hit or a miss,
  counts its bytes). It starts only when every record fits — below that,
  file order is a worse judge of what belongs in a slot than LFRU is — and
  it is a win at every length measured, including 16 tokens, where it reads
  three times the bytes and still comes out ahead. `WASTE_PRELOAD=0` turns
  it off.

- **The thread pool spins before it parks, and small jobs stop waking the
  slow cores** (`docs/BACKENDS.md`, `docs/LEARNED.md` §67). With the expert
  set resident a Kimi-Linear step profiles as arithmetic — and a third of
  its dispatches were going into thread wake-ups. Measured with an empty
  kernel on this 18-core machine (6 performance, 12 efficiency): **54.1 us**
  to dispatch to the parked full pool, **1.5 us** to the spinning fast
  group. A decode step dispatches ~150 times.

  Workers now spin on a packed (epoch, workers) word before falling back to
  the condvar (`WASTE_SPIN`, 0 restores the old pool), the chunk cursor and
  completion count are atomics, and a dispatch only reaches for the slow
  group when the weights it touches exceed `WASTE_WIDE_MIN` — 4 MB by
  default, a measured optimum rather than a trend (0 gives 14.86, 2 MB
  16.56, 4 MB 16.62, 8 MB 15.84, infinite 14.91).

  Kimi-Linear, 150 tokens: **14.41 -> 16.74 tok/s**, three runs each. K3 is
  neutral inside its +-9% spread — it reads 465 GB for 20 tokens and 8 ms a
  token of dispatch is not where its time goes. On a machine with one kind
  of core the threshold selects between the pool and itself, so nothing
  changes.

  **§47's inversion is gone with it.** Six threads used to beat eighteen on
  Kimi-Linear (14.49 against 14.46) and now eighteen wins, 16.97 against
  15.35. Capping the pool was buying "no parked threads to wake", not
  "fewer, faster cores".

  The logits are bit-identical across every combination of `WASTE_SPIN`,
  `WASTE_WIDE_MIN` and `WASTE_THREADS`, and `tests/run.sh` asserts it.

- **Five checks and one fixture.** `tests/run.sh` builds its own GLM-shaped
  container — seed 0, byte-reproducible, a few megabytes — because none of
  the checks that run on a Kimi reach any of the three new things and all
  three fail quietly. `tests/test_convert_glm.py` covers the converter's
  config handling separately, against what the release says rather than
  against the container the converter wrote.

### Fixed

- **The QoS demotion was a 25% regression, and it was paying for its own
  speedup.** The fast group was created at `QOS_CLASS_USER_INTERACTIVE` and
  the rest at `QOS_CLASS_UTILITY` — a class the Apple Silicon scheduler
  answers with an efficiency core, so every full-pool job lost twelve of its
  eighteen threads. K3 top-8: **0.957-0.984 tok/s with no split at all,
  0.745-0.749 with the split present and no call site using it.** A
  regression from a line that dispatches no work differently. The fix is one
  clause: raise the fast group and leave everyone else as they were.

  It was also inflating what the fast-core pool was credited with. With it
  gone that pool is worth 1.39x on Kimi-Linear and nothing on K3, where this
  branch had claimed 1.28x and 1.11x.

- **GLM's `kda_layers` is 0-based and a WASTE manifest's is 1-based.** Copied
  through it puts KDA on the wrong layers: every tensor is found, every shape
  checks out, the container loads, and the model answers noise. `convert.py`
  rebuilds the list from `layer_types` instead. `eos_token_id` is a list of
  three on GLM and one id in the engine's config; the first is taken.
  Neither is visible to an oracle diff, because the oracle reads the same
  manifest.

- **Two converter fixes the real conversion found.** GLM nests the text
  model the other way round from K3 (`model.language_model.layers.N` against
  `language_model.model.layers.N`, and `lm_head` outside the wrapper rather
  than inside), which no `tensor_prefix` can reconcile — caught statically
  against the checkpoint's index before converting anything. And
  `build_trunk` dropped every tensor whose name ends in `_scale`, which is
  the companion of a quantized weight *and* GLM's 90 learned `hc_attn_scale`
  / `hc_ffn_scale` parameters; the container converted and then refused to
  open. The test now matches `.weight_packed` / `.weight_scale` /
  `.weight_scale_inv`, which also stops 89 fp8 companions being written into
  the trunk as if they were weights.

- **`tools/fetch_weights.sh` recovers from a shard longer than
  Content-Length.** Killing it mid-shard and restarting left 9.03 GB where
  the shard is 5.36 GB; `curl -C -` then asked for a range past the end and
  every retry failed the same way, on a shard whose bytes were all on the
  CDN. It is deleted and refetched now.

- **Two suite checks stopped reporting a missing prerequisite as a defect**,
  which this repo's rules forbid. `WASTE_REF_SRC` defaults to Kimi-Linear's
  weights, so pointing `WASTE_REF_MODEL` at K3 round-tripped K3's container
  against Kimi-Linear's safetensors and called it a converter bug; and the
  hotlist check opens at a hardcoded `--budget 5G`, below K3's 29.19 GB
  floor, so the engine refused and the check read that as the hotlist doing
  nothing. Both SKIP with the reason now.

### Measured and not adopted

- **An f32 trunk.** With the experts resident the profile is 49.7% trunk
  matvec, so the obvious next thought is to spend the spare RAM there too.
  `WASTE_Q8=0` on Kimi-Linear: **12.74 tok/s against 15.02**, eight times
  the trunk RAM for a 15% loss. The quantized matvec is memory-bandwidth
  bound and dequantizing makes it worse.

- **Chunked prefill for mHC containers.** GLM's prefill runs at decode
  speed because it takes the per-token path, and teaching the chunked path
  mHC looked like the fix. On the model that already has that path, warm
  cache: 4063 ms against 4075 for 64 tokens, 27067 against 27231 for 512.
  Nothing, at either length — 82.8% of a chunked prefill is the VQ apply,
  and a chunk expands nearly the whole bank. Chunking trades disk reads for
  VQ decode, which is a loss when the reads were already cached.

### Not implemented

- **Tools, for a container served from `chat.json`.** GLM's template carries
  a full tool-call protocol and four strings cannot express one; it is
  refused by name rather than half-rendered. The raw `.jinja` is in the
  container for a host that does interpret Jinja.
- **Chunked prefill on a GLM container**, which falls back to the per-token
  path — and measured not to matter, above.
- **Cross-layer top-k sharing** (`indexer_types` other than `"full"`), which
  this release refuses to convert rather than produce; the MTP layer, which
  is dropped; and video.

## 0.6.9 — 2026-08-24

Almost none of this is the engine. It is the space no CI job could reach —
Windows, real containers, and the four feasibility gates that were still open —
and most of it was found by people running hardware this project does not own.
`GATES.md` now has no open gate for the first time, and two of the three that
closed this cycle closed *against* the change they were proposing.

**No recompile needed.** `src/waste.h` changed only in a comment this cycle, so
unlike 0.6.8 there is no ABI move. `waste_kernels` gained a slot, but that lives
in `src/waste_backend.h` and is not public.

### Added

- **VQ4P's apply is a dispatch slot, and x86 has a kernel**
  ([#38](https://github.com/sqliteai/warp/issues/38),
  [#41](https://github.com/sqliteai/warp/pull/41)). `vq_rows_p6` was an
  `#ifdef` in `model.c` with an ARM-only body, so §47's tuning did not travel.
  It is now `waste_k.vq_rows_p6`, with the NEON body moved behind it unchanged
  and an AVX-512 VBMI implementation registered beside it. The refactor is
  **bit-identical on ARM against a real `index_bits: 6` container**, checked
  here on both a VQ4P and a same-geometry 8-bit control. The VBMI kernel itself
  is `compiled and dispatched, never executed` — no machine here has AVX-512,
  the same standing caveat `docs/BACKENDS.md` records. It carries
  `__attribute__((target("avx512vbmi")))` so a VBMI instruction cannot be
  placed in the F+BW kernels that share its translation unit and SIGILL on
  Skylake-SP.

- **`WASTE_DUMP_SCORES=path`** ([#45](https://github.com/sqliteai/warp/pull/45)),
  the whole router vector per (token, layer), not only the winners. It records
  `score[e] + bias[e]` — what the selection loop ranks on — rather than `w[j]`,
  which takes `score[best]` without the bias and is applied after the choice is
  made. A residency prior, a dynamic-k rule and a pruning criterion are all
  questions about the experts that *lost*, and a trace of the chosen cannot
  answer them.

- **`WASTE_CCR_LAMBDA`** ([#46](https://github.com/sqliteai/warp/pull/46)), a
  cache-residency bonus applied to ranking only, off unless set
  ([arXiv:2412.00099](https://arxiv.org/abs/2412.00099)). `w[j]` still takes the
  untouched `score[best]`; only which K run moves. Reproduced here on
  Kimi-Linear at `--budget 4G`, `-n 120`: hit 76.1% → **89.5%** at lambda 0.10
  and bytes per token 0.231 → 0.167, a 27.7% saving. See *not adopted* below
  for why the default did not move.

- **`waste plan` prints one token's working set.** It quoted the quantity in
  the line below it and never showed it, so the only way to see the number was
  to compute it — and computing it from `FORMAT.md`'s record size gives a
  different answer, because the engine counts every layer including the dense
  ones. 17.19 against 17.01 GiB on K3, deliberately, in the direction that
  recommends slightly more. `src/waste.h` had described the field as `top_k`
  records per *MoE* layer, which is the sentence that produces the mismatch for
  anyone who believes it.

- **The fp8 block-scale mapping has a test**
  ([#39](https://github.com/sqliteai/warp/issues/39),
  [#40](https://github.com/sqliteai/warp/pull/40)). The only stub naming it
  replaced it with the identity, so nothing exercised the tile arithmetic
  0.6.8 shipped.

### Fixed

The five gaps of [#36](https://github.com/sqliteai/warp/issues/36) — a real
Kimi-Linear container on Windows, whole-model oracle PASS on AVX2 — and the two
that survived the first attempt:

- **`WASTE_MLOCK` wired nothing on Windows.** `VirtualLock` is bounded by the
  process *minimum* working set; the first fix raised the maximum and preserved
  the minimum, so it wired exactly as much as the bug did: 5928 of 5928 slots
  refused, twice. Measured on the reporter's host rather than reasoned about —
  the maximum went 1.3 MB → 2049 MB and the lock still failed. Raising both
  bounds wires the cache. `SE_INC_WORKING_SET_NAME` turns out not to be
  required.

- **`diskbench` truncated its offset through a 32-bit `off_t`**
  ([#43](https://github.com/sqliteai/warp/pull/43)), at its own default 8 GB
  file size, under LLP64. The loud half wrapped negative and was refused; the
  quiet half wrapped *positive* and read the wrong place successfully, keeping
  the working set inside the first 2 GiB — inside an SSD's SLC cache, which is
  the flattery the tool exists to avoid. Random rows went 0.36 → 1.48 GB/s on
  the reporter's box and stopped inverting with thread count. `docs/GATES.md`
  Gate H and the README's 12.78 GB/s are unaffected: macOS `off_t` is 64-bit.

- **A container's JSON went through Python text mode**, so the same conversion
  produced byte-different containers on Windows and the provenance gate added
  below reported a good fixture as stale. `convert.py` and `requant_vision.py`
  had it too, which means a K3 container converted on Windows was not
  byte-comparable with one converted anywhere else.

- **A missing tool said the engine was broken**
  ([#42](https://github.com/sqliteai/warp/pull/42)). Without `diffutils` the
  suite reported `expert cache changes results` — a bit-identity violation in
  the expert cache — on a clean checkout, because `cmp` was absent. Eight
  failures became eight skips naming the tool, verified here with `cmp` hidden;
  `python3` joins `curl` in guarding the download checks. UCRT64 is the
  documented Windows path and ships neither.

- **`check_budget.sh` said FAIL where it meant "could not measure"**, so the
  budget ceiling was untested on Windows rather than merely misreported. It now
  exits 77 and reads the real number through `PeakWorkingSetSize`.

- **The oracle fixture gated on `trunk` and not on provenance.** A container
  converted with a different `--device` trains different codebooks; the sidecar
  now carries `codebooks_sha256`, which covers every conversion knob at once.
  This is [#7](https://github.com/sqliteai/warp/issues/7) recurring on a second
  axis and it is not Windows-specific.

- **CRLF in `.shards`** made every promise that file's header makes about
  surviving a long haul inert on Windows, first run included; the API listing
  had it too.

Also: `get_small()` dropped API-listed files silently and called the run a
success ([#35](https://github.com/sqliteai/warp/issues/35)); the cpu-list bind
check raced on a wall-clock assumption
([#30](https://github.com/sqliteai/warp/issues/30)); `diskbench` is now built by
CI natively and cross, under `-Werror`, which caught a mingw driver flavour on
its first real run; and the CLI help said `--threads 0` is one per *core* when
it is one per logical CPU ([#44](https://github.com/sqliteai/warp/pull/44)).

### Documentation

`docs/GATES.md` has **no open gate**. Gate 7 closed refuted — see below —
and `docs/LEARNED.md` gains §61 through §64.

`docs/BACKENDS.md` carries a correction it owed: two sentences took one
measured mechanism — synchronous launch and a round trip per call over several
hundred small dependent matvecs — and restated it as a claim about *shape*.
Coherent memory removes the round trip without removing the dependency, so
neither claim follows. The superseded paragraphs are marked and kept.
`docs/ENGINE.md` gains thread placement on a single-CCD part, written as a
bound with the noise floor stated rather than as a result.

### Measured, and not adopted

- **The budget resolver's quantum stands.** Gate 7, run on a 128 GB Strix Halo
  ([#37](https://github.com/sqliteai/warp/issues/37), §63): at 200 tokens
  rather than four, a 3–4 GB cache measures **64.6%–77%** of `floor + 1x`,
  where the kill criterion asked for 90%. §39's window is a four-token window,
  and cross-token reuse is what the extra size buys.

- **Cache-conditional routing stays a knob.** README prices `top-8` against two
  bars — KL 0.037 *and* it reproduces top-16's greedy continuation. Lambda 0.10
  clears the first at 0.0232 and not the second; confirmed here on a second
  machine, where `contiguous` became `consecutive`.

- **The thread default did not move**, though 16 threads on an 8-core SMT part
  measure ~1.6x below the 6–8 plateau. §47 already has the setting inverting
  between models and [#37](https://github.com/sqliteai/warp/issues/37) added
  that it inverts between expert formats on one machine. Three axes is a
  decision, not a table.

- **No budget policy fixes two K3 opens on a 64 GB machine**
  ([#31](https://github.com/sqliteai/warp/issues/31), §64). Gate 1 ran: neither
  process reached decode, and the machine needed a power cycle. Two floors are
  58.38 GB against a 51.54 GB ceiling, so a reservation ledger and a trimmed
  multiplier meet the same wall — they divide the cache *above* the floor, and
  the floor is what busts the budget. The narrower case survives in
  [#49](https://github.com/sqliteai/warp/issues/49).

- **VQ4P is not a throughput upgrade over VQ3R on a GB10**, despite complete
  and exact CUDA coverage, because it read 8.11% more expert bytes (§62). And
  `src/cuda.cu` still does not exist: the build guard stays, now for a
  maintenance reason rather than a performance one.

## 0.6.8 — 2026-08-13

Nothing changes for the two models this project ships numbers for: a
Kimi-Linear forward is byte-identical to 0.6.7, and so is K3's. What changed
is which *other* models the engine can read. The DeepSeek-V3 family — V3, R1
and Kimi K2 — converts and now attends over an ordered sequence, which it did
not before. Alongside that, a host can ask for exclusive ownership of a
container, opt-in and off by default.

**Callers must recompile against this header.** `exclusive_open` was appended
to `waste_cfg`, and `WASTE_E_BUSY` was added to `waste_status`. The library is
still pre-1.0 and does not promise a stable ABI; `serve/engine.py`'s ctypes
mirror moved with the C header.

### Added

- **`convert.py` reads DeepSeek-V3 family checkpoints**
  ([#26](https://github.com/sqliteai/warp/pull/26)). Three things stood between
  a V3/R1/K2 checkpoint and a container, and each failed differently. fp8
  block-scaled weights (`F8_E4M3` with a `_scale_inv` companion) are now
  dequantized by both safetensors readers, with the tile size read from
  `quantization_config.weight_block_size` rather than inferred from the two
  shapes — inferring looks possible and is wrong whenever a dimension is not a
  multiple of the tile, and a compatible-but-wrong size passes the shape check
  while placing every scale on the wrong rows. DeepSeek's MoE tensor names
  (`mlp.experts.E.{gate,up,down}_proj`) are detected from what is on disk and
  normalised to the one spelling the engine knows; without it the expert probe
  missed on every layer and reported `0 MB [missing]` after the download had
  already finished. And the MoE *config* keys are normalised into the manifest
  the same way — `src/model.c` reads `num_experts`, a DeepSeek config only
  spells it `n_routed_experts`, so the finished container was refused at load
  with no diagnostic. `moe_renormalize` is emitted only when true, because
  `model.c` keys it on the field being present rather than on its value.
  Verified end to end on `Kimi-K2-Instruct`: 61 layers, 384 experts top-8,
  VQ3R, 354 GB expert set, 6.9 GB trunk.

- **Opt-in single-process container ownership**
  ([#29](https://github.com/sqliteai/warp/pull/29)). On POSIX hosts,
  `waste_cfg.exclusive_open`, or `--exclusive-open` in the CLI and server,
  takes a non-blocking advisory `flock` on the container directory. Multiple
  contexts in one process share a device/inode-keyed reference; a cooperating
  process that also requests exclusivity receives `WASTE_E_BUSY`. The last
  close and every planning, budget, and partial-load failure release ownership;
  descriptors are close-on-exec, and a forked child discards the copied
  registry. Windows keeps its existing lifecycle behavior.

  This is host policy, not RAM accounting. Container identity is a proxy for
  memory oversubscription and a poor one — two processes on *different*
  containers oversubscribe just as badly and are untouched by this, while two
  small containers on a large machine are refused for nothing. That is why it
  is off by default and why the budget question stayed open as
  [#31](https://github.com/sqliteai/warp/issues/31).

### Fixed

- **MLA applied no rotary at all, so a non-NoPE container attended over an
  unordered sequence** ([#27](https://github.com/sqliteai/warp/pull/27)).
  Every occurrence of `rope` in `src/` was `qk_rope` used as a width;
  `rope_theta`, `rope_scaling` and `mla_use_nope` were read nowhere. That is
  correct for the Kimi models, which set `mla_use_nope` and pass those dims
  through unrotated, and wrong for everything in the DeepSeek-V3 family, where
  those dims are the only positional signal there is.

  It is quiet rather than obvious: lexically determined answers still come out
  right, which is why casual use does not catch it. On Kimi-K2 at VQ3R,
  `"The capital of France is"` still answers `Paris.`; add a second turn
  boundary and the top-1 next token is `<|im_end|>` at p=0.968 — an empty
  assistant turn, because the model cannot tell which turn came first. With the
  rotation it is `Hi` at 0.491. Not a degraded answer, an unordered one.

  `rope_init` follows `DeepseekV3YarnRotaryEmbedding`: YaRN's ramp on
  `inv_freq`, and `mscale_all_dim` squared onto the attention scale (1.8133 on
  K2). The rotation is GPT-J interleaved, and k is rotated *before* it enters
  the latent cache, because a cached entry is reused by every later query and
  carries its own token's position. A rope shape this does not implement is
  refused at load rather than run unrotated. Checked against an oracle whose
  YaRN helpers are `exec`'d verbatim out of the DeepSeek release's
  `modeling_deepseek.py`: 0.000023% relative L2 on this branch against 0.162%
  on 0.6.7, and the mirror image the other way, so the comparison discriminates
  rather than merely agreeing.

  **Nothing moves for Kimi-Linear or K3.** `rope_init` returns before building
  a table when `mla_use_nope` is set and leaves `att_mul` at exactly `1.0f`, so
  those models take the previous path by construction rather than by a runtime
  branch — verified byte-identical on a full forward.

- **Advisory locking fails open when ownership cannot be established.** Only
  actual `EWOULDBLOCK`/`EAGAIN` contention returns `WASTE_E_BUSY`. A directory
  that is search-only, a filesystem without `flock`, or another non-contention
  locking failure continues through the ordinary model-open path. This keeps
  external FUSE, SMB, and NFS containers usable and leaves their real read
  errors to the existing loader diagnostics.

## 0.6.7 — 2026-08-10

The engine decodes exactly as 0.6.6 did — same logits, same container
format — and two things underneath it changed. The second model this project
ships numbers for is now usable the way K3 is: converted with a chat format
it can actually read, and served over HTTP rather than from the command line
only. And the automatic memory budget stopped being able to choose a value
ten times slower than a smaller one.

**`waste_memplan` gained a field, so this is not a drop-in header.** A caller
that only reads the struct is fine; one that allocates or copies it must
recompile. `serve/engine.py` mirrors the new layout.

The chat half came out of one support report — a Kimi-Linear container on a
16 GB MacBook where `waste run` worked, `waste chat` answered oddly, and
`python3 -m serve` printed its banner and exited. The budget half came out
of an experiment that failed: collapsing K3's experts down to one, which
does not work and is recorded below, is what made a small enough working set
to expose the ceiling.

### Added

- **A second prompt format for the server** (`serve/chatfmt.py`,
  `docs/SERVE.md`). At startup the richer format is asked for first: XTML
  when the container's tokenizer carries `<|open|>`, `<|sep|>`, `<|close|>`
  and `<|end_of_msg|>` as single tokens, and otherwise the container's own
  `chat.json` — the same four prefix/suffix strings `waste chat` has always
  read. A container is now addressed identically over HTTP and on the
  command line, and a hand-edited `chat.json` is honoured by both.

  Plain means plain: system / user / assistant turns, blocking and
  streaming, with the stop token taken from the template's assistant suffix
  rather than guessed. Everything four strings cannot express is refused
  with a 400 naming the field — `tools`, `reasoning_effort`, an image part,
  a tool-result turn. None of it is dropped silently, on the same reasoning
  the effort mapping already followed: a server that ignores
  `reasoning_effort` reports a different amount of reasoning than it did.

  `chat.json` is validated harder here than by the CLI's reader, which has a
  person watching and an interrupt key. Serving requires an `open`, a `user`
  turn, and an assistant suffix carrying a control token — without the last
  one every reply runs to `max_tokens` and reports `finish_reason: "length"`,
  which reads as a broken model rather than a broken template. And every
  `<|…|>` in the file is resolved against the real vocabulary before the
  format is used at all, because markup the tokenizer does not have encodes
  as ordinary text and the model then reads its own turn structure as prose
  and answers anyway, plausibly and wrongly. The rendering keeps the split
  that makes the XTML path safe: the template's strings go out as markup
  segments, the caller's content never does.

  Measured on a Kimi-Linear container: a multi-turn conversation
  round-trips, streaming deltas arrive, and both refusals come back as 400s
  naming `tools` and `reasoning_effort`. The serve suite is 211 checks, up
  from 174.

- **`tests/sweep.c` takes a `topk=` arm**, lowering only, since the scratch
  is sized at load from the manifest's `top_k`. One load with the arms
  interleaved, which is what made §56's curve trustworthy after two earlier
  attempts were spoiled by comparing arms across process lifetimes — the
  failure `sweep.c` exists to prevent and that §32 and §33 already record.

- **`tools/convert.py` installs a `chat.json` per architecture**, with
  `examples/chat-kimi-linear.json` alongside K3's. The architecture was
  already recognised; it simply had nothing to install and said so, which
  left every Kimi-Linear container to be finished by hand — and the obvious
  hand fix, copying the ChatML `examples/chat.json`, is wrong in a way
  nothing reports: `<|im_start|>` is not in Kimi's vocabulary and encodes as
  six ordinary tokens.

  So the converter also **refuses to install a template whose markup the
  release's tokenizer does not carry**, naming the missing markers. Only
  when there is a specials list to check against: a release without
  `tokenizer_config.json` is no evidence, and refusing on none would be
  worse than the unconditional copy it replaces.
  `tests/test_convert_chat.py` covers both claims per architecture.

### Fixed

- **The automatic budget could choose a value 10x slower than a smaller one**
  (`src/waste.c`, [LEARNED.md](docs/LEARNED.md) §57). It stepped down a whole
  working set at a time until the total fit under **7/8** of usable RAM. 7/8
  of 64 GB is 56, inside the 46-52 GB band §39 measured as an eightfold
  collapse: the engine stays inside its budget, the machine does not, and a
  cache hit becomes a page fault.

  It had stayed harmless by luck. K3 at top-16 asks for 80.77 GB, cannot have
  it, and the step-down lands on `floor + 1x` = 46.39 GB — the measured
  optimum, reached for the wrong reason. Lower `num_experts_per_token` to 8
  and a token's working set halves, three multiples fit under the old
  ceiling, and the default took 54.77 GB and ran at **0.08 tok/s against 0.77
  at 46 GB**, with a *higher* hit rate and a *lower* RSS — which is what
  paging looks like from inside the process.

  The ceiling is now **3/4**, measured rather than assumed: 46 GB is the
  largest budget on this machine known to be on the good side and 52 the
  smallest known to be on the bad one. K3 at top-16 still resolves to
  46.39 GB, Kimi-Linear is untouched, a 128 GB machine still gets the full
  `floor + 3x`, and K3 at top-8 now resolves to 46.18 GB and **0.88 tok/s**
  on the path a user gets by typing nothing.

- **The server refused to start on any container without XTML markers**
  ([#34](https://github.com/sqliteai/waste/issues/34)). `ChatServer.__init__`
  resolved the four control tokens and let the `EngineError` out, so the
  process died before binding a port — taking `/health`, `/v1/models` and
  `/v1/completions` with it, none of which need a chat format at all, and
  reporting only that `<|open|>` had come out as five tokens. The reason is
  now held rather than raised, and a container with neither format serves
  everything except `/v1/chat/completions`, which returns 400 with
  `code: "unsupported_chat_format"` and both reasons — no XTML markers, and
  what was wrong with the `chat.json`.

### Changed

- **`waste_memplan` reports `working_set_bytes`** — one token's expert
  traffic, `top_k` records per MoE layer. Callers recovered it as
  `(recommended_bytes - floor_bytes) / 3`, which stopped being true here:
  `recommended_bytes` is now capped at the container's whole expert set,
  because a cache holding every expert cannot be improved by growing. On an
  ordinary container nothing moves — K3's bank is 952 GB against a 3x working
  set of 52 GB — but the two are no longer three times apart in general, so
  the quantity the rule is built on is reported instead of re-derived.
  `waste plan --json` carries it.

- **`README.md` figures re-measured on this commit**: K3's floor 29.06 →
  29.19 GB, its default budget 46.25 → 46.39 GB, Kimi-Linear's floor 1.28 →
  1.32 GB and its decode 10.65 → 10.62 tok/s. Drift from earlier commits,
  not from any change here; the K3 decode range is left as it was, because
  its low end was measured under conditions this pass did not reproduce.

### Measured and not adopted

- **Merging a layer's experts into one, or into sixteen, is not a
  compression of a MoE — it is a deletion of it** (§53-§55). Built because it
  was asked for: 982 GB becomes 30 GB, decode goes 0.60 to 1.58 tok/s, and
  the model emits `<|close|>` forever. No weighting helps, and the reason is
  geometric — distinct experts are mutually **orthogonal** (cos 0.0006), so
  their average has `1/sqrt(E)` of their norm and is 99.8% orthogonal to
  every one of them. A gain sweep confirms it from the other side: scaling
  the merged expert to *zero* is better than using it. Clustering into 16
  does not rescue it, and pruning to the 16 busiest — which beats every
  merge — still answers that the capital of Italy is Paris. The tooling
  stays on the `k3-mini` branch rather than main.

- **Fewer experts per token is worth taking; the default still does not
  take it** (§56). `num_experts_per_token` 8 instead of 16 is **1.49x** at
  KL 0.037, with top-16's greedy continuation reproduced; top-4 is 1.78x,
  keeps the right argmax, and stops following the prompt within a few
  tokens. It is a quality trade, so it is documented in `README.md` and left
  to the caller rather than changed under anyone.

- **§4D re-priced: batching is worth less in this regime, not more** (§58).
  `docs/EFFICIENCY.md` §1's 1.62x reproduces exactly at top-16 (**1.60x**)
  and falls to **1.28x** at top-8 — batching takes its gain from the I/O and
  truncating `top_k` has already taken half of it, so the two overlap rather
  than compose. The ceiling holds for batching across independent streams
  too, which §4D never separated from grouping within one: `vq_apply` costs
  one pass per (token, expert) pair however they are grouped, and that is
  64.2% of a step, so no scheme beats **1.56x**.

- **The trunk's contextual sparsity is real and unusable** (§59, §60). A
  quarter of the shared expert's intermediate channels carry 99% of the
  layer's output — but the trunk is **49.9% attention** against 16.5% FFN,
  so perfect sparsity where the technique fits is worth **1.14x**. And the
  channel identity is near-random across tokens: Jaccard 0.196-0.271 against
  a 0.143 chance baseline, 2-4% of the set common to eight consecutive
  tokens, 70-83% of all channels appearing in at least one of them. No
  static core to prune, no cheap prediction to make. Taken together these
  put the honest ceiling from 0.88 tok/s at about **2x**, and name what the
  rest would cost: 6.1x of bandwidth efficiency, which is a rewrite of the
  forward pass, and 11.2x of bytes per token, for which no mechanism was
  found.

- **Tool calls over `chat.json` are not built**, and the reason is not
  effort. Four prefix/suffix strings cannot carry a tool declaration, an
  argument list, or a result turn — K3's encoder needs 647 lines for it.
  Kimi-Linear's tokenizer does carry `<|tool_call_begin|>` and friends, so
  it is reachable in principle, but the markup is not transcribed anywhere
  in this repo and cannot be derived from the release on disk: that copy
  ships no `chat_template` and no reference encoder. #34 holds this half.

- **The `chat.json` renderer is transcribed and tested, not differentially
  verified.** `serve/xtml.py` earns its confidence from a segment-for-segment
  differential against the release's own `encoding_k3.py` under `K3_DIR`,
  and there is no equivalent program for Kimi-Linear to check against. If an
  Instruct release ships a `chat_template`, HF's Jinja renderer would be a
  real oracle and a `tools/`-side check like the existing ones; until then
  the weaker claim is the honest one.

## 0.6.6 — 2026-08-05

The engine decodes exactly as 0.6.5 did and no container format moved. Two
things make it a tag. A host can now say which CPUs the compute pool runs
on, which is the first answer this project has to a machine whose cores are
not interchangeable. And `tools/diskbench` — the tool whose entire job is to
certify the storage a container will be streamed from — was measuring the
page cache on Linux, so it had been answering that question wrong for
everyone who is not on macOS.

**Callers must recompile against this header.** `cpu_list` was added to
`waste_cfg` between `n_threads` and `cache_policy`, so a caller built
against 0.6.5's struct reads `cache_policy` and every field after it from
the wrong offset. `serve/engine.py`'s mirror moved with it; an out-of-tree
ctypes or FFI binding has to move too. The library is 0.6.x and promises no
stable ABI yet, but a silent misread is worth the sentence.

### Added

- **`--cpus LIST`** on the CLI and the server, `waste_cfg.cpu_list` in the
  API, `WASTE_CPUS` in the environment — a Linux-style cpu list (`0-5`,
  `0-2,6-8`) that the compute pool binds to. Linux and Windows; macOS has
  no call that binds a thread to a core, so a list there is refused with
  `WASTE_E_UNSUPPORTED` rather than ignored.

  It exists because on a machine whose cores are not interchangeable,
  placement is worth more than the thread count.
  [Issue #23](https://github.com/sqliteai/waste/issues/23) measured
  Kimi-Linear-48B on a Ryzen 9 9900X — two 6-core CCDs, separate 32 MB L3 —
  and found six threads on one CCD 16-25% faster than the same six split
  across both, at identical `bytes_read` and identical hit counts. Handing
  those six threads all 24 CPUs to migrate between costs a further ~10%.
  Not reproduced here: this repo has no multi-CCD machine, and the numbers
  above are the reporter's. What *is* checked here is the mechanism —
  `tests/test_cpus.c` reads back every participant's affinity mask.

  **No default changed.** The engine still names no CPUs and leaves
  placement to the OS. `docs/LEARNED.md` §47 measured the tempting default
  — cap the pool at the fast cores — as a 25% gain on Kimi-Linear and a 34%
  loss on K3, so it stays a switch. `--threads 0` with a cpu list means one
  thread per CPU listed; an explicit `--threads` still wins. The thread
  that calls into the engine is bound too, on its first parallel region,
  because it is one of the workers; the expert cache's reader threads are
  not, because they are blocked in `pread` rather than competing for a
  core. `docs/ENGINE.md`, "Thread placement", has the rest.

- **The converted K3 container over BitTorrent**, in `README.md` ahead of
  the conversion recipe. The default conversion is deterministic, so the
  982 GB directory is byte-identical for everyone who produces it, and
  nearly all of what the recipe costs — a 1.42 TB source download, 4.7
  hours, and staging storage that has to exist before it can be freed — is
  paid to reproduce a fixed artifact. The torrent's own piece hashes verify
  it as it arrives. Converting from the published weights stays documented,
  for anyone who would rather not trust a third-party copy.

### Fixed

- **`tools/diskbench` measured the page cache on Linux, not the disk**
  ([#22](https://github.com/sqliteai/waste/pull/22), `docs/LEARNED.md`
  §49). It documented itself as reading with the cache bypassed, and did
  neither: `nocache()` had an `#ifdef __APPLE__` body and nothing else in
  it, and `O_DIRECT` appeared nowhere in the file. Against a Samsung 970
  PRO on Gen3 x4 it reported 44.67 GB/s sequential and 65.72 GB/s random
  over a 3.94 GB/s link — 11x and 17x the ceiling. Bypassed: 3.15 and 3.33
  GB/s, saturating at two threads, which is what that drive should do.

  This is LEARNED §14 in the one place §14 did not reach. The engine's own
  bypass was written blind and fixed on 2026-07-28; the tool that exists to
  characterise the engine's I/O kept reading RAM, which means §46's standing
  rule — run `diskbench` and divide before claiming anything is disk-bound —
  returned a fiction on Linux for that whole window. **No published number
  moves**: every `diskbench` figure in `docs/GATES.md`, `docs/EFFICIENCY.md`
  and LEARNED §44/§46 was measured on macOS, where `F_NOCACHE` did work.

  The flag alone is not enough, for the reason `bank_open` already knows:
  `O_DIRECT` is accepted at open and refused at transfer (tmpfs does this),
  so a bare flag turns a refusing filesystem into a table of zeroes with no
  cause given. It follows `bank_open` instead — probe with one aligned
  transfer, fall back to a plain open plus `POSIX_FADV_RANDOM`, and label
  every row, because a bench that quietly measures something else is worse
  than one that says it could not. The write is bypassed too, and that is
  not symmetry for its own sake: `F_NOCACHE` stops new pages being cached
  but does not evict resident ones, so a buffered write leaves the file in
  the UBC and every read row below it reports RAM — 8.07 GB/s sequential
  with the write bypassed against 26.04 GB/s with it buffered, 1 GB file on
  an M5 Pro. Also fixed alongside: a sub-page record rounded to zero and was
  divided by, and a failed sequential read ended the loop and silently
  shortened the row.

  Reported, diagnosed and fixed by fab2s. Verified here on macOS as
  unchanged within noise; the Linux figures are the reporter's, on hardware
  this repo does not have.

- **`diskbench`'s tok/s column answered for K3 whatever was being sized.**
  The derived column carried 12.5 GB/token in its format string — K3's
  figure — so on a 48B model at a measured 1.61 GB/token it was ~8x off,
  and silent about the assumption, which is what made it a trap rather than
  an approximation. It is now the fifth positional argument with no default:
  without it the column is not printed. A tool cannot derive bytes-per-token
  from a scratch file — that number belongs to a container, and `waste
  bench` already reports it. `docs/GATES.md`'s Gate H table keeps its
  "tok/s @12.5 GB/token" header: that was a K3 decision, and the figure is
  stated in the header rather than hidden in a format string.

## 0.6.5 — 2026-08-04

Nothing in the engine changed: a binary built from this tag decodes exactly
as 0.6.4 did, and no container format moved. What changed is on either side
of it — converting K3 on the disk you actually have, and how long a reply
the server gives a client that never asked for a length.

### Added

- **`tools/convert.py --reclaim {off,dry,on}`** — deletes each source shard
  once its last consumer has published, so peak staging is the container
  plus the shards still owed rather than the container plus all of them. On
  K3 that is the difference between 1.42 TB of staging beside a 982 GiB
  container — two disks — and one. It is safe because every tensor has
  exactly one consumer, so a shard whose last consumer has finished is never
  opened again. `--reclaim` also runs the trunk pass first: it consumes
  every non-expert tensor, and while it ran last almost no shard was ever
  spent. The reordering is neutral — `off`, `dry` and `on` produce
  byte-identical containers.

  **Off by default, and not reversible.** A reclaimed shard has to be
  downloaded again, and `verify_container.py` loses the comparison against
  source for good — which is why `pipeline.sh` now converts one probe layer
  and round-trips *that* while the checkpoint is still whole, before
  converting the rest. Stages renumber to six; stage 4 passes `--skip-trunk`
  because stage 2 built it, a K3 trunk being hours to do twice.

  It refuses **before** deleting rather than during: `--experts`, a
  container inside the checkpoint or the reverse, a shard that is neither on
  disk nor already reclaimed, and a bank that is not a whole bank —
  `bank_is_sound` walks the records by their own block counts for 48 bytes
  each, which catches a bank truncated by a kill, a full disk, or a torn
  rename. Releases are recorded in `<src>/.reclaimed`, fsynced ahead of the
  unlink, because a name recorded but not deleted costs nothing while the
  reverse is indistinguishable from an unfinished download.
  [K3.md](docs/K3.md) has the refusals and the ledger discipline.

  Proven on a copy of a real Kimi-Linear checkpoint rather than on stubs:
  `--layers 1,2 --reclaim on` deleted exactly the one shard the dry pass
  named (4.7 GiB, 92 -> 87 GB), left the other 19 unchanged, and wrote a
  container that reads back 256 records with 0 problems. The second run over
  the now-incomplete source refuses — pointing at `--skip-trunk` — and
  deletes nothing while refusing.

### Changed

- **The server's default `--max-tokens` is 4096, was 512.** Clients mostly
  do not send the field; Open-WebUI does not unless you set it in the
  model's advanced parameters, so the server default was every reply's
  length, and a reply that ends at the cap is indistinguishable from a model
  that stopped on its own. `--ctx` never lifted it and could not: the limit
  is clamped to the room left after the prompt, so raising the context only
  ever lowers the cap. **This is a behaviour change, not a fix** — a
  deployment that relied on 512 to bound per-request cost should now pass
  `--max-tokens` explicitly. `Engine.generate`'s own default is untouched:
  the server always passes the value, so the two never meet.

- **[SERVE.md](docs/SERVE.md) documents Open-WebUI** — the base URL, why no
  compatibility mode is needed, `--host 0.0.0.0` for a client in a
  container, the background title and tag requests that queue behind the
  reply on the lock every generation takes, and `reasoning_content`, which a
  client that does not know the field renders as a server that has stopped.

### Fixed

- **A resumed `--reclaim` run believed the download ledger over the disk.**
  `ST.have()` reads `fetch_weights.sh`'s `.download-state`, so a shard that
  the run had itself consumed still read as present and both refusals were
  skipped. Absence is now asked of the filesystem, and `have()` only asked
  whether a shard that is there finished downloading. The pipeline test
  found it; the file-backed test stub could not have, and now mirrors
  `mxfp4.ST`, trap included.

## 0.6.4 — 2026-08-04

A container format addition and the measurements that price it. **Nothing
changes by default and this is not a speedup release**: no shipped container
uses the new format, the new switches are off, and an engine built from this
tag behaves exactly as 0.6.3 did unless it is asked otherwise. The reason to
tag it is that `fmt 8` is now allocated and public, and a format code that
lives only in a working tree is one somebody else reassigns.

### Added

- **`WQ_VQ4P`, fmt 8** — 4 residual VQ stages of 64 entries with 6-bit
  indices packed four into three bytes. Same 3.00 bits/weight as VQ3R, same
  record size, same blocked index layout; what changes is that a 64-entry
  stage table is 64 bytes, which is one NEON `vqtbl4q`, where VQ3R's
  256-entry table is sixteen vector registers on a machine that has
  thirty-two and cannot be held at all. That is why the VQ3R gather is
  scalar. [FORMAT.md](docs/FORMAT.md) specifies the packing;
  [LEARNED.md](docs/LEARNED.md) §41 has the derivation.

  It is a distinct fmt rather than a manifest flag because a VQ4P payload is
  byte-for-byte the size of a VQ3R one, so a reader taking the three bytes
  for three one-byte indices would decode silently and wrongly. The engine
  additionally refuses a record whose fmt byte disagrees with the manifest's
  `index_bits`, which is the only read-path behaviour that changed.

- **`tools/convert.py`: `--entries`, `--index-bits`, `--stages 4|6`.**
  Default output is unchanged VQ3R. The parameters travel in the job tuple
  rather than a module global: the worker pool uses `spawn`, so a global
  would have written every layer at 256 entries without saying so.

- **`WASTE_XPAR`** — one task per routed expert instead of one per row
  range, **off by default**. `WASTE_XPAR_BATCH` bounds how many experts are
  held at once, `WASTE_P6_CHUNK` sizes the VQ4P apply, and
  `-DWASTE_P6_SCALAR` builds the kernel's portable path, which is
  bit-identical to the NEON one rather than merely close — §43 explains why
  an int8 lookup table raises that bar.

- **`tools/lutbw.c`, `tools/lutmt.c`** — the two benches the sections below
  rest on: kernel throughput against a working set from 4 MB to 1 GB, and
  the same kernels driven through the engine's own `waste_parallel_for`.

### Measured

Numbers on this commit, medians of repeated runs, each container on the same
storage as the baseline it is compared against.

- **The kernel is 3.88x** and that survives a gigabyte of index stream, so it
  is not a cache artefact (§46).
- **In the engine it is 1.24x** on Kimi-Linear best-configuration against
  best-configuration, and **1.74x** against what the engine does untouched;
  **1.09x** on K3 (§46, §47).
- **Quality costs +2.7% perplexity** on Kimi-Linear (10.937 -> 11.237), all
  of it from the smaller codebook. Quantizing the runtime lookup table to
  int8 — which is what makes a byte shuffle possible at all — measured free,
  because the scale is per 32 vector positions rather than global (§41).

The gap between 3.88x on a bench and 1.17x in place is the useful part, and
§47 is the answer: 3.88x is a single-thread ratio, this machine is 6
performance cores and 12 efficiency cores with the pool taking all 18, and
the fast kernel is the one an E-core straggler hurts. **On K3 the same
settings invert** — six threads are 34% worse than eighteen there. No
default was changed because every one of them is right on one model and
wrong on the other.

### Not adopted

- **6x16 codebooks** (FAISS FastScan's shape) are a third faster than 4x64
  and cost 18% more reconstruction error against 4x64's 7.5%. Speed was not
  the binding constraint (§41).
- **Expert-parallel MoE as a default.** Worth 1.24x on Kimi-Linear, a
  regression on K3: the batch that gives it parallelism is the batch that
  barriers the read-ahead, and no batch size wins both (§44).
- **Capping the thread pool at the performance cores.** A 25% win on
  Kimi-Linear and a 34% loss on K3 (§47).

## 0.6.3 — 2026-08-03

### Fixed

- **The automatic budget sized against the host's RAM inside a container**
  ([#14](https://github.com/sqliteai/waste/issues/14)). `waste_physical_ram()`
  is `sysconf(_SC_PHYS_PAGES)` on Linux, which reports the host's `MemTotal`
  from inside a cgroup that is allowed a fraction of it, so a `--budget`-less
  open resolved `floor + 3x` against memory the kernel would never hand over:
  K3 in a 32 GiB cgroup on a large host asks for ~80 GB and is killed. Unlike
  the paging cliff of [LEARNED.md](docs/LEARNED.md) §16 this has no gradual
  form and no cache policy softens it. The ceiling is now
  `min(physical, cgroup limit)` — the smallest finite `memory.max` or
  `memory.high` across the cgroup and its ancestors, since the limit is
  hierarchical — and the rest of the resolver is unchanged. See §40.

  Current pressure (`MemAvailable`, `memory.current`) was considered and
  deliberately left out: a budget is resolved once and held for a whole run,
  so bounding it by an instantaneous sample would make the same command on
  the same machine two different runs. Whether it should trim the working-set
  multiplier instead is #14, still open.

- **`tools/convert.py` spawned `--jobs` × cores threads**
  ([#13](https://github.com/sqliteai/waste/pull/13), contributed by
  @andrewwhitecdw). torch sizes its intra-op pool from `os.cpu_count()`, and
  the native VQ encoder reads `nthreads=0` as "every core" (capped at 64), so
  N worker processes meant N×cpus threads competing for the machine: on a
  224-core box `--jobs 8` spawned ~1792 threads and the codebook phase ran
  ~20x slower than the measured baseline. Each worker is now capped at its
  fair share, `cpu_count // jobs`, for both pools — set in the parent before
  the workers spawn, since that is the only point torch reads it, and with
  `setdefault`, so an explicit `OMP_NUM_THREADS` stays the caller's.

- **The trace simulator modelled a different cache than the engine.**
  `tools/routing_stats.py simulate` kept a frequency count across evictions
  that `ec_claim` resets and sampled 32 victims where `EC_SAMPLE` is 16:
  against the same trace it read **36.6% where the engine measured 30.4%** —
  optimistic, plausible and wrong. Both constants now come from `ecache.c`,
  which brings it within 1.5 points across a 0–30% range, and `tests/run.sh`
  asserts the agreement rather than remembering it. Two smaller things went
  with it: the route dump writes the absolute position of the token each row
  belongs to, so readers stop re-deriving token boundaries from where the
  layer index wraps (a heuristic that is simply wrong on the chunked path,
  where rows group by layer), and `simulate --data` takes a container, whose
  manifest states the record size the engine actually `pread`s. See §37,
  *The simulator was modelling a different cache*.

### Added

- `waste_usable_ram()`: physical RAM, or a smaller cgroup-v2 limit when one
  applies — what a budget of 0 sizes against, and what an embedding host
  should size its own ceiling from. `waste plan --json` reports it beside
  `physical_ram_bytes`, which stays what it always was.

- **`tests/sweep.c`, a one-process measurement harness.** It loads a
  container once and runs the arms back to back, interleaved, resetting the
  session and clearing the expert cache between each — a warm cache would
  hand the second arm the first one's work and measure the order instead of
  the setting. Kimi-Linear, two arms, three repeats: spreads of 2.6% and
  0.5%, where nine paired runs of the same comparison across processes
  spanned 0.79x to 1.79x. The variance was the harness, not the feature. On
  K3 the deterministic columns come out exact and the clock still drifts,
  which is the machine's memory system rather than the process — what the
  harness buys there is that the noise is visible as noise. §38.

- **`docs/TECHNICAL.md` and `examples/`.** The measurement tables move out of
  `README.md` into TECHNICAL.md, and `examples/` carries three compilable
  programs against the public header — `api_plan.c` (budget arithmetic
  without loading), `api_text.c`, `api_vision.c` — with a README that walks
  through them.

### Changed

- **§4's cache floor still reproduces exactly, and has stopped binding.**
  The oldest load-bearing measurement here — below one token's working set
  the hit rate is zero, not low — was re-measured across four cache sizes in
  one process, two repeats, hit rate and bytes read identical to the digit
  across both:

  | budget | expert cache | slots | hit | decode |
  |---|---|---|---|---|
  | 32 GB | 3.32 GB | 287 | 29.1% | 0.56–0.58 tok/s |
  | 46 GB | 17.32 GB | 1498 | 36.2% | **0.63 tok/s** |
  | 52 GB | 23.32 GB | 2018 | 38.4% | 0.07–0.09 tok/s |
  | 58 GB | 29.32 GB | 2537 | 41.3% | 0.07–0.08 tok/s |

  The 29.1% at 287 slots is not a refutation of §4: with the lookahead off
  the same 287 slots give **0.0%**, exactly §4's zero. What breaks it is that
  a speculative record has to survive one attention rather than one token, so
  a cache far too small for a token's working set is ample to hold six
  experts. A 3.32 GB cache is now within 10% of a 17.32 GB one, which means
  the premise the default resolver is built on — that cache is only worth
  buying in whole multiples of a working set — no longer holds. **The
  resolver is unchanged**: that is a decision, not a measurement, and it is
  [GATES.md](docs/GATES.md) Gate 7, open. The cliff is exactly where it was,
  and the last two rows say what it is — throughput falls eightfold while the
  hit rate rises and the bytes read fall, because the engine is inside its
  budget and the machine is not. §39.

- **0.6.2's "total bytes read unchanged" for the router lookahead was the
  harness.** Measured with both arms starting from an identically cleared
  cache, the byte economics depend on the cache size: **6.6% fewer** bytes at
  1498 slots, **8% more** at 287, where speculative records are evicted
  before use often enough to be re-read. It is a prefetch at small caches and
  a scheduling change at large ones, and 0.6.2 measured only the large end.
  The feature and its default are unchanged. §38, §39.

## 0.6.2 — 2026-08-01

### Fixed

- **`waste info` and `waste run` crashed on K3 on every x86 build**
  ([#10](https://github.com/sqliteai/waste/issues/10)). The tensors the
  loader skips — the vision tower, and anything outside `tensor_prefix` —
  kept `group` at 0, and the row-scratch sizing divided by it. The
  architecture decided what that meant: arm64's `sdiv` answers 0 and the
  run continues, x86's `idiv` raises `#DE`. `waste plan` was unaffected
  because it does not load. §37.
- **`WASTE_Q8=0` could not load a 4-bit trunk**
  ([#6](https://github.com/sqliteai/waste/issues/6)) — that is, any
  container a default `tools/convert.py` run produces. The dequantizer
  read one byte per weight, true of Q8G alone, while catching every
  quantized format. It now decodes through `waste_deq_row`, the one place
  that knows all three widths. The same lines also predated `waste_f16`'s
  subnormal fix and flushed group scales below 6.1e-05 to zero.
- **`embed_tokens` stays on disk under `WASTE_Q8=0`**, as it does
  otherwise: 7.93 → 6.52 GiB of peak RSS on Kimi-Linear, identical logits.
  The f32-equivalence check now differs from the default path in the
  storage width alone, which is what it claims to compare.

### Added

- **Router lookahead in the decode path.** At the end of a MoE layer, once
  its reads are consumed and the disk is about to idle through the next
  layer's attention, layer L+1's router runs on layer L's hidden state and
  issues speculative reads for its top 6. Demand hit rate 14–19% → 38–40%
  with total bytes read unchanged (254.2 → 254.5 GB): the records were
  going to be read anyway, and only *when* changes. Nine paired runs,
  median 1.17x. `WASTE_LOOKAHEAD=0` disables. §34, §35.
- **`WASTE_MLOCK`** wires the trunk and the expert cache; `WASTE_MLOCK=cache`
  wires the cache alone. Off by default — Linux's `RLIMIT_MEMLOCK` is
  commonly 8 MB. Wiring the trunk is worth 3x in the transition zone around
  52 GiB and nothing below it; it does not move the knee. §30, §31, §32.

### Changed

- **`tests/run.sh` generates its own PyTorch oracle** from the container
  under test (16.9 s) instead of diffing against a shipped fixture, which
  can only ever be valid for the container that produced it: expert
  codebooks are k-means, and the same seed on a different `--device`
  trains different books — one layer of 26 moves the logits by 1.24
  against a 1e-3 threshold. The fixture remains as the fallback where `uv`
  is absent, with its provenance recorded beside it.
  ([#7](https://github.com/sqliteai/waste/issues/7)), §33.
- **`tools/make_test_container.py` emits what a real conversion does**: a
  `Q4G/Q8G/F32` trunk rather than Q8G throughout, and `--prefix` for a
  container whose tensors are not all under its `tensor_prefix`. Both are
  shapes the suite could not previously reach, and both had a live bug
  behind them.
- Checks that cannot run now say why instead of reporting a refusal as a
  divergence: `WASTE_Q8=0` on K3 wants 211 GB of f32 trunk on a 64 GB
  machine, the oracle prompt is Kimi-Linear's, and a cold hotlist run that
  already missed nothing demonstrates neither outcome
  ([#5](https://github.com/sqliteai/waste/pull/5)).

### Measured and not adopted

- **Cross-layer prefetch from `next_layer_top`** — gated before building
  and refused: 29.0% recall against a 60% break-even. §29, revisited and
  superseded by the lookahead above in §34.
- **The lookahead in the prefill path.** Built, measured, removed: a chunk
  layer's disk is busy continuously, so a prefetch there does not move a
  read into idle time, it moves it in front of another read and pays an
  eviction for it — 7% more bytes. Decode keeps it. §36.

## 0.6.1 and earlier

Not covered here; see `docs/LEARNED.md`, which is dated and append-only,
and the git history.
