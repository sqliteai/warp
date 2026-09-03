# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WARP is an embeddable MoE inference engine in C11 (no third-party runtime
deps) that keeps a model's dense trunk resident and streams routed experts
from disk, using the remaining RAM as a bounded expert cache. Its proof
point is Kimi K3 (2.78T params, 982 GiB container) on a 64 GB laptop.

Three things follow from that and shape everything else:

- **Disk I/O is the budget, not arithmetic.** ~53% of a K3 decode step is
  expert reads. Optimizations are judged on bytes read per token and on
  cache hit rate, not on FLOPs.
- **RAM is a hard ceiling, not a hint.** `waste_cfg.ram_budget_bytes`
  bounds *everything* the engine allocates. Exceeding it means the OS
  pages, and a paged "cache hit" is slower than the disk read it replaced.
- **Correctness is measured against an oracle**, not asserted. Every layer
  is diffed against a PyTorch reference (`tools/kimi_ref.py`,
  `tools/vision_ref.py`, `tools/k3parts_ref.py`, `tools/kda_ref.py`).

## Commands

```bash
make                     # libwaste.a, waste CLI, libwaste.$(SOEXT), libwastevq
make test                # build the test binaries (separate target from `make`)
make check               # tests/run.sh — the whole C-side suite (rebuilds first)
make serve-check         # the Python server suite (needs libwaste.so/.dylib)
make asan                # full rebuild under ASan/UBSan + the suite, then clean
make fuzz FUZZ_RUNS=200  # container parser fuzzer
make fuzz-asan           # what CI runs: fuzzer against an instrumented build
make WASTE_ENABLE_METAL=1   # accelerators are build-time options
make CC=x86_64-w64-mingw32-gcc-posix   # Windows cross-build (arch from -dumpmachine)
```

`make` and `make test` are separate on purpose — a stale test binary is one
of the two failure modes `tests/run.sh` was written to catch, so it always
rebuilds both before running anything.

### Running the suite, and single checks

```bash
tests/run.sh                          # $HOME/models/kimi-linear.waste by default
tests/run.sh /path/to/model.waste     # a real container unlocks the skipped checks
tests/run.sh /nonexistent             # forces the synthetic container (what CI does)
```

With no container it builds a few-MB synthetic one via
`tools/make_test_container.py` and reports SKIP — loudly — for anything
needing real weights. A fresh clone on macOS is 30 pass / 13 skip; with
both K3 and Kimi-Linear containers on disk it is 45 checks, 43 of which
pass. Linux skips more, having no `uv` in `Dockerfile.test`: 26 / 16.

The download-script checks start `tests/range_server.py` on an ephemeral
port and read the number back through `--port-file`. Keep it that way — a
hardcoded port fails on a machine already using it, and it fails as
"resume" rather than as "that port is taken".

Env it reads: `WASTE_REF_MODEL` (container — **point it at a default
`convert.py` conversion**, i.e. a 4-bit trunk *and* VQ3R experts; a
`--trunk8` container is a shape nobody ships, and running the suite on one
is how the Q4G load path stayed broken through green runs — the same
applies to `--index-bits 6`, which exercises a different kernel and a
different record fmt), `WASTE_REF_SRC` (source safetensors, for
the round-trip), `WASTE_ORACLE` (logits from `tools/kimi_ref.py` — **must be
the same token ids run.sh uses**, or a mismatched dump looks exactly like an
engine bug; setting it also turns off both generating an oracle from the
container and the provenance check on the shipped fixture, so it is the one
way to compare against weights that are not the ones under test), `K3_DIR`
(the K3 release directory, for the XTML differential), `K2_DIR`
(a Kimi-K2 release directory — only its `chat_template.jinja` is
read, and it is the grammar for the tool markers Kimi-Linear's
tokenizer carries and its own release never spells out).

**On a MoE, never compare two paths by the distance between their logits
alone.** A top-K router turns an arbitrarily small arithmetic difference
into a discrete one, and past the first flipped expert the two paths are
running different weights — the distance then measures how much the model
cares which of two indistinguishable experts it used, not how far the
arithmetic moved. On K3 one flipped tie at token 12 is worth max-abs 0.286
against a 1e-3 threshold, with every logit before it agreeing to 1e-6, and
the tie's own margin is 7.3e-07 — the tightest of all 1472 decisions in that
prefill. The suite therefore compares logits *and*, when they part, asks
`tests/route_diff.py` whether the first routing decision that differed was
one the reference itself could resolve. LEARNED §71 has the distribution
that sets `--eps`; the two K3 checks that failed through 0.7.0 failed on
this and not on a kernel.

Individual checkers, after `make test` (all binaries land at the repo root):

```bash
python3 tools/make_test_container.py /tmp/tiny.waste     # anything below needs a container
./test_forward /tmp/tiny.waste 3,7,11,5 out.bin 0        # forward pass; 0 = no generation steps
WASTE_CHUNK=1 ./test_forward ...                         # chunked prefill instead of sequential
./test_container /tmp/tiny.waste/experts-L0.bin 2        # ONE bank + expected record count
./test_image /tmp && ./test_state MODEL && ./test_tokenizer MODEL "text"
./test_k3parts out.bin && uv run --with torch python tools/k3parts_ref.py out.bin

# why two paths disagree: identical routes, a tie, or a real divergence
WASTE_DUMP_ROUTE=a.route WASTE_DUMP_SCORES=a.scores ./test_forward M IDS a.bin 0
WASTE_BACKEND=cpu WASTE_DUMP_ROUTE=b.route ./test_forward M IDS b.bin 0
tests/route_diff.py --ref a.route --other b.route --scores a.scores

python3 -m unittest discover -s tests/serve -t . -p "test_*.py"   # all serve tests
python3 -m unittest tests.serve.test_regions -t .                 # one module
K3_DIR=... python3 -m unittest tests.serve.test_xtml.TestAgainstUpstream -t .
K2_DIR=... uv run --with jinja2 python -m unittest tests.serve.test_chatfmt_upstream
```

Python reference checks run through `uv run --with torch --no-project` —
torch is never a repo dependency and never in the inference path.

### Useful runtime env

`WASTE_PROFILE=1` (phase timings), `WASTE_CACHE_MB=N` (expert cache size in
the test harness), `WASTE_BACKEND=cpu` (disable SIMD/accelerator dispatch,
for bisecting numeric diffs), `WASTE_VERIFY=1` (crc32 every record on the
read path), `WASTE_THREADS`, `WASTE_CPUS` (cpu list the pool binds to —
`--cpus` on the CLI and the server, Linux and Windows; refused rather than
ignored elsewhere, see docs/ENGINE.md "Thread placement"),
`WASTE_DIRECT=0` (keep the page cache),
`WASTE_SPIN` (iterations a pool worker spins before parking; 0 = the pure
condvar pool), `WASTE_WIDE_MIN` (bytes below which a dispatch goes to the
fast group rather than waking the whole pool — 4 MB, measured; see
docs/LEARNED.md §67),
`WASTE_Q8=0` (dequantize the trunk to f32 at load, any width — 8x the RAM
on a 4-bit trunk, so it is out of reach on K3), `WASTE_I8MM=1`,
`WASTE_TOK_PLAIN=1`, `WASTE_VIS_STAGE`, `WASTE_DUMP_LATENT/HIDDEN`,
`WASTE_DUMP_DSA` (the sparse-attention selection: which pools won and on
what scores, so two implementations can be diffed on the decision rather
than on the logits it produced), `WASTE_DUMP_ROUTE` / `WASTE_DUMP_SCORES`
(the same, for the MoE router: the top-K ids that won, and the full score
vector they were ranked from — see `tests/route_diff.py` below).

MoE scheduling and the VQ4P kernel: `WASTE_XPAR=1` (one task per routed
expert instead of one per row range — **off by default**: worth ~1.18x on
Kimi-Linear and a regression on K3, because the batch that gives it
parallelism is the same batch that barriers the read-ahead, LEARNED §44),
`WASTE_XPAR_BATCH=N` (experts held at a time, default 4),
`WASTE_P6_CHUNK=N` (rows per chunk in the VQ4P apply, in index blocks,
default 16). **These three and `WASTE_THREADS` interact, and the best
setting inverts between models** — on Kimi-Linear `WASTE_XPAR=1` is worth
1.24x and six threads beat eighteen; on K3 six threads are 34% *worse* than
the default because its applies are large enough to use the E-cores too.
LEARNED §47 has the table; do not carry a setting from one model to the
other. Building with `-DWASTE_P6_SCALAR` drops the VQ4P kernel to its
portable path; the two are meant to be **bit-identical**, not merely close,
and that is checkable rather than asserted — see LEARNED §43 for why an
int8 lookup table raises the bar that far.

Profiling a decode step:
`WASTE_PROFILE=1 WASTE_CACHE_MB=17735 ./test_forward MODEL ids out.bin 5`.

## Architecture

### Library first — the rule that keeps it honest

`src/waste.h` is the entire public surface (~26 functions, opaque
`waste_ctx`, no global state, errors returned and never printed, nothing
calls `exit()`). **If the CLI needs a capability, it goes into `waste.h`
first.** `cli/main.c` and `serve/` are both clients of that header and touch
nothing private; `serve/engine.py` reaches it through ctypes rather than
keeping a second copy of the model code in Python. Argument parsing,
logging, signal handling and config files belong to the host, not the API.

### The C engine (`src/`)

| file | role |
|---|---|
| `waste.c` | public API, memory planning, budget arithmetic |
| `model.c` | container load + forward pass; one token per call (prefill is repeated steps, so decode is the only path) |
| `ecache.c` | bounded LFRU expert cache over the per-layer banks |
| `kda.c`, `kda_neon.c` | Kimi Delta Attention recurrence |
| `vq.c` | residual VQ decode; also built standalone as `libwastevq` for `convert.py` |
| `vision.c`, `image.c` | the 27-layer ViT + projector, and file → patch tensor |
| `tokenizer.c` | tiktoken BPE in C, Unicode classes coded directly (no regex engine) |
| `backend.c`, `simd_*.c`, `metal.m` | kernel dispatch |
| `platform.h` | the six calls that are not POSIX (Windows) |

### Container format (`docs/FORMAT.md`, `src/waste_format.h`)

A `.waste` model is a **directory**: `manifest.json`, `trunk.bin`,
`experts-L{n}.bin` per MoE layer, `codebooks.bin`, `tokenizer.model`,
`specials.json`, plus optional `vision.json` / `chat.json` / `usage.waste`.

The invariant that gives the whole design its speed: every expert's
gate/up/down matrices live in **one 4 KiB-aligned record**, so routing to an
expert costs exactly one `pread`. Reads bypass the page cache (`F_NOCACHE`,
`O_DIRECT`, `FILE_FLAG_NO_BUFFERING`) — deliberately, because a container
smaller than RAM would otherwise produce hit rates that are fiction.

**Containers are untrusted input.** The manifest is hand-parsed JSON with
every dimension bounded by `cfg_sane()`; every record's header is validated
on the read path (magic, the expert the index asked for, offsets that fit)
at O(1) cost. The payload `crc32` is `--verify` / `WASTE_VERIFY=1`, off by
default. `tools/fuzz_container.py` exists for exactly this surface — run it
after touching the manifest parser or the record path.

### Backend dispatch (`docs/BACKENDS.md`)

One struct of function pointers, filled with a CPU baseline that is always
compiled in and always correct; better backends overwrite the slots they
implement. Dispatch resolves once at init, never in a hot loop, and no
backend does dynamic loading. SIMD lives in one translation unit per ISA
(`simd_avx2.c`, `simd_avx512.c`, `kda_neon.c`) selected at run time from
CPUID, so a single x86 binary adapts. In the Makefile those objects use
`override CFLAGS += -mavx2 …` — the `override` is load-bearing, since
`make asan` re-enters make with CFLAGS on the command line and would
otherwise silently drop the ISA flags.

Accelerator backends are build-time options and each must have its source
file present, or the build stops with a message (CI checks that).

### Memory budget (`docs/ENGINE.md` §3)

`waste_plan_memory` computes a floor (resident trunk + state + scratch +
minimum cache). A budget under the floor is **refused** with
`WASTE_E_RAM_BUDGET` rather than swapping. A budget of 0 means the engine
chooses: it steps down a whole *token working set* at a time from however
many of them cover the container's **entire expert set** (`plan.bank_bytes`,
the point past which a bigger cache cannot help) until it fits under 3/4 of
`waste_usable_ram()`, then says on stderr what it picked. On a container
smaller than the machine that lands on every expert resident — `waste plan`
calls it **fully resident** — and a background thread then reads the bank in
so the demand stream does not have to discover it a miss at a time
(`WASTE_PRELOAD=0` turns that off).

Cache size is only meaningful in whole multiples of one token's working set
(K3: 16 experts × 92 layers ≈ 17 GB). Below one multiple the hit rate is
zero, not low. Above the machine's comfort it is worse than useless — 58 GB
measured 8× slower than 46 GB on a 64 GB machine, which is why the ceiling
is a fraction of RAM and not the bank. `tests/check_budget.sh` verifies peak
RSS actually stays inside the ceiling.

Residency also decides *scheduling*: `moe_layer` runs one task per routed
expert when the layer's experts are already cached and one per row range
when they are not, because holding K records before doing any arithmetic is
a barrier against the read-ahead. `WASTE_XPAR=0/1` forces it; the default
asks the cache. The two paths are **bit-identical** and `tests/run.sh`
asserts it — an automatic choice that changed the numbers would make results
depend on how warm the cache happened to be.

### Prompt safety: markup vs content

The tokenizer has two entry points and the split is a security boundary,
not a convenience. `waste_tokenize_markup` resolves `<|open|>` to a control
token; `waste_tokenize` treats the same bytes as ordinary text. Template
structure goes through the first, and **anything a user, document or tool
wrote goes through the second**. Concatenating them into one string and
encoding once would let content forge a system message with real control
token ids. `serve/xtml.py` therefore emits a list of `Segment(text,
markup=)` — never a string — and `serve/engine.py` tokenizes them one
segment at a time.

### The server (`serve/`, `docs/SERVE.md`)

Stdlib-only OpenAI-compatible HTTP. `xtml.py` is a **port** of the release's
`encoding_k3.py` (K3 ships no Jinja template), checked segment-for-segment
against that file whenever `K3_DIR` is set; `regions.py` is the streaming
parser that reads replies back into reasoning / content / `tool_calls`;
`chatfmt.py` is the fallback for a container with no XTML markers, serving
it from the same `chat.json` the CLI reads — plain conversation, with
thinking and images from the format and everything else refused by name
rather than dropped; `kimitools.py` and `glmtools.py` are the two native
tool-call protocols a container's *tokenizer* can carry while its
`chat.json` says nothing about them — Kimi K2's five control tokens and
GLM-5.3-Flash's `<tool_call>` XML grammar, each with its rendering and its
reply reader, enabled only when the whole marker set resolves.
`tests/serve/test_chatfmt_upstream.py` diffs the first against K2's
published one and `tests/serve/test_glm_upstream.py` the second against
GLM's, `K2_DIR`/`GLM_DIR` naming the release, the way `test_xtml` does for
K3;
`engine.py` is the ctypes binding plus one lock held for a whole generation
(a `waste_ctx` is not thread-safe). Struct layouts in `engine.py` mirror
`waste.h` field for field — change one, change the other.

## Conventions

- **`docs/LEARNED.md` is append-only and dated. Later wins.** Refuted ideas
  stay in with the numbers that killed them (3-bit trunk, GPU offload,
  index-layout blocking, per-expert bit allocation). Read it before
  proposing an optimization — several obvious ones are already measured and
  dead. Do not quietly correct an old number; append the new one.
- **`CHANGELOG.md` is updated with every new tag, in the same commit that
  bumps `WASTE_VERSION_*` in `src/waste.h`.** A release whose changelog
  lands later is a release nobody can read from the outside — LEARNED.md
  carries the reasoning, but it is dated by experiment, not by version, and
  a user asking "what changed in 0.6.2" cannot reconstruct it from there.
  Record what was measured and *not* adopted too: the measurement is the
  useful part even when the feature is gone.
- Every number in `README.md` and the docs was measured on the commit it
  ships with. Don't add a figure you haven't run, and don't restate one
  under a change that would move it.
- A missing prerequisite is a **SKIP**, never a silent pass. `tests/run.sh`
  exists because two checks once quietly did not run.
- SPDX `Apache-2.0` header on every `src/*.{c,h,m}`, `cli/*.c`, `tests/*.c`,
  `tools/*.{py,sh}` — CI fails the build without it.
- Python (`tools/`) converts and validates models. It never runs alongside
  the engine, and torch is never a dependency of the inference path.
- **`convert.py --reclaim on` deletes source shards as it consumes them**,
  which is what lets K3 convert on one disk instead of two (1.42 TB of
  staging beside a 982 GB container). It is safe because every tensor has
  exactly one consumer, it refuses before deleting rather than during, and
  it is **not reversible** — a reclaimed shard has to be downloaded again
  and `verify_container.py` can no longer check the container against its
  source. Prove a recipe with `--reclaim dry` first; docs/K3.md has the
  refusals and the ledger discipline.
- Comments here explain *why*, usually with the failure that motivated
  them. Match that: a comment that only restates the code is noise, but the
  Makefile's and CI's explanations of past breakage are load-bearing.
