# WARP — Weight-Aware Runtime and Paging (formerly WASTE)

WARP is an embeddable inference engine written in C, with no third-party runtime dependencies. It keeps the model trunk in memory, streams selected experts directly from disk, and uses the remaining RAM as a bounded expert cache.

The project is driven by humans: the ideas, hypotheses, priorities, tests, and decisions are human. The code is written by LLMs. At this scale, that is the only way to iterate on new algorithms and test hypotheses fast enough.

The goal is to run huge frontier models such as Kimi K3 on consumer hardware. Today, the complete 2.78-trillion-parameter Kimi K3 runs on a 64 GB MacBook Pro at about **0.6 tokens per second**, and the 313-billion-parameter GLM-5.3-Flash — text and images — at about **3.9**.

**Ultimately we want WARP to execute Kimi K3 locally to improve itself** (we are currently using Opus 5 with extra thinking).

WARP is intentionally narrow, and it exists to find out how far local inference can be pushed when model weights live mostly on fast storage instead of RAM.

```text
$ waste run ~/models/k3.waste 'What is the capital of Italy?'
waste: no --budget, using 46.39 GB of 64.00 GB (expert cache 17.56 GB)
The capital of Italy is **Rome**.
[16 tokens, 26.87 s, 0.60 tok/s | experts 9000 hit / 14552 miss = 38%]
```

**This is the full model, not a distilled or pruned version.** Its published weights occupy 1.42 TB; the converted WARP container is 982 GB.

## How it works

Kimi K3 is a mixture-of-experts model. It has 2.78 trillion parameters, but only about 4% of them are active for each token. WARP keeps the shared part of the model in RAM and reads only the selected experts from disk.

The container is arranged so that one expert requires one aligned read. Those reads overlap with computation, while unused RAM becomes a bounded expert cache. A lookahead router predicts the experts needed by the next layer and starts reading them early; the real router still makes the decision, so this changes timing, not the result. Experts use 3-bit residual vector quantization, while the more sensitive shared weights remain at 4 or 8 bits.

K3's linear attention and compressed latent KV cache also matter: at 4K context, the KV cache is about 0.21 GB instead of 11.25 GB. The result is an engine that needs 29.19 GB to open K3 and uses the rest of the available memory to avoid repeated disk reads.

For the full design and measurements, see [docs/ENGINE.md](docs/ENGINE.md) and [docs/EFFICIENCY.md](docs/EFFICIENCY.md). The on-disk layout is documented in [docs/FORMAT.md](docs/FORMAT.md), while [docs/KDA.md](docs/KDA.md) describes Kimi Delta Attention.

## Performance

Measured on a 64 GB MacBook Pro with an M5 Pro and the model container on the
internal SSD:

| Model | Container | Minimum RAM | 64 tokens | 200 tokens |
|---|---:|---:|---:|---:|
| Kimi K3 2.78T | 982 GB | 29.19 GB | 0.45–0.62 tok/s | — |
| GLM-5.3-Flash 313B | 112 GB | 5.14 GB | 3.32 tok/s | **3.86 tok/s** |
| Kimi-Linear 48B | 19 GB | 1.32 GB | 14.29 tok/s | **17.22 tok/s** |

The longer run is faster because the expert cache is still filling during
the first few dozen tokens; both columns are what the same command prints,
not a steady state extrapolated from it. K3 has no 200-token column here
because one run of it takes ten minutes and reads 4.6 TB.

For K3, 64 GB is the practical minimum. A 32 GB machine can open the model but will page heavily. The default memory budget on the test machine is 46.39 GB, including a 17.56 GB expert cache.

Kimi-Linear's figure is the one that moved: the automatic budget used to stop three working sets short of the machine, so a 19 GB container got a 1.65 GB cache on a 64 GB laptop. It now climbs to the container's whole expert set when the machine has the room — 18.48 GB resolved, every expert resident — and that is worth 11.13 → 12.60 tok/s over 64 tokens, with the bytes read falling from 66.3 GB to 17.7. On top of it the thread pool stopped waking its efficiency cores for jobs too small to hide the ~54 µs that costs, which is another 14.41 → 16.74 over 150 tokens. K3 is unchanged by both: its 962.83 GB of experts do not fit on any machine here, and at 465 GB read per 20 tokens neither residency nor dispatch is where its time goes. [docs/LEARNED.md](docs/LEARNED.md) §66, §67.

Most of that requirement is the 27.28 GB resident trunk rather than the cache. Shrinking the expert cache from 17.32 GB to 3.32 GB costs about 10% of throughput; enlarging it past the default costs everything. Measured across four cache sizes in one process:

| expert cache | hit rate | decode |
|---:|---:|---:|
| 3.32 GB | 29.1% | 0.56–0.58 tok/s |
| 17.32 GB | 36.2% | **0.63 tok/s** |
| 23.32 GB | 38.4% | 0.07–0.09 tok/s |
| 29.32 GB | 41.3% | 0.07–0.08 tok/s |

The last two rows are the failure mode worth knowing about: the hit rate keeps climbing and the bytes read keep falling while throughput drops eightfold. The engine is inside its budget and the machine is not, so a cache hit becomes a page fault. Giving the process more memory is not always faster.

Decoding with fewer experts per token is a knob rather than a rebuild:
`num_experts_per_token` in the container manifest. K3 ships at 16. Measured
on this machine, one load with the arms interleaved:

| experts/token | decode | KL from top-16 | working set |
|---:|---:|---:|---:|
| 16 | 0.59 tok/s | — | 17.01 GiB |
| 12 | 0.70 tok/s | 0.007 | 12.76 GiB |
| **8** | **0.89 tok/s** | **0.037** | **8.50 GiB** |
| 4 | 1.06 tok/s | 0.118 | 4.25 GiB |

Top-8 is 1.49x for a divergence twice that of a quantization this project
rejects elsewhere, and it reproduces top-16's greedy continuation on the
prompts tested. Top-4 does not: its next-token distribution still looks
close, and it stops following the prompt within a few tokens — which is why
the gate here is a continuation and not a KL. This is a quality trade and
the default stays 16.

Storage is the main constraint. A cold K3 token reads about 17 GB of experts. The internal SSD sustains 12.78 GB/s; a tested USB enclosure managed 0.94 GB/s. Put the converted container on internal NVMe storage.

If you have more than one drive, since 0.7.2 the expert banks can be spread
across them: `WASTE_BANK_SHARDS=/mnt/a,/mnt/b` reads expert `e` from shard
`e % N`, so the k experts a single token routes to land on different
devices instead of queueing behind one. `tools/split_banks.py` writes and
byte-verifies the shard sets, and the logits are identical either way.
**No speedup is claimed here** — that needs two drives of comparable speed
and a real container. Striping across the internal SSD and the USB
enclosure above would measure the enclosure, not the striping. The
mechanism ships; the measurement does not.

All layers are checked against a PyTorch reference. Final logits agree within 3.6e-06, and the vision tower agrees with its oracle within 2.3e-06.

Additional measurements, profiling data, router-lookahead results, and quantization experiments are collected in [docs/TECHNICAL.md](docs/TECHNICAL.md).

## Vision

Kimi K3 and GLM-5.3-Flash are both multimodal, and WARP can use one or more images together with text. Pass `--image` once per image:

```bash
./waste run ~/models/k3.waste "Describe this image" --image photo.jpg
./waste run ~/models/k3.waste "Compare these images" \
    --image before.png --image after.png
```

In interactive mode, `/image FILE` attaches an image to the next message. An image is expanded into many prompt positions: an 896×896 image uses 256 positions at the default patch budget. The vision tower takes about 15.7 seconds for 1024 patches on the test machine, but most of the cost comes afterward because every image position passes through the language model like a text position. In the current K3 measurements, that is about 2.8 seconds per image position.

GLM's tower is a different one and cheaper to feed: the 200×140 picture in
the [GLM section](#glm-53-flash) costs 40 prompt positions, and generation
after it runs at the same speed as without it. Its tower is 282 MB against
K3's 434 MB, and both are loaded only when images are asked for.

See [docs/K3.md](docs/K3.md) and [docs/GLM.md](docs/GLM.md) for the two vision architectures and their measurements, and [examples/README.md](examples/README.md) for CLI, C, and HTTP multimodal examples.

## Other models

K3 is the target and the best-tested model, and Kimi-Linear is the small one to
start with. Since 0.6.8 the converter and the engine also handle the
**DeepSeek-V3 family — V3, R1 and Kimi K2**, which needed two changes rather
than one.

`convert.py` now reads fp8 block-scaled weights, applying the per-tile scales
these checkpoints ship in a companion tensor, and normalises DeepSeek's MoE
tensor and config names to the single spelling the engine reads. And MLA now
applies rotary to its rope dims. The engine had implemented none: the Kimi
models set `mla_use_nope` and pass those dims through unrotated, which is
correct for them and wrong for everything in the V3 family, where in MLA those
dims are the only positional signal there is. A container built before this was
not degraded, it was unordered — it could not tell which turn of a conversation
came first.

No throughput figures here, because nobody on this project has a K2 container.
What has been measured, by [@fab2s](https://github.com/fab2s) who contributed
both changes: a `Kimi-K2-Instruct` conversion — 61 layers, 384 experts top-8,
VQ3R, a 354 GB expert set and a 6.9 GB trunk — opens and reports 1.03 T
parameters total, 31.69 B active per token; and the rotary arithmetic agrees to
0.000023% relative L2 with an oracle whose YaRN helpers are taken verbatim from
the DeepSeek release's own `modeling_deepseek.py`.

Kimi K3 and Kimi-Linear are unaffected: their forward pass is byte-identical to
0.6.7, by construction rather than by a runtime branch.

### GLM-5.3-Flash

`zai-org/GLM-5.3-Flash` — 313 B parameters, 328 GB of fp8 as published — is
converted and running, text and images.

```
$ waste run ~/models/glm53.waste "What is the capital of Italy? Answer in one sentence."
waste: no --budget, using 46.37 GB of 64.00 GB (expert cache 41.36 GB)
The user is asking a simple factual question: What is the capital of Italy?
They want the answer in one sentence.

The capital of Italy is Rome. This is a well-established fact. I should
answer in one sentence as requested.</think>The capital of Italy is Rome.
[56 tokens, 12.79 s, 4.38 tok/s | experts 17327 hit / 1489 miss = 92%]
```

Everything before `</think>` is the model's reasoning. GLM's generation
prompt always opens that channel and the model closes it itself; the CLI
prints both, and over HTTP they come back as `reasoning_content` and
`content` separately.

| | |
|---|---:|
| parameters | 313.89 B total, 17.31 B active per token |
| container | 112 GB — 5301 MB trunk, 42 expert banks of 2598 MB |
| minimum RAM | 5.14 GB, plus 805 MB when images are enabled |
| default budget here | 46.37 GB, of which 41.36 GB expert cache |
| decode | 3.32 tok/s over 64 tokens, **3.86 over 200** |

Against a PyTorch oracle built from the same container: relative L2
**2.41e-5**, argmax and top-10 identical.

**It is the model that fits this class of machine.** K3 needs 29.19 GB
before it caches a single expert and then gets a token's working set and a
half; GLM's floor is 5.14 GB, so a 64 GB laptop caches 36% of its entire
expert set — and a much smaller machine still runs it. Measured over 64
tokens, varying only `--budget`:

| budget | expert cache | hit rate | read | decode |
|---:|---:|---:|---:|---:|
| 9 GB | 4.0 GB | 66.0% | 228 GB | 2.82 tok/s |
| 12 GB — what a 16 GB machine resolves | 7.0 GB | 70.2% | 190 GB | 2.99 tok/s |
| 16 GB | 11.0 GB | 74.0% | 160 GB | 3.06 tok/s |
| 24 GB | 19.0 GB | 79.4% | 121 GB | 3.14 tok/s |
| 46 GB — the default here | 41.4 GB | 87.7% | 73 GB | 3.32 tok/s |

The curve is shallow because the reads overlap the arithmetic: six times the
cache cuts the disk traffic by two thirds and buys 18% of throughput. What
that means in practice is that **a 16 GB machine runs a 313 B model at 90%
of the speed a 64 GB one does** — the engine leaves a quarter of RAM to the
OS, so it resolves about 12 GB there — and more RAM mostly buys quiet disks.
Until the disk is slow: on the 0.94 GB/s enclosure this project has
measured, 228 GB against 73 is four minutes of reading against one.

### What is new in it

It turned out to be mostly this engine already: the same KDA recurrence, the
same MLA with `kv_b_proj` absorbed, the same router, the same fp8 reader and
the same nested config layout as K3. Three things are new and each is behind
a config key that is absent everywhere else: **mHC**, which carries four
parallel residual streams instead of one and mixes them through a
Sinkhorn-projected matrix at every sublayer; a **clamped SwiGLU**; and
**DeepSeek Sparse Attention** in its k-pool form, where a full-attention
layer scores pools of four cached tokens and attends over the best 512 of
them plus the tail.

The re-encoded tokenizer agrees with the release's own on 21 of 21 strings,
and VQ3R lands at the same 0.195 relative error on GLM's experts as on K3's.

### Images

Its vision tower is a second one — 24 blocks with 2D rope, per-head q/k
norms, a gated merger — and matches its own PyTorch oracle to rel L2 3.3e-5:

```
$ waste run ~/models/glm53.waste "What does this image look like? One sentence." \
      --image x.png -n 200
[x.png: 40 image tokens]
The image is a colorful, abstract pattern. It consists of diagonal stripes
of various colors (green, blue, purple, pink, yellow, red) with some
vertical lines within them.</think>This image displays a vibrant, abstract
pattern of diagonal stripes in various colors like green, blue, purple, and
pink, overlaid with fine vertical lines.
[104 tokens, 24.98 s, 4.16 tok/s | experts 32067 hit / 2877 miss = 92%]
```

`x.png` is a 200×140 test pattern generated from
`(x*7+y*3, x*x+y, x+y*11) mod 256`, which really is diagonal colour bands
with vertical structure — the description is of the file, not of a
plausible-sounding picture.

An image costs the context what text of the same length costs: 40 merged
tokens for that 200×140 picture, and the generation that follows runs at
the same speed as any other. The tower itself is 282 MB and is loaded only
when images are asked for.

[docs/GLM.md](docs/GLM.md) has the architecture, the three places the
release states something differently, and what is still left out.

## What you need

To build and test WARP:

- a C11 compiler and `make`;
- macOS, Linux (arm64 and x86_64), or Windows via MinGW-w64 — CI builds
  and runs the suite on all four, plus an ASan/UBSan job;
- no BLAS, Python, CUDA, or other external dependency for the current CPU
  inference path.

To run GLM-5.3-Flash, which is the one most machines can hold:

- **16 GB of RAM is enough**; 5.14 GB is the hard floor at 4K context, and
  64 GB is what the measurements above were taken on;
- **112 GB of internal NVMe storage** for the converted container;
- another **306 GiB of temporary storage** if converting the published
  weights yourself. This staging storage may be external and can be freed
  afterward, or reclaimed as the conversion proceeds.

To run Kimi K3:

- **64 GB of RAM recommended**; 29.19 GB is the hard floor at 4K context;
- **about 1 TB of internal NVMe storage** for the converted model;
- another **1.42 TB of temporary storage** if converting the published weights
  yourself. This staging storage may be external and can be freed afterward.

If you only want to try the engine, start with Kimi-Linear. Its container is 19 GB, it needs 1.32 GB of RAM, and it runs at about 14.4 tok/s on the same machine.

Python, PyTorch, and safetensors are needed only for model conversion and validation, never for inference.

## Getting started

Build the engine and run the model-free test suite:

```bash
git clone https://github.com/sqliteai/warp
cd waste
make
make check
```

`make` builds the `waste` CLI and `libwaste.a`. `make check` creates a small synthetic model, so it does not download weights.

### Quick start: GLM-5.3-Flash

The shortest path to a working 313 B model on a laptop. Every figure below
was measured on the machine at the top of this file; the download and the
conversion are both resumable and both safe to kill.

**Before you start**, you need room for two things at once: **306 GiB** of
published weights on the staging disk, and **112 GB** for the container. The
container belongs on internal NVMe — a container on an external disk is
correct and slow, and the difference is 12.78 GB/s against 0.94 on a tested
enclosure. The staging weights can live anywhere.

```bash
# 1. Build. Takes under a minute; no weights involved.
git clone https://github.com/sqliteai/warp
cd waste
make

# 2. Check the download before starting it: shard count, size, free space.
tools/fetch_weights.sh --repo zai-org/GLM-5.3-Flash \
    --dest /Volumes/staging/glm53 --dry-run

# 3. Download. 62 shards, 306 GiB. About 2 hours here, at a rate that
#    varied between 36 and 97 MB/s. Re-run it if it stops; nothing
#    already fetched is fetched twice.
tools/fetch_weights.sh --repo zai-org/GLM-5.3-Flash \
    --dest /Volumes/staging/glm53

# 4. Convert. About 45 minutes with three workers: 42 expert layers at
#    ~160 s each, then the trunk. Put the output on the internal SSD.
uv run --with torch python tools/convert.py \
    --src /Volumes/staging/glm53 \
    --out ~/models/glm53.waste \
    --jobs 3

# 5. Run it. Leave room for the reasoning channel: GLM thinks before it
#    answers, and -n counts both.
./waste run  ~/models/glm53.waste "What is the capital of Italy?" -n 200
./waste chat ~/models/glm53.waste
```

That is all of it. There are no GLM-specific flags: the converter recognises
the architecture, writes the chat format and the vision config, and
re-encodes the tokenizer, and the engine picks its own memory budget and
says what it picked.

```
$ ./waste chat ~/models/glm53.waste
waste: no --budget, using 46.37 GB of 64.00 GB (expert cache 41.36 GB)
chat format from ~/models/glm53.waste/chat.json

> What is the capital of Italy? Answer in one sentence.
The user is asking a simple factual question: What is the capital of Italy?
They want the answer in one sentence.

The capital of Italy is Rome. This is a well-established fact. I should
answer in one sentence as requested.</think>The capital of Italy is Rome.
```

The text before `</think>` is the model's reasoning channel, which GLM
always opens and closes itself. Over HTTP it comes back as
`reasoning_content`, separate from the answer. `-n` counts both, so a
question that needs thinking needs a larger budget than the answer alone
suggests.

**What to expect on the way.** The first few dozen tokens are slower than
the rest — the expert cache is still filling — so a short reply runs at
around 3.3 tok/s and a long one settles near 3.9. Prefill runs at the same
speed as decode, so a 2000-token prompt takes minutes before the first
output token; that is a property of the engine and not of this model.

**If you have less RAM**, nothing changes about the commands: the engine
resolves a smaller budget on its own and says so. The table in the
[GLM-5.3-Flash](#glm-53-flash) section above measures what each budget
buys — 16 GB is enough, at 3.06 tok/s against 3.32.

**If you are short of disk**, `--reclaim on` deletes each source shard as
the converter finishes with it, so the peak is the container plus the
shards still owed instead of both in full. It is not reversible and
`tools/verify_container.py` can no longer check the result against its
source, so prove the recipe with `--reclaim dry` first — [docs/K3.md](docs/K3.md)
has the refusals and the ledger discipline.

### Get Kimi K3, already converted

The fastest route is to download the already converted container over
BitTorrent. It skips the 1.42 TB source download, the 4.7-hour conversion, and
the temporary staging storage entirely — only the 982 GB container lands on
disk. Any BitTorrent client will do; the torrent's own piece hashes verify the
container as it arrives.

```text
magnet:?xt=urn:btih:54db69b0df8baf5e617744dda5d46c90a2d0f632&dn=k3.waste&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Fopen.demonii.com%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce&tr=https%3A%2F%2Ftracker.tamersunion.org%3A443%2Fannounce
```

With [aria2](https://aria2.github.io/), which resumes and needs no GUI:

```bash
aria2c --dir ~/models --seed-time=60 \
  'magnet:?xt=urn:btih:54db69b0df8baf5e617744dda5d46c90a2d0f632&dn=k3.waste&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=udp%3A%2F%2Fopen.demonii.com%3A1337%2Fannounce&tr=udp%3A%2F%2Ftracker.torrent.eu.org%3A451%2Fannounce&tr=https%3A%2F%2Ftracker.tamersunion.org%3A443%2Fannounce'
```

This writes `~/models/k3.waste`, the directory the commands below expect. Point
`--dir` at internal NVMe storage: a container downloaded onto an external disk
has to be copied before it is usable at full speed. Raise `--seed-time` if you
can afford to share it back.

### Get Kimi K3, from the published weights

Convert it yourself if you would rather not trust a third-party copy, or if you
already hold the original weights. The download and conversion are resumable:

```bash
# Check required download space.
tools/fetch_weights.sh --dest /Volumes/staging/k3 --dry-run

# Download the original weights.
tools/fetch_weights.sh --dest /Volumes/staging/k3

# Convert them. Put the output on the internal SSD.
uv run --with torch --with safetensors python tools/convert.py \
    --src /Volumes/staging/k3 \
    --out ~/models/k3.waste \
    --jobs 3
```

Conversion takes about 4.7 hours with three workers on the test machine. See [docs/K3.md](docs/K3.md) for validation, recovery, and storage details.

### Run it

```bash
./waste plan ~/models/k3.waste
./waste run  ~/models/k3.waste "The capital of France is" -n 32
./waste chat ~/models/k3.waste
```

Do not set `--budget` unless you have a reason to. By default WARP chooses a
safe memory budget, reports it, and refuses to start below the model's floor.
Inside a container it sizes against the cgroup limit rather than the host's RAM.
Use `./waste --help` for the complete command list.

More CLI examples, including evaluation, tokenization, saved sessions, and multimodal prompts, are in [examples/README.md](examples/README.md).

### Serve it

The optional server implements the OpenAI chat-completions API:

```bash
make libwaste.dylib                 # use libwaste.so on Linux
python3 -m serve ~/models/k3.waste --port 8000

curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"k3","messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

It supports streaming, tools, structured output, thinking controls, and images. See [docs/SERVE.md](docs/SERVE.md) for the protocol and [examples/README.md](examples/README.md) for complete requests.

A GLM container is served the same way, from its own `chat.json`: plain
conversation and images, with the reasoning channel returned as
`reasoning_content` beside `content`. Tools work here too: GLM's tokenizer
carries its own tool protocol (`<tool_call>`, `<arg_key>`, `<arg_value>`)
as single tokens, so `serve/glmtools.py` renders a request and reads a
reply the way GLM's own `chat_template.jinja` spells them — flat XML, an
`<|observation|>` turn for results.

Kimi-Linear's is the other one. Since 0.7.2 a container whose tokenizer
holds all five of Kimi's native tool-call markers gets tool calling over
HTTP even though its `chat.json` describes only the ordinary turns — the
format lives in `serve/kimitools.py`, and the server says which of the
three capabilities a container has when it starts. All or none, for either
protocol: half of that rendering encodes as ordinary text, so a partial set
is a different protocol rather than a smaller one.

```bash
python3 -m serve ~/models/glm53.waste --port 8000
```

## Library

WARP is also an embeddable C library. The CLI and server both use the public API in [src/waste.h](src/waste.h). The inference path depends only on libc and pthreads. Text generation, memory planning, session persistence, and multimodal C examples are available in [examples/README.md](examples/README.md).

## Why the name

Every token answered by a cloud service is paid for twice: once on the invoice, and once in the electricity of a datacenter running a model that would fit — barely, awkwardly, but genuinely — on hardware already sitting on a desk. WARP means to be the first concrete step toward ending that waste of tokens — which the project's first name said outright, and which the rename did not change.

## Project status

The format and API are not frozen. K3 is the main target and the best-tested model. The CPU path is currently the fastest measured implementation for this workload, but it is not assumed to be the final answer. CUDA, Metal, and other hardware-specific optimizations remain to be explored and may provide significant gains. Current backend results are documented in [docs/BACKENDS.md](docs/BACKENDS.md), while open directions are tracked in [docs/RESEARCH.md](docs/RESEARCH.md). Read [docs/LEARNED.md](docs/LEARNED.md) before proposing an optimization: failed ideas and negative results are kept there deliberately.

Contributors are more than welcome. New experiments, support for additional hardware, and open discussion about how to improve performance are all encouraged—even when an idea produces a negative result.

The software is currently changing very quickly. Before each release, a large QA run is executed; however, instabilities are definitely possible.

Measurements are treated as experimental results rather than marketing numbers. Each result is tied to the hardware, container, configuration, and commit on which it was obtained; unstable measurements are reported as ranges, and results later found to be wrong remain recorded as such. The detailed snapshots are in [docs/TECHNICAL.md](docs/TECHNICAL.md) and the full history, including negative results, is in [docs/LEARNED.md](docs/LEARNED.md).

Validation covers more than successful generation. The model-free suite builds a synthetic container; real-model checks compare individual layers and final logits against PyTorch, verify conversion round trips, test vision against its oracle, and exercise the server prompt renderer segment by segment against K3's reference encoder. The validation criteria and current evidence are documented in [docs/GATES.md](docs/GATES.md), with server-specific differential tests in [docs/SERVE.md](docs/SERVE.md).

On a mixture-of-experts model, a distance between logits is not by itself a verdict. A top-K router turns an arbitrarily small arithmetic difference into a discrete one, and past the first flipped expert the two paths are running different weights — the distance then measures how much the model cares which of two indistinguishable experts it used, not how far the arithmetic moved. Since 0.7.1 the suite compares the logits and, where they part, asks which decision moved and whether anything could have resolved it: [tests/route_diff.py](tests/route_diff.py) over the expert ranking and [tests/dsa_diff.py](tests/dsa_diff.py) over the sparse-attention pool ranking, each answering *identical*, *tie*, or *diverged*. Three checks were red against a 1e-3 threshold and none was an engine defect: on K3 the paths first disagree on the closest call in the entire forward pass, a relative margin of 7.3e-07 where the median decision is 7.4e-03, and on GLM they disagree on an exact tie between pools scoring zero. The result is a stricter suite rather than a looser one — a flip on a margin the reference could resolve now fails while naming the token and the layer, and a difference with the routing *unchanged* is its own verdict instead of being pooled with the tie. [docs/LEARNED.md](docs/LEARNED.md) §71–§72 has the distributions behind the thresholds.

The general form of that, and the reason 0.7.2 is mostly test code: a check that compares a thing to itself is not a weak oracle, it is not an oracle. Kimi's tool rendering shipped with 438 self-consistent tests, every one of them passing on a defect one of them had pinned; it is now diffed against the release's own chat template ([tests/serve/test_chatfmt_upstream.py](tests/serve/test_chatfmt_upstream.py)), the way K3's encoder has always been diffed against `encoding_k3.py`. The synthetic `index_bits 6` container the VQ4P checks run on is packed by a second implementation of the converter's own layout, so the two are now run against each other ([tests/test_vq4p_packing.py](tests/test_vq4p_packing.py)) — the arm itself cannot notice, since both backends would decode a wrongly packed container the same wrong way. Neither defect was visible from inside the code, and both were found by comparing against something outside it.

Useful references:

- [docs/FORMAT.md](docs/FORMAT.md): container format;
- [docs/BACKENDS.md](docs/BACKENDS.md): CPU, SIMD, and Metal backends;
- [docs/KDA.md](docs/KDA.md): Kimi Delta Attention;
- [docs/GATES.md](docs/GATES.md): correctness and performance gates;
- [docs/RESEARCH.md](docs/RESEARCH.md): current research directions.
- [docs/TECHNICAL.md](docs/TECHNICAL.md): detailed measurements and technical experiments.

## License

WARP is distributed under the permissive Apache 2.0 license, and **the project will always remain open source under a permissive license**. See [LICENSE](LICENSE).

Copyright 2026 SQLite Cloud, Inc.
