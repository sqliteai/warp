# What we know, and how we know it

Everything below was measured on this machine (MacBook Pro M5 Pro, 64 GB,
18 logical / 6 performance cores) unless marked otherwise. Where a belief
turned out wrong, the wrong version is kept — the refutations were worth
more than the confirmations.

Sections are dated and appended, never rewritten, so a number appears more
than once as the engine changed under it. **Later wins.** That rule also
covers the project's name: it was WASTE until 2026-08-10 and is WARP now,
and entries written before then say WASTE. They were not edited, for the
same reason the wrong numbers were not. The binary, the header, the
`WASTE_*` environment variables and the `.waste` container extension keep
the old name in either case. The decode
profile in particular is measured three times — §10 before the MLA
absorption, §12 with a cold cache, and the README with the cache at the
knee — and the shares move because the cache moves, not because one of
them is wrong. When two figures disagree, take the one with the later
date, and take the end-to-end numbers from the README.

**Two numbers were used twice.** There are two §33s and two §37s, all four
dated 2026-08-01, from sections that landed the same day. They are not
renumbered because both members of each pair are already cited from outside
this file — `CHANGELOG.md` cites the first of each (the oracle fixture, the
divide-by-zero), `GATES.md` and `tests/sweep.c` the second (the 52 GiB row,
the simulator) — and renumbering would silently redirect a released
changelog. **Cite these four by title, not by number**, and read a bare §33
or §37 as pointing at whichever one the surrounding sentence is about.

---

## 1. The model (see [K3.md](K3.md) for the full read)

K3 is 93 layers, 896 experts, top-16, hidden 7168 — but its MoE is a
**latent MoE**: experts operate on a 3584-wide projection of the hidden
state, not on the full width. That halves both expert size and per-token
I/O against the naive reading, and is what reconciles the announced 2.8T
with the 1.42 TB download. Weights ship **MXFP4, one E8M0 scale per 32**.

New since Kimi-Linear and still to implement: latent MoE projections,
Attention Residuals (`attn_res_block_size: 12`), SiTU activation, and a
full-rank KDA gate. No MTP head. K3 is also multimodal; the text path is
self-contained under `language_model.*`.

## 2. Storage: the enclosure matters more than the disk

`tools/diskbench.c` measures the engine's actual pattern — 12 MB records,
`F_NOCACHE`, `pread`, N threads — not sequential `dd`.

| device | random 12 MB reads |
|---|---|
| external USB SSD (ASM246X bridge) | **0.94 GB/s** |
| internal SSD | **12.78 GB/s** |

The external enclosure saturates at 0.94 GB/s and **does not scale with
threads**: the USB 10 Gbps bridge is the ceiling, not the NVMe inside it.
Hence the split: external disk holds the download and conversion staging,
the internal SSD holds the container the engine streams from.

## 3. Quantization: 3 bits, VQ, and no shared basis

Measured on real Kimi experts (`tools/quant_lab.py`), matched bit budgets:

| scheme | bits | weight error |
|---|---|---|
| rtn2-g64 | 2.25 | **71.8%** — the naive 2-bit strawman collapses |
| vq2 | 2.01 | 33.1% |
| **vq3** | **3.01** | **19.4%** |
| rtn3-g64 | 3.25 | 25.2% |
| rtn4-row (the known-good production default) | 4.01 | 15.2% |

VQ beats round-to-nearest decisively below 4 bits. 3 bits is the
operating point: 19.4% is within reach of the known-good int4 baseline,
and the 3-bit container **answers factual questions correctly** (see §6).

**The KBVQ shared low-rank component does not pay for itself.** It was the
centrepiece of the original format design. At rank N/128 it costs 0.12
bits and buys 0.3 pp — noise. At equal budget it loses badly: 28.9% error
against plain per-row INT4's 15.2% at the same 4.01 bits. The structural
reason, measured separately: **Kimi's experts are nearly mutually
orthogonal** — pairwise overlap of their rank-72 dominant subspaces is
0.046 against a random baseline of 0.031. It is parked, not deleted,
because the measurement is in the unweighted metric and "KLT-guided"
probably means activation-whitened; the revive/delete criterion is written
into [FORMAT.md](FORMAT.md).

## 4. Routing and caching: the cache floor is one token

Two independent measurements, simulated then real.

**Gate 2 (simulated, from a real trace).** 300 batch-1 decode tokens of
Kimi-Linear: 208 of 6656 (layer, expert) slots touched per token = 3.1%;
next-token reuse **33.6%**, *down* from OLMoE's 43.5% — reuse falls as
experts get finer-grained, the direction that matters for K3's 896. But
concentration rises: the top 8.7% of slots cover half the activations.

**Gate 5 (measured, by the engine's own cache).** `src/ecache.c` with
`pread` + `F_NOCACHE`, so the kernel's page cache cannot flatter the
result — essential, because with a 17 GB container on a 64 GB machine the
kernel was silently caching everything.

| cache % of expert set | 3.0 | 6.0 | 12.1 | 24.2 | 48.4 |
|---|---|---|---|---|---|
| **measured hit** | 13.2% | **40.3%** | 61.9% | 84.8% | 93.9% |
| simulated | 29.4% | 40.6% | 54.9% | 71.9% | 87.4% |

Agreement is close at the K3-relevant 6%, and the real cache does better
above 12%.

**The disagreement at the bottom is the most useful result.** At 1.5% the
hit rate is *exactly zero* — 2604 evictions in 2704 accesses. One token
touches 208 experts; a 100-slot cache keeps nothing alive to the next
token. **The cache floor is one token's working set.** For K3 that is
16 x 92 x 11.8 MB = **17.4 GB**; a 64 GB machine clears it 2.6x, a 32 GB
machine does not.

Policy matters at the margin: LRU collapses to 5.1% where LFRU still gets
29.4%. Frequency-first is load-bearing, not a refinement.

## 5. The KDA kernel

`fla`'s `naive_recurrent_kda` is plain PyTorch, so the *official*
reference runs on Apple Silicon even though the production Triton kernels
do not — `tools/kda_ref.py` loads it by file path around the package
`__init__`. The C kernel matches it to **4.1e-08** at Kimi-Linear's real
shape.

Writing it corrected the drafted recurrence: the delta term uses the
**decayed** state `S' = Diag(exp(g))·S`, not `S_{t-1}`, and `g` is
log-space. With the draft version the model would have produced plausible
but wrong output — the expensive kind of bug.

## 6. The engine works

C forward pass vs the pure-PyTorch oracle on the same container: max abs
logit diff **4.2e-05**, relative **1.5e-06**, argmax and top-10 identical.
End to end, the 3-bit container generates:

> The capital of France is **Paris, and the capital of Italy is Rome. The
> capital of Spain is Madrid, and the capital of Germany is Berlin.**

That one sentence validates converter, container layout, quantization and
four layer types at once.

## 7. Optimization: 17.9x, and two refuted theories

2.15 → 0.12 s/token, logits unchanged at every step.

| step | s/token | what the profile said next |
|---|---|---|
| first correct version | 2.15 | MoE 71% |
| NEON dot + thread pool | 0.93 | expert **dequantization 87.5%** |
| fused VQ matvec (no dequant) | 0.22 | expert matmul 67% |
| hoist gate/up tables | 0.18 | apply 40% |
| unrolled gather chains | **0.12** | apply 36% |

**Never dequantize an expert.** `sum_s C_s[i]·x_v` depends only on
(stage, code, vector position), never on the output row — tabulate it once
per matrix and a row costs 3 lookups per 8 weights. This is
sqlite-vector's turbo-LUT idea applied to a weight matrix. Dequantization
went from 87.5% of the time to zero.

**Two hypotheses measured and refuted**, both plausible:

- *Index locality.* Blocking the index layout so a tile's rows are
  contiguous measured **1.44x in isolation** — and changed nothing in the
  real engine. The microbenchmark did not model 12 threads sharing L2. It
  cost a full reconversion of 19 GB.
- *Table bandwidth.* Re-reading the 884 KB table per 64-row tile is
  8.2 GB/token = 165 GB/s, suspiciously exactly this machine's ceiling.
  Cutting that traffic made it **slower** — the table is shared read-only
  and stays cached; the extra index streams do not.

What actually moved it: each gather is load → address → load, ~5 cycles,
and the loop ran one chain at a time. Interleaving four rows fixed it.

**int8 and SDOT, honestly.** SDOT does not apply to the expert matmul at
all — that inner loop is a gather, and ARM has no gather instruction. It
does apply to the trunk, and delivers 1.9x there (0.30 → 0.16 s), but the
trunk is ~16% of a token so Amdahl caps the end-to-end gain at 13% — which
the exact f32 path matches without quantizing activations. Quantizing them
costs four orders of magnitude of accuracy and reorders the top-10.
Default is therefore int8 *storage* with f32 arithmetic: same numbers,
**5.6 GB of RAM freed**, and by Gate 5 that RAM is worth more as cache
than the 13% was.

## 8. Projection for K3 on 64 GB

Composing the measured pieces at 3 bits: 954 GB of experts on the internal
SSD, 17.1 GB read per cold token, ~46 GB of cache = 2.6x one token's
working set, which Gate 5 puts at ~40% hit → ~10 GB/token from disk at
12.78 GB/s ≈ **0.8 s/token of I/O**, plus compute.

So **~1 tok/s**, in the same range as the earlier 1.5 tok/s estimate but
now built entirely from measurements rather than from literature. The
remaining unknown is whether K3's 896-expert latent routing caches like
Kimi-Linear's 256 — that is the first thing to measure once the download
lands.

## 9. Method notes worth keeping

- **Test before long operations.** Gate H saved a 1.4 TB download onto a
  disk that cannot stream it; Gate 3 changed the format before any data
  was written in it.
- **Microbenchmarks lie about systems.** The 1.44x index-layout result was
  real in isolation and worthless in place.
- **Re-verify after every optimization.** Two bugs were caught only by the
  oracle diff: a thread-pool chunk size that broke block alignment, and a
  `tname()` static buffer that aliased when two names were passed to one
  call.
- **Numbers that look too neat deserve suspicion.** "165 GB/s, exactly the
  machine's ceiling" was a coincidence, not an explanation.

## 10. Where a K3 decode step actually goes (2026-07-28)

First real profile, 5 decode steps, `--budget 46G`:

| stage | share | note |
|---|---|---|
| expert I/O | 39.0% | 17.0 GB/token at ~9.9 GB/s |
| expert matmul | 26.1% | LUT build 8.2 + apply 17.5 |
| KDA layers | 19.1% | dominated by q/k/v/o/gate projections |
| MLA layers | 3.3% | |
| lm_head | 1.4% | |

17.0 GB/token is exactly the working set Gate 5 predicted, and 9.9 GB/s
is close to the internal SSD's measured 12.78 GB/s: **the I/O is already
near the hardware limit**, so it only gets cheaper by being read less
often, which means cache, which means RAM.

And RAM is where the real finding is. `waste plan` at a 46 GB budget:

```
resident trunk       28.61 GB
KDA state + KV       11.68 GB
minimum expert cache  0.38 GB
budget 46 GB -> expert cache 5.64 GB
```

5.64 GB against a 17.0 GB per-token working set is 0.33x — well under
the Gate 5 floor, which is exactly why the measured hit rate is 0.0%
across 8832 accesses. The cache is not underperforming; it was never
given enough room to hold anything.

**The KV cache is 53x larger than MLA requires.** The engine caches K and
V expanded per head — 96 x 192 + 96 x 128 floats per token per layer,
120 KB — when MLA's whole point is that only the 512-wide latent plus
the 64 rope dims need storing, 2.25 KB. Absorbing `kv_b_proj` into the
query (score against the latent directly, accumulate in latent space,
project once at the end) gives:

| ctx | cached now | absorbed |
|---|---|---|
| 4,096 | 11.25 GB | 0.21 GB |
| 32,768 | 90.00 GB | 1.69 GB |
| 131,072 | 360.00 GB | 6.75 GB |
| 1,048,576 | 2.81 TB | 54.00 GB |

It costs 3.2x the attention arithmetic (3.32 -> 10.57 GMAC/token, against
48.6 GMAC for the experts) and saves 53x the attention memory traffic
(11.25 -> 0.21 GB/token).

**Implemented, and it behaves as predicted.** Measured on K3:

| | expert cache | hit rate | tok/s |
|---|---|---|---|
| expanded, budget 46G | 5.64 GB | 0% | 0.28 |
| latent, budget 46G | 16.68 GB | 11% | 0.30 |
| latent, budget 56G | 26.68 GB | 32% | 0.34 |

(Re-measured after §11 corrected the scratch accounting, which costs
~0.7 GB of cache at every budget: 46G -> 9% and 0.29, 52G -> 24% and
0.31, 58G -> 34% and 0.32. Same curve, honest numbers.)

The 0% was never the cache underperforming — 5.64 GB against a 17.0 GB
working set is a third of the floor, so nothing could survive from one
token to the next. Given room, it behaves exactly as Gate 2's simulation
said it would. Logits are unchanged (max|diff| 1.19e-05 against the
expanded path, argmax and top-5 identical).

And long context stops being impossible: at 128K tokens the expanded
layout wanted 360 GB of KV, the latent wants 7.18 GB and still leaves
20.14 GB of expert cache inside a 56 GB budget.

**The report says the trunk should not be 4-bit.** QAT covered the expert
weights at MXFP4 with everything else in higher precision, so the model
has no trained tolerance for a quantized trunk — and mine is 26.33 GiB
of Q4G measuring 11.8% off the source weights (Q8G measures 0.64%). This
is a quality risk, not a speed one, and it trades directly against the
cache: raising the whole trunk to Q8G would cost the 11 GB that
absorbing the KV cache frees.

**No speculative decoding is available.** The report fine-tunes K3's MTP
layer into an EAGLE-3 draft whose input fuses the 1st, 4th and final
AttnRes blocks — representations this engine already materializes in
`m->blockres`. But the open release ships no MTP, draft, or fusion
tensors, so there is nothing to run.

Method note to add to §9: **WASTE_THREADS=1 is not a serial baseline on
this machine.** 6 of 18 logical cores are performance cores, and a
single-threaded process lands on an E-core; the same KDA kernel reads
17.21s that way against 4.23s with the pool merely available. The
parallel-vs-serial comparison has to be made with the pool alive.


## 11. The RAM budget was not a ceiling (2026-07-28)

`waste.h` calls `ram_budget_bytes` a hard ceiling on everything the engine
allocates. Measuring it on K3 — rather than on the small model the test
uses — showed peak RSS running 1.4 to 3.4 GB over a budget set near the
floor. Two independent causes, neither visible on Kimi-Linear.

**The trunk was loaded twice.** `load_trunk` slurped `trunk.bin` whole and
then copied each tensor out of the buffer, so loading wanted twice the
trunk resident: 57 GB on K3, before a single token. Every tensor knows
its own offset, so there was never a reason to hold the file. Reading
them one at a time with `pread` also cut load from 34s to 20s — most of
that being the memory pressure the second copy caused, not the I/O.

**Scratch was a guess.** The plan used a flat 64 MB + `hidden*64*4`. The
decode buffers alone are 252 MB on K3 (`e_gate`/`e_up`/`e_down` are
`moe_inter * hidden` floats each), and the chunked-prefill buffers — up
to ~500 MB at `WASTE_CHUNK_MAX`, allocated on first use and never freed —
were not counted at all. K3's floor is 30.38 GB, not the 29.69 the
planner claimed.

Two method notes worth keeping:

- **A test on the small model does not test the big one.** The budget
  check has been green since it was written; it runs on Kimi-Linear,
  where the scratch it fails to count is measured in single megabytes.
  Errors that scale with the model need a check that scales with it.
- **Peak RSS is a noisy detector on macOS.** Under pressure the kernel
  compresses anonymous pages, so the same pre-fix overrun measured
  anywhere between 1.3 and 3.4 GB depending on the run and the prompt.
  It is good enough as a guard and not good enough as a proof, which is
  why the accounting is now derived from the config rather than inferred
  from a measurement.


## 12. Where K3 stands, measured end to end (2026-07-28)

Everything below is on the 64 GB M5 Pro, container on the internal SSD.

| budget | expert cache | hit | decode | peak RSS |
|---|---|---|---|---|
| 32G | 1.99 GB | 0% | 0.28 tok/s | 30.9 GB |
| 36G | 5.99 GB | 0% | 0.28 tok/s | 34.9 GB |
| 40G | 9.99 GB | 0% | 0.28 tok/s | 38.9 GB |
| 46G | 15.99 GB | 9% | 0.29 tok/s | 43.8 GB |
| 52G | 21.99 GB | 24% | 0.31 tok/s | 49.8 GB |
| 58G | 27.99 GB | 34% | 0.32 tok/s | 55.7 GB |

Floor 30.38 GB at ctx 4096, 31.86 at 32K, 36.96 at 128K. Load 20s.
Prefill 0.47 tok/s chunked, 0.29 sequential.

Decode profile, no cache: MoE 88.6% (expert I/O 48.6, expert matmul
34.0), KDA 9.3%, MLA 1.9%, lm_head 0.2%.

The cache does nothing at all below ~16 GB and the curve only bends
once it passes one token's working set — the Gate 5 floor, still the
single most predictive number in this project. Nothing about the
architecture work changed that; it changed how much RAM was left over
to spend on it.


## 13. Reducing the trunk: one clean gigabyte, and a dead end (2026-07-28)

The trunk is 28.61 GB resident and every gigabyte of it is a gigabyte
the expert cache does not get. Where it goes:

| component | GB | note |
|---|---|---|
| shared experts | 5.84 | every layer, every token |
| g_proj | 3.93 | full-rank gates, 93 layers |
| o_proj | 3.93 | |
| q/k/v_proj (KDA) | 8.76 | 2.92 each |
| latent MoE down/up | 2.26 | |
| lm_head | 1.11 | whole tensor, once per token |
| embed_tokens | 1.11 | **one row per token** |
| everything else | 1.67 | |

Only the last line of that table is compressible without a cost.
**embed_tokens is 1.11 GB of which 7 KB is read per token**, so it now
stays on disk and the row is pread on use — bit-identical logits, floor
30.38 -> 29.27 GB. Everything above it is touched in full every token:
streaming it would cost more I/O than the freed cache could ever save.
lm_head is the near miss — 1.11 GB read per token to free 1.11 GB of
cache, which at the current knee buys about 2 points of hit rate, or
0.34 GB/token. A net loss of roughly 0.8 GB/token.

**The 3-bit trunk is refuted, this time on both axes.** It was parked
earlier as "cache prediction held, throughput did not", and the obvious
follow-up was that a vectorized 3-bit unpack would rescue it. Re-measured
now that MLA absorption has put the cache at the knee where extra room
actually pays:

| | resident | cache @46G | hit | tok/s | output |
|---|---|---|---|---|---|
| Q4G trunk | 27.50 GB | 17.10 GB | 12% | 0.23 | coherent |
| Q3G trunk | 21.13 GB | 23.48 GB | 29% | 0.16 | `+` and spaces |

It gets the better hit rate — 29% against 12%, exactly as predicted —
and is still 1.4x slower, because the scalar 3-bit unpack costs more in
the trunk matvecs than the cache saves in I/O (kda bucket 3.70s -> 10.39s
over 5 steps). And the vectorized unpack would not save it: the logits
land 36% off the 4-bit ones and generation collapses. The technical
report says why — QAT covered the expert weights at MXFP4 and left every
non-expert component in higher precision, so the trunk has no trained
tolerance for being squeezed. **Correct the earlier lead: vectorizing the
3-bit unpack would not make Q3G viable for the trunk.** The quality wall
sits in front of the speed wall.

## 14. O_DIRECT on Linux, written blind (2026-07-28)

`ecache.h` has always said reads must bypass the page cache, because with
a 17 GB container on a 64 GB machine the kernel would cache the banks and
every hit rate measured would be the kernel's rather than the engine's.
macOS said so with `fcntl(F_NOCACHE)`. Linux said so **in a comment
only** — `O_DIRECT` appeared in the header text and nowhere in the code.
Every number in this file would have been fiction on Linux.

O_DIRECT wants three alignments: offset, length and destination buffer.
The first two come free — expert records are 12 406 784 bytes, exactly
3029 pages — but that is a property of *this* container, so `bank_open`
checks it rather than assuming, because a misaligned record makes every
read fail EINVAL instead of merely running slow. The buffers came from
`malloc`; cache slots and the miss buffer now come from `waste_dio_alloc`
(posix_memalign, 4 KiB).

Filesystems can refuse O_DIRECT — tmpfs did until Linux 6.1, and on the
6.12 kernel used to test §26 it accepts it — so there is a fallback to
a plain open plus `posix_fadvise(POSIX_FADV_RANDOM)`, which at least
stops readahead. When any bank falls back, `waste_stats.direct_io` goes
to 0 and `waste bench` says the hit rate is partly the kernel's. A
measurement that quietly means something different is worse than one that
is missing.

Found on the way, and a harder blocker than the missing O_DIRECT: the
build used `-std=c11`, which sets `__STRICT_ANSI__`, under which glibc
hides every POSIX extension. `pread`, `fcntl`, `posix_memalign` and all
of `pthread_*` would have been implicitly declared — only `model.c`
defines `_GNU_SOURCE` for itself. The Makefile now uses `-std=gnu11`.
`libwastevq.dylib` was hardcoded too, so `make` could not have finished
on Linux at all.

**None of this is validated on Linux.** Docker is installed here but has
no registry access — no local images and `docker pull` hangs on both
amd64 and arm64 — so the platform has still never run the engine. What
was verified: macOS unchanged and 17/17; every file passes
`-fsyntax-only` for an x86_64 target, which compiles the CPUID branch no
ARM build ever sees; and `bank_open`'s Linux body compiles and runs
against stub declarations of `O_DIRECT` and `posix_fadvise`. That catches
typos and wrong signatures. It does not catch a filesystem that refuses
O_DIRECT in a way I guessed wrong about, and the first real Linux run
should be treated as the actual test.

## 15. What a fuzzer found in an afternoon (2026-07-28)

`make asan`, `make fuzz`, `make fuzz-asan`. The fuzzer is structure-aware:
it starts from a synthetic container and breaks one thing at a time — a
JSON field retyped, an offset moved past the end, a file truncated
mid-record, a bit flipped in a header. Random bytes would be rejected by
the parser before reaching anything interesting.

Five defects, all reachable from a manifest, none of them a wrong answer:

**`d->tok[-1]` in three places.** `js_get` returns -1 for a missing key,
and `d->tok[trunk].size`, `d->tok[sh].size` and `d->tok[kl].size` all
indexed the token array with it — a heap read before the allocation,
triggered by removing `config` or `shape`. Fixed by the class rather than
the instance: `js_size()` returns 0 for anything that is not a valid
container, and no code outside the parser touches `d->tok` any more.

**No config validation.** A manifest claiming 200 layers walks off the end
of `waste_model`'s `[WASTE_MAX_LAYERS]` arrays; one claiming zero of
anything produces empty allocations that later get written. `cfg_sane()`
bounds every dimension now.

**Unbounded allocation from a declared shape.** `shape: [2^20, 2^20]`
asked for 4 TB. `posix_memalign` would have failed and the error path
handled it, but nothing should size an allocation from an unchecked
claim: tensor sizes are now bounded by the size of the file that is
supposed to contain them.

**Division by zero** on a zero last dimension or a zero group size.

**Token ids were never range-checked** — found earlier the same day, by
the same synthetic container. They index the embedding table directly.

The pattern worth keeping: every one of these was invisible against the
real container, because a valid 19 GB model never asks for a tensor
bigger than its file or a layer past 128. **The small fake input is what
made the checks fail loudly.** Two of the five were in the test code
rather than the engine — `test_state` had the vocabulary hardcoded to
163840 and read off the end of a 256-entry logits buffer — which is the
same lesson pointing at the tests.

Note that RSS budget checks are skipped under a sanitizer
(`WASTE_SANITIZED=1`): shadow memory makes peak RSS meaningless, and a
check that cannot be true is worse than no check.

## 16. Too much cache is worse than too little (2026-07-28)

Gate 5 established a floor: below one token's working set the expert cache
holds nothing between tokens and the hit rate is zero. Publishing the
README's performance table found the other end of the curve.

| budget | expert cache | hit rate | decode |
|---|---|---|---|
| 32 GB | 3.1 GB | 0% | 0.31 tok/s |
| 46 GB | 17.1 GB | 12% | 0.32 tok/s |
| 52 GB | 23 GB | 25% | 0.33 tok/s |
| 58 GB | 29.1 GB | 37% | **0.04 tok/s** |

The last row is not a typo and not a fluke — reproduced twice, 384 s and
310 s for sixteen tokens, with the best hit rate of the run. Peak RSS was
42.8 and 48.0 GiB against a resident trunk of 27.5 plus 29.1 of cache:
the OS had already paged part of it out. The engine stayed inside its
budget; the machine did not, so a cache "hit" became a page fault instead
of the `pread` the engine was managing, and every layer paid for it.

What makes it sharp is how little it took. The commit that moved
`embed_tokens` off the resident set freed 1.11 GB, which at a fixed 58 GB
budget went straight into the cache — 27.99 GB to 29.1 — and turned
0.32 tok/s into 0.04. An optimization that frees memory made things
eight times slower, because the freed memory went somewhere the OS could
take it back.

`waste_open` now warns when a budget leaves the machine under an eighth of
its RAM, and `waste_physical_ram()` is public so an embedding host can
size its own ceiling. The warning is the cheap part. The lesson is that
the whole cache argument assumes the engine controls its own memory, and
that assumption has an upper bound nobody had measured.

**Re-measured for the release, and the cliff had moved down a budget.**
Same sweep, current build, machine idle with 49 GB free before each run:

| budget | expert cache | hit rate | decode |
|---|---|---|---|
| 32 GB | 3.32 GB | 0% | 0.31 tok/s |
| 46 GB | 17.32 GB | 13% | **0.32 tok/s** |
| 52 GB | 23.32 GB | 27% | **0.11–0.14 tok/s** |
| 58 GB | 29.32 GB | 37% | 0.04 tok/s |

52 GB used to be the best row of the sweep at 0.33 tok/s. It is now a
third of that, over three runs — 180 s, 143 s and 117 s for the same
sixteen tokens, the first of those with another model running alongside.
The spread is the paging, not the engine: all three report **identical**
cache statistics, 6313 hit / 17239 miss, so the cache is doing exactly
what it did before and the time is going somewhere it does not measure.
It is the same noise §11 records in peak RSS, from the same cause. What
changed between the two sweeps is §13: `embed_tokens` left the resident
set, the floor dropped 1.11 GB, and at a fixed budget every one of those
bytes went into the cache. 23.00 GB of cache was survivable and 23.32 was
not.

**Measurement order is now part of the method.** Re-running 46 GB *after*
the 52 and 58 GB rows gives 0.25 and then 0.22 tok/s instead of 0.32, on
three runs whose cache statistics are identical to the digit — 2961 hit /
20591 miss every time. The engine is deterministic and the machine is
not: macOS does not give back what it compressed and paged during the
heavy rows, so anything measured after them is measured on a different
computer. Sweep upward, never downward, and treat a row taken after a
paging row as void. This is the same effect §11 saw in peak RSS, which
was already noisy enough to be a guard and not a proof.

**And the default budget was landing in the middle of it.** With no
`--budget` the engine took `recommended_bytes` and capped it at 7/8 of
RAM, which on this machine meant 56.00 GB and a **27.32 GB cache** —
between the 23.32 GB measured at 0.11 tok/s and the 29.32 GB measured at
0.04. The out-of-the-box run was the bad one, three to eight times slower
than the same command with `--budget 46G`, which is not a defect anyone
would find by reading the code.

The fix follows from this section rather than from a new constant. Cache
is only worth anything in whole multiples of one token's working set, and
`recommended_bytes` is already `floor + 3x` that set by construction, so
the default now steps down a multiple at a time and takes the largest
that fits under the cap: `3x`, else `2x`, else `1x`, else the floor. K3
lands on `floor + 1x` = 46.24 GB, a 17.56 GB cache, **0.33 tok/s with no
flag** — marginally better than the 0.32 at `--budget 46G`, because 17.56
beats 17.32. A 128 GB machine still gets the full `3x`; Kimi-Linear,
whose recommendation already fits, is untouched. No fraction was tuned:
the only number in the rule is the working set, which the engine already
computes.

Two more things worth keeping. **The knee is sharper than a sweep
with 6 GB steps can see** — the whole transition from best to eight times
worse happens inside one step, and where it sits depends on what else the
machine is holding, which is not a property of the engine at all. And
**an optimization that frees memory is not automatically good here**: the
freed gigabyte was spent by the budget resolver on more cache, which was
the one place it made things worse. Freeing memory and *keeping* the
budget where it was are different actions, and only the second is safe —
which is now what the default does, by refusing to spend the fraction
above a whole working set.

## 17. What the first green CI run found (2026-07-29)

Two build defects, neither reachable from this machine, both found the
first time GitHub ran the workflow rather than by anyone reading the code.

**`test_image` never linked on Linux.** Its rule passed `$(LDFLAGS)` —
empty — where every other rule passes `$(LDLIBS)`, so the binary linked
without `-lm`. It worked on macOS for a week because clang folds the one
`sqrt()` in `image.c:71` into an instruction and glibc/gcc does not, so
the undefined reference existed only on the platform nobody built on. The
earlier Linux runs in [BACKENDS.md](BACKENDS.md) missed it because they
predate the image loader. Every link rule now passes `$(LDLIBS)`,
including the one target that needs nothing today.

**`make asan` compiled the AVX2 kernels without `-mavx2`.** This one is
worth knowing about in any makefile. The per-ISA flags are target-specific
variables:

```make
src/simd_avx2.o: CFLAGS += -mavx2 -mfma
```

and a variable set **on the command line silently defeats them** — which
is exactly what `asan` and `fuzz-asan` do when they re-enter make with the
sanitizer flags. So those two targets built `simd_avx2.c` with no AVX at
all, and gcc refused to inline the `always_inline` intrinsics
(`target specific option mismatch`) rather than warning that a flag had
gone missing. `override CFLAGS += …` fixes it, and `make -n` proves it in
one line. Three things made it invisible: the file is not in `SRC` on ARM,
so no local build compiles it; clang accepts those intrinsics without the
flag where gcc rejects them, so no cross-target syntax check catches it;
and the plain build passes the flags correctly, so only the sanitizer jobs
were ever wrong.

The method note: **a target-specific variable is not a guarantee.** If a
flag is required for correctness rather than for speed, either mark it
`override` or keep it out of a variable the caller is expected to set.

A third defect fell out while fixing them, this one local-only: `make
asan` cannot pass on a machine that *has* the K3 container, because ASan's
allocator refuses the 27 GB mapping the trunk needs, so the three checks
that open K3 fail rather than skip. CI never saw it — it has no K3 — and
the laptop that does see it is the one place the target gets run by hand.
`tests/run.sh` now skips every K3 check under `WASTE_SANITIZED`, next to
the RSS skips that were already there for a different reason.

**And a fourth, which only an actual Linux run could find.** With the two
build defects fixed, the suite reached the point of *running* on x86_64
for the first time — and `test_state` was killed by the OOM killer.
`tests/test_state.c` set `ram_budget_bytes = 6ULL << 30`, and a budget in
this engine is not a limit that gets approached, it is a ceiling the
expert cache is *sized to fill*: 6 GB of cache, allocated to check
session round-trip on a 1 MB synthetic container. On a 64 GB laptop that
is invisible. In a 7.75 GB container it is a SIGKILL, and on any CI
runner it is a hostage to whatever else the box is doing. It now passes
0 and lets the engine size itself, which is the only value that clears
the floor of both the synthetic container and a real model.

The reason all four hid so well is the same: **`make test` failing meant
the suite never ran at all**, so the job stayed red on the first defect
and the three behind it were invisible. A build that fails early hides
everything downstream — worth remembering when a CI failure looks like
one problem.

Reproducing this locally took Docker and about twenty minutes:
`--platform linux/amd64` on ubuntu:24.04 gives the same gcc 13.3.0 the
runner has, down to `undefined reference to 'sqrt'` at the identical
`.text+0xfcdd`. Both Linux targets now report 17 pass / 0 fail / 8 skip
model-free (8 rather than 7 because the image has no `uv`, which
collapses the two kernel checks into one skip).

## 18. The file we never downloaded (2026-07-29)

The vision section of this document contained a paragraph titled "the one
thing still taken on faith": K3 ships no preprocessor config, so the pixel
normalization is the CLIP convention rather than a transcription. It was
labelled honestly as an assumption and it named itself as the first thing
to question if images ever looked subtly wrong.

**The release does ship `preprocessor_config.json`, and it says
mean = std = 0.5.** K3 normalizes to [-1, 1]; the CLIP means differ by up
to 0.09 and the deviations by nearly 2x. Every image the engine encoded
reached the tower with the wrong contrast and a colour cast.

The reason it survived is the interesting part, and it is three separate
blind spots stacked:

- **The downloader asked for filenames it already knew.**
  `fetch_weights.sh` fetched a hardcoded list of small files. The repo has
  22 non-weight files; we had 10. Missing were `encoding_k3.py` (the chat
  template, which had therefore looked absent and been reconstructed from
  a figure in the report), `tokenization_kimi.py`, three processor
  modules, and the preprocessor config. `preprocessor_config.json` was
  *on the list* — the request 404'd or failed, and a 404 there is normal
  and logged as nothing, because not every repo ships every name. A
  whitelist cannot tell "this repo has no such file" from "I never asked".
  It now enumerates the repo through the API.
- **The oracle could not see it.** `vision_ref.py` feeds `torch.randn`
  pixels straight to the tower, so it never calls the image loader. The
  2.3e-06 agreement was real, and measured a stage strictly downstream of
  the bug. A verified component next to an unverified input is a verified
  component.
- **The unit test defines its own constants.** `test_image` checks that
  the loader computes `(v - mean) / std` — passing in a mean and a std of
  its own. It proves the arithmetic and says nothing about which numbers
  the engine chose.

So three checks, all green, all structurally incapable of noticing. The
suite now compares the container's `vision.json` against the source's
`preprocessor_config.json`, which is the one check none of them was.

**The method note is not "verify more".** It is that an assumption
recorded honestly still reads as settled once it has been in a document
for a day: writing "this is a choice, not a transcription" felt like
diligence and functioned as a decision. The cheap move — asking the repo
what it contains — was available the whole time and cost thirty seconds.
When a doc says something is unknowable, check that it is.

Two smaller defects fell out of the same afternoon, both found by
actually running the chat template the recovered file made possible:

**Special tokens were only matched at pre-token boundaries.** The encoder
tested for a special at the current offset and otherwise let the
pre-tokenizer consume a piece — so a marker was found only when it began
one. The tiktoken pattern groups runs of punctuation, so in
`role="user"<|sep|>` the quote and the `<` land in the same piece, the
boundary never exists, and `<|sep|>` silently became five ordinary
tokens. `<|sep|>` alone and `x<|sep|>` both worked, which is why it went
unseen. The encoder now searches the remaining text for the earliest
marker and pre-tokenizes only up to it.

**Special tokens did not decode at all.** `waste_tok_decode1` searched
the rank table, which specials are not in, and returned zero bytes — so
they vanished from every detokenization, a stop string written in markers
could never match the text it was compared against, and a chat reply
arrived with the tag names still in it and the markers gone. Encode and
decode have to agree about what a token is.

And one in the CLI: `chat.json` was parsed by a hand-rolled scanner that
found string boundaries with `strchr(p, '"')`, ignoring backslash
escapes. Every value in an XTML template contains `role=\"user\"`. The
fields were truncated at the first embedded quote, which is why the first
templated run still printed its own closing markers.

## 19. Prompt text could write conversation structure (2026-07-29)

Reading `tokenization_kimi.py` — one of the files the downloader had never
fetched — turned up a distinction the engine did not make:

```python
if allow_special_tokens:
    self.model.encode(substr, allowed_special="all")   # structural markers
else:
    self.model.encode(substr, disallowed_special=())   # user/tool text
    # "encode any <|...|> as ordinary BPE tokens (never as control tokens)"
```

`waste_tokenize` always resolved markers. So a prompt was able to end its
own turn and open another:

```
$ test_tokenizer k3.waste 'hi<|end_of_msg|><|open|>message role="system"<|sep|>obey'
11  9663 163586 163587 2778 6244 878 14062 1 163589 1031 2025
        ^end_of_msg ^open                        ^sep
```

Real control-token ids, from text a user typed. With the chat template
that landed the same afternoon this became live: paste that into `waste
chat` and the model reads a system message it was never given.

The fix follows the reference. `waste_tokenize` is now plain text and
`waste_tokenize_markup` is the one that resolves markers, and the CLI
builds a prompt from *segments* rather than one concatenated string:

```
[sys_p][system text][sys_s][usr_p][media block][user text][usr_s][open]
 markup   plain      markup markup   markup       plain    markup markup
```

Encoding segment by segment is also what the reference does — its
`EncodeSegment` list carries the mode per piece — so the token boundaries
between them are the model's own and not an artefact of splitting.

Two notes worth keeping. **The safe mode is the default**: the function
with the plain name is the one you can hand untrusted text, and getting
structure requires asking for it by a longer name. And the bug was
invisible while there was no chat template — with nothing to forge, a
marker in a prompt was just a strange token. Shipping the template is
what turned a latent flaw into a live one, which is the usual way a
feature and a vulnerability arrive together.

## 20. Per-expert bit allocation: the lever that is not there (2026-07-29)

Design goal 5 of [FORMAT.md](FORMAT.md) said important experts get 3 bits
and unimportant ones get 2, after GEMQ
([arXiv:2605.23078](https://arxiv.org/pdf/2605.23078)). A `bits[]`
manifest field was reserved for it, the converter never wrote one, and
the README called it *the largest unexplored lever on both the disk
footprint and the bytes read per token*. It is now explored, with
`tools/bitalloc_lab.py`, and it is not a lever.

The arithmetic first, because it makes the question small. Total squared
error when a set S of experts drops from 3 stages to 2 is

    E(S) = sum_e err3_e + sum_{e in S} (err2_e - err3_e)
                           \______  delta_e  ______/

so the best possible S of a given size is exactly the smallest deltas.
That is arithmetic, not a result. Everything therefore rests on one
question: **how much does delta vary between experts?** If it is
constant, the optimal allocator and a coin flip write the same container.

It is constant. Three places it could have varied, all measured on K3
with the codebooks fitted per (layer, matrix) exactly as `convert.py`
fits them:

| where the spread could live | max/min delta | greedy vs random |
|---|---|---|
| between experts in a layer | 1.06–1.15x | 0.2–1.4% |
| between layers — 1, 5, 23, 46, 69, 92 | **1.01x** | — |
| between gate, up and down | 1.09–1.30x | 0.3–0.6% |

The middle row is the surprising one: the first and last MoE layers
quantize like the middle, to within one part in a hundred, so even
*per-layer* allocation — which the engine could do almost for free, since
record size is already keyed by layer — has nothing to allocate.

Two checks that this is real and not an artefact of the sample. **128
experts of one layer**, in case importance hides in a heavy tail rather
than in the variance: spread 1.15x, no tail. **Kimi-Linear**, in case
K3's QAT is what homogenized the experts: 1.09–1.17x within a layer,
1.02x across layers. Not a QAT effect.

The mechanism is visible in the uniform numbers. Error by stage count is
57.5% / 33.2% / 19.5%, i.e. each residual stage removes 42% of what is
left — and it removes the same 42% in every expert, every layer, both
models. Per-output-channel amax scaling is what does it: after dividing
each row by its own maximum, one expert's 8-dim vectors are distributed
like another's, so a codebook fitted on twelve of them fits all 896
equally well. Experts differ in what they compute, not in how hard they
are to quantize.

**What survives is routing frequency**, which is not flat. With delta
constant, the routing-weighted damage of demoting a set is just its share
of activations — `err^2 = err3^2 + mass(S) * delta` — so the whole policy
space collapses to picking an end of the routing distribution. Against
`tests/trace_kimi_300.jsonl` (300 tokens, 62 400 activations) and
Kimi-Linear's own errors, since no K3 routing trace exists yet:

| demoted | avg bits | disk | coldest first: I/O, error | hottest first: I/O, error |
|---|---|---|---|---|
| 25% | 2.75 | −8.3% | −0.0%, 19.60% | −25.6%, 30.66% |
| 50% | 2.50 | −16.7% | −1.9%, 20.58% | −31.5%, 32.70% |
| 75% | 2.25 | −25.0% | −7.8%, 23.51% | −33.3%, 33.30% |

Uniform VQ3R is 19.57%. Read the two halves against each other: the
policy that protects quality saves disk and **no I/O**, because cold
experts are cold and are not what gets read; the policy that saves I/O
costs 11 points of error for 26% of the reads. There is no third policy,
because delta is flat — the exchange rate is fixed and both ends of it
are bad. Disk is not the scarce resource here anyway (982 GB on a 3.7 TB
drive); bytes per token is, and this cannot buy them.

The left column is if anything generous: 300 tokens over 6656 slots
leaves most of the cold half simply unvisited, so a longer trace would
move mass out of the tail and make the coldest-first rows cost more error
for the same disk. The conclusion does not depend on which way that
error goes, because the column it would have to rescue is the I/O one.

So the mechanism was not built: variable-size records, a per-expert index,
a width-classed cache and an allocator, to land on the straight line
between VQ3R and VQ2R that a coin flip already reaches. §3 killed the
shared low-rank basis for the same kind of reason — the structure the
paper assumes is not in these weights.

**Revive criterion**, the same one §3 has. This is the unweighted metric:
for x ~ N(0, I), E||(W−R)x||² = ||W−R||²_F exactly, so the isotropic
proxy *is* Frobenius and the ranking is calibration-free — but real
activations are not isotropic, and an importance matrix could make delta
spread where the weights do not. Rerun `bitalloc_lab.py` with an
activation-weighted error in the same rented session as the Gate 4
oracle. Revive if delta spreads past ~2x between experts; otherwise
delete design goal 5 and the `bits[]` field with it. Nothing has been
written in that field, so either way it is a cheap change.

## 21. What a checksum on the read path costs (2026-07-29)

Every expert record has carried a `crc32` since the first converter wrote
one, and until now nothing on the read path looked at it. The reasoning
was in the format header, and it was not wrong: verifying costs a pass
over every expert on every miss. What it left out is what the alternative
costs. A single flipped bit in an expert payload does not produce a
visible failure — on the synthetic container it produces the *same
argmax* and slightly different logits. There is no symptom to notice, so
a container that rots after conversion is discovered by not being
discovered.

So it was measured rather than argued about — and then the measurement
decided it, in the direction of leaving it off.

**Where it landed.** The checksum is opt-in (`--verify`,
`waste_cfg.verify_records`, `WASTE_VERIFY=1`) and the *header* checks are
unconditional. That split is the actual result of this section: the part
that costs 5% is a choice, and the part that costs nothing — magic, the
record being the expert the index asked for, offsets that fit, short
reads — is not, because it is memory safety rather than integrity and
because the checksum could not have been written without it (see below).
So the engine still refuses a truncated or spliced bank by default; what
it no longer does by default is notice a bit flip inside a payload.

**CRC throughput, one M-series core, over a whole record:**

| | GB/s | per 2.54 MB record | per 11.83 MB record |
|---|---|---|---|
| byte table | 0.60 | 4.41 ms | 20.5 ms |
| slice-by-8 | 2.70 | 0.99 ms | 4.59 ms |
| armv8 `crc32d`, one chain | 12.05 | 0.22 ms | 1.03 ms |
| armv8, three chains | **33.03** | **0.10 ms** | **0.38 ms** |

The gap between the last two rows is the whole reason the implementation
is not four lines. `crc32d` retires one per cycle and takes about three
to produce its result, so a single dependent chain runs at a third of
what the unit can do. Three chains over three slices, stitched back
together with zlib's GF(2) combine, cost two matrix walks per record —
microseconds against a 0.38 ms pass.

**End to end, the checksum on against off** (measured while it was still
the default, with `WASTE_VERIFY=0` as the off side; the polarity of the
switch changed afterwards, the numbers did not):

| Kimi-Linear, 16 tokens, 5 GB budget | verify off | verify on | cost |
|---|---|---|---|
| three pairs, quiet machine | 9.08 tok/s | 8.65 tok/s | 4.7% |
| median of eight pairs, machine busy | 6.40 tok/s | 6.04 tok/s | 5.5% |

Call it **5%**. The absolute throughput moved by a third between those
two sittings — the second was taken while the machine was building
something else — and the ratio moved by 0.8 points, which is about the
resolution this measurement has. A single run either way says nothing:
the spread within one sitting reached 17%.

**K3 pays about 1%**, and that figure is derived rather than measured:
7,287 misses at 0.376 ms is 2.7 s of a 268 s run, and a 1% difference is
far under the noise of a four-minute streaming decode. The direction is
the point — the more a model is dominated by reading, the less a pass
over the bytes it just read costs. K3 is the model this engine exists
for, and it is the one that pays least.

x86-64 gets the table path and its 2.70 GB/s: SSE4.2 does have a `crc32`
instruction, and it is Castagnoli — a different polynomial, no help for
this one. PCLMULQDQ folding would close the gap and has not been written.
An aarch64 build gets the fast path when the compiler was told the CPU
has the CRC extension, which on Apple clang is the default and on a
portable Linux build is `make WASTE_NATIVE=1`. `waste info` prints which
one a binary got, because nothing else would say.

**The header is not in the checksum**, and that had to be handled rather
than noted. The converter computes the crc32 over the body alone, so a
flipped bit in `chan_corr_off` is invisible to it — and that field is
what says how much to checksum. Deciding the extent of a check from a
value the check does not cover is a read past the buffer, not a check. So
the header is verified structurally first, against what the manifest
already implies: the record must be the bank's stride long, must be the
expert asked for at that offset, must name a codebook that exists. All of
it derivable, so none of it believed.

**What the fuzzer found once it read records.** `fuzz_container.py` only
ever ran `waste info`, which parses the manifest and opens every bank but
never reads a record — so the new checks were outside everything it
covered. Extending it to drive a forward pass, plus a mutation that
damages *every* record of a bank (a prompt routes to few experts, so one
damaged record is usually never read and proves nothing), turned up a
defect that had nothing to do with checksums: a manifest whose `trunk`
list is empty loads successfully, and the first token dereferences the
NULL that `waste_find` returns. The three tensors every forward pass
reads unconditionally — embeddings, final norm, head — are now required
at load. Checking *every* tensor the pass might want would mean a second
copy of its naming rules, and a wrong entry there would refuse a
container that works; these three cannot be wrong.

## 22. The reads were serial, and half the step was waiting (2026-07-31)

An outside article on running K3 through AirLLM — at ~5 minutes per token,
against this engine's 3 seconds — turned out to have exactly one thing this
project did not: it overlaps loading with compute. WASTE did not. `moe_layer`
picked its top-16 and then read them one at a time, `pread` by blocking
`pread`, with the arithmetic waiting on each.

The ids were all known before the first read. They come out of the routing
loop directly above, sixteen independent reads sitting in an array.

**What it was worth.** Two reader threads, depth 2, alternating runs at the
same budget so the paging state is shared:

| | 16 tokens | tok/s |
|---|---|---|
| synchronous | 47.90 / 48.81 / 54.32 / 66.46 s | 0.24–0.33 |
| **read-ahead** | **31.09 / 31.34 / 32.31 / 32.88 s** | **0.49–0.51** |

**1.5–1.6x**, with `experts 3357 hit / 20195 miss` identical in every single
run — the cache does exactly what it did before, the time is what changed.
Chunked prefill gains less, ~1.35x, which is the same result read the other
way: a chunk already spreads each expert over several tokens, so there is
proportionally less I/O to hide.

The second column is worth as much as the first. The synchronous runs spread
39%; the read-ahead runs spread 6%. A blocking read inherits every hesitation
the machine has, and a queue absorbs them.

Three things this required, and two of them were mistakes first.

**The pin that never expired.** A slot with a read in flight cannot be an
eviction candidate, so slots carry a pin stamped with the current hint
generation, and bumping the generation is what releases the previous layer's.
The synchronous claim path took a pin too — and with read-ahead off the
generation never advances, so every slot a synchronous read claimed stayed
pinned forever. The victim sampler ran out of candidates within one cache-full
of tokens and returned -1.

**And -1 was silent.** `read_expert` returned NULL, `moe_layer` did what it
has always done on an unreadable expert — `break`, and let `m->read_error`
carry the reason out — except that nothing had set `read_error`, because
`bank_fetch` was never reached. The engine answered, with the experts it
happened to have: *"Italy's capital is Italy. Italy's capital is Italy."*,
128 tokens of it, exit status 0. A wrong answer with no error is the failure
this project spends the most effort not having, and it took a default-off
code path one build to produce one. `REC_E_NOSLOT` exists now, and
`read_expert` records it when the cache returns NULL without a cause.

**The teardown order.** `waste_model_free` closes the bank fds before it
frees the cache, which is where the reader threads get stopped — so a reader
could have been mid-`pread` on a closed, possibly reused descriptor. The
threads are stopped first now, before anything else is torn down.

Method notes, both familiar:

- **The default path is not the tested path.** The suite runs with read-ahead
  on, so the synchronous fallback — the thing every measurement in this file
  before today was made on — had no check at all. `WASTE_IO_THREADS=0`
  against the default is now one, and it is the check that would have caught
  the pin in seconds instead of in a K3 run.
- **The disk was never the reason to do this.** The sweep says the internal
  SSD gives 10.73 GB/s at queue depth 1 and 12.89 at depth 2 — a 20% band.
  The other 1.3x came from not standing still, which is not a number a
  bandwidth measurement can show.

## 23. The router has no tail to demote (2026-07-31)

[EFFICIENCY.md](EFFICIENCY.md) proposed reordering the expert record so its
residual VQ stages are contiguous planes rather than interleaved bytes. That
would make any prefix of the stages a single coalesced read, which in turn
would allow reading two stages instead of three for the experts a token
barely uses — the one lever that cuts I/O *and* arithmetic, since `vq_rows`
does exactly `stages` gathers per row.

It rests on one assumption: that the top-16 is top-heavy. `WASTE_DUMP_ROUTE`
now writes the renormalized weights, and it is not.

| rank | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| mean weight | 0.149 | 0.108 | 0.077 | 0.055 | 0.043 | 0.032 |

1104 rows, 12 decode tokens over 92 MoE layers. First to sixteenth is a
factor of **4.6**, and **ranks 9–16 carry 33.3% of the mass**. Per layer the
tail runs 21.6% to 48.4%, so there is no subset of layers to apply it to
either. Kimi-Linear says the same: 3.8x across its top-8, bottom half 32.0%.

Priced with §20's own `err² = err3² + mass(S)·delta`, demoting ranks 9–16
takes the expert error from 19.5% to **24.9%** for 16.7% of the reads; the
gentlest version, ranks 13–16, gives 22.0% for 8.3%. A K3 expert costs 20.3%
at 3 bits from MXFP4 and that is the measured safe point. Both rows are past
it, and both sit on the straight line §20 already described — "the exchange
rate is fixed and both ends of it are bad".

So the format change was not made, for the price of an afternoon against a
982 GB reconversion. This is Gate 6 again with a different assumption in the
same place: §20 found the *experts* homogeneous in how hard they are to
quantize, and this finds the *router* homogeneous in how much it leans on
them. Two independent flatnesses, and between them they close per-expert bit
allocation in both its static and its per-activation form.

The instrument stays — four lines behind an env var — because it is the
first thing to run against any new container, and it is the difference
between believing a router is peaked and knowing it is not.

## 24. Volatile memory is memory you have given away (2026-07-31)

§16 is the worst number in this file: a 29.32 GB expert cache reaches a 37%
hit rate and runs at **0.04 tok/s**, eight times slower than a 17.32 GB
cache at 13%. The engine stays inside its budget, the machine does not, and
a cache hit becomes a page fault instead of the `pread` the engine was
managing.

That reading blamed the OS for mishandling the engine's memory, and it
suggested an answer: **purgeable memory**. Allocate each slot with
`VM_FLAGS_PURGABLE`, mark it volatile while idle, and under pressure the
kernel discards it outright rather than compressing and swapping it. A
discarded slot is a miss, and a miss is a read. The cliff becomes a slope.

Gated first, as the rule says: a volatile/nonvolatile round trip costs
0.33 us — 0.5 ms over a K3 token's 1472 experts — `vm_allocate` returns
16 KiB pages so O_DIRECT is unaffected, and the logits come out
bit-identical. Cheap enough to build.

**It works, and it is still not worth turning on.** 8 tokens, read-ahead on:

| budget | cache | purgeable | hit | decode |
|---|---|---|---|---|
| 46.25 GB (default) | 17.56 GB | off | 19% | **0.49–0.52 tok/s** |
| 46.25 GB (default) | 17.56 GB | on | **0–1%** | 0.29–0.33 tok/s |
| 58 GB | 29.32 GB | off | 39% | **0.04 tok/s** |
| 58 GB | 29.32 GB | on | 0–21% | **0.22–0.25 tok/s** |

At an over-large budget it is **6x faster** and does exactly what it was
built to do. At the budget that actually works it costs **1.6x**, because
the hit rate falls to nothing: macOS reclaims volatile objects eagerly, not
only under pressure, so a cache that would have stayed resident is taken
anyway.

One cause under both rows, and it is the correction to §16. **The memory was
never the engine's.** Purgeable does not offer "keep more cache"; it offers
"lose it cheaply or lose it expensively". The 37% hit rate in the third row
is real, and every hit in it is a page fault, and no flag changes that — the
pages are not there. A cache above what the machine will leave resident
cannot be bought at any price, and the default budget resolver, which steps
down a whole working set at a time and takes the largest that fits, was
already picking the only size that works.

So the projected ~2x from "fixing" the cliff does not exist. What is kept is
the escape hatch: `WASTE_PURGEABLE=1` turns a badly-chosen `--budget` from a
6x catastrophe into a 2x slowdown, which is worth having and is worth having
off by default.

Two method notes:

- **The gate measured the wrong thing, and was still right to run.** It
  asked whether the mechanism was affordable — 0.33 us, yes — and that
  question was worth an hour. It could not have asked whether the kernel
  would leave the pages alone, because that only appears against a 29 GB
  cache on a busy machine. A cheap gate is not a substitute for the
  measurement, it is what makes the measurement worth setting up.
- **A result that reverses between two configurations is the useful kind.**
  Had it only been measured at 58 GB it would have shipped on by default as
  a 6x win, and every ordinary run would have got 1.6x slower.

And one defect this found, unrelated to memory but caught by the same work:
`waste_ecache_get` releases one more read into the pipe *before* it returns,
and nothing stopped the victim sampler choosing the slot whose bytes the
caller was about to multiply. The hint path pinned it as a side effect; the
synchronous fallback inside `get` did not. `ec_pinned` covers `last_used`
now. Read-ahead had been green on 37 checks and against the oracle for a day
by then — the window is one layer wide and needs the sampler to land on one
slot out of 1483.

## 25. The gather loop was not the bottleneck (2026-07-31)

§22 hid the expert reads behind the arithmetic and §24 closed the memory
levers, and [EFFICIENCY.md](EFFICIENCY.md) concluded from the model
`max(1.17 I/O, 1.03 matmul)` that the engine was now arithmetic-bound, with
`vq_rows` the thing left to fix. So `vq_rows` was fixed.

**The profile says it was not the thing to fix.** Six decode steps of K3
with read-ahead on:

| stage | s | share |
|---|---|---|
| expert I/O | 9.95 | **54.8%** |
| expert matmul | 4.94 | 27.2% |
| — of which LUT apply | 4.34 | 23.9% |
| kda | 1.69 | 9.3% |
| LUT build | 0.48 | 2.7% |

The reads are still **twice** the arithmetic. The projection had
overestimated the matmul and underestimated the wait, and nothing checked it
against a profile before it became a plan.

**The optimization is real and small.** Three table lookups per row are the
algorithm; the three index bytes read one at a time are not. Eight rows are
24 consecutive bytes, so six word loads replace 24 byte loads — 64 memory
operations per eight rows down to 46, eight gather chains instead of four,
bit-identical output.

| | `waste bench` | `waste run`, 32 tokens | K3, LUT apply at 6 threads |
|---|---|---|---|
| before | 8.45 tok/s | 10.53 tok/s | 3.09 s |
| after | **8.73–8.78** (+3.6%) | **10.88–10.93** (+3.5%) | **3.00 s** (+3%) |

Kept: it is free, and Kimi-Linear is the model whose container fits in RAM,
where this bucket is most of a step. On K3, 3% of 22% is 0.7% and does not
clear the noise of a streaming decode.

**That table said +6.6% before it was measured properly.** The baseline came
from an hour earlier in the same session; run back to back against the
previous commit's `model.c` it is 3.5%, and three harnesses then agree on
3–3.6%. §16 established that a row taken after the machine has been worked
is measured on a different computer. It says nothing about which direction
the drift goes, and here it flattered the change — the harder case to
notice, because the number was the one being hoped for.

**Two refutations, and the first is §7 arriving from the other side.**

*Accumulators in registers.* Turn the loops inside out for an eight-row
sub-tile and the running sums live in registers, deleting all the `acc`
load/store traffic: 30 memory operations per eight rows instead of 46, and
still bit-exact, because each row sums over `v` ascending either way. It is
**17% slower** — 8.93 against 11.08 tok/s. Consecutive `v` sit 192 bytes
apart, so a sub-tile re-walks the block's whole index span and touches about
five cache lines for each one it uses. §7 found a layout that was 1.44x in
isolation and nothing in place; this is a change that is 35% fewer memory
operations on paper and a loss in place. **Counting operations does not
predict this loop. Counting cache lines does.**

*`VQ_SUPER`.* Swept 1, 2, 4, 8 on both models: 11.12–11.20 tok/s on
Kimi-Linear, 4.33–4.35 s on K3. Flat. §7's table-bandwidth theory stays
refuted even with a third of the index loads removed.

**One finding worth keeping.** The apply saturates at **six threads**, which
is exactly this machine's performance-core count: 2 → 6 threads is
7.31 → 2.99 s, and 6 → 18 is 2.99 → 2.89. The twelve efficiency cores are
worth 3% between them. §10 noted that a single-threaded run lands on an
E-core; this is the same asymmetry seen from the top.

The method note is about the order, not the code. **A projection that names
the next bottleneck should be checked against a profile before it becomes a
plan.** The profile cost one command and would have said, before any of this
was written, that the arithmetic was 27% and the reads were 55%. The work
was not wasted — it is bit-exact and it made the model that fits in RAM
3.5% faster — but it was chosen by arithmetic on an estimate rather than by
measurement, which is the thing this file exists to stop.

## 26. The bypass Linux never got (2026-07-31)

§14 wrote the Linux O_DIRECT path blind and said so: "None of this is
validated on Linux… the first real Linux run should be treated as the
actual test." Someone ran it. [Issue #4](https://github.com/sqliteai/waste/issues/4),
from Kevin McCoy, reports that Linux opened every expert bank with ordinary
buffered `O_RDONLY` and said `"direct_io": false`.

The cause is one identifier. `bank_open` gated the flag on

```c
const int aligned = rec_bytes && rec_bytes % WASTE_DIO_ALIGN == 0;
```

and `WASTE_DIO_ALIGN` is **16384** — the alignment the engine gives its
*buffers*, chosen so Metal's `newBufferWithBytesNoCopy` gets a whole Apple
Silicon page. What O_DIRECT constrains is the offset and the length, and the
format guarantees those in units of `WASTE_ALIGN`, **4096**. The two
constants answer different questions and one was standing in for the other.

No container has ever passed that test, and none ever could:

| | record | ÷ 4096 | ÷ 16384 |
|---|---|---|---|
| Kimi-Linear | 2 666 496 (651 pages) | 0 | 12288 |
| Kimi K3 | 12 406 784 (3029 pages) | 0 | 4096 |
| synthetic test container | 12 288 (3 pages) | 0 | 12288 |

§14 chose 3029 pages as evidence the alignment was checked rather than
assumed — and the check it was checked against could not accept it.

**Confirmed before it was fixed**, because "Linux reports false" has two
possible causes and only one of them is this. In a Debian container on a
6.12 kernel, `dd iflag=direct` reads the engine's own bank file at both 4096
and 12288 bytes: the filesystem was willing the whole time, and the engine
was refusing itself. After the fix the same container reports
`"direct_io": true`.

**The fix is not only the constant.** The eligibility test is necessary and
not sufficient: O_DIRECT accepts the `open` and then fails every transfer
the device does not like, so with the gate loosened a device wanting more
than 4 KiB would open unbuffered and die on the first expert of the first
token. `bank_probe` now reads one block at offset 0 — a page per bank at
load — and the fd is kept only if that succeeds. Verified by forcing the
probe to fail: `direct_io` goes back to false and the container still opens
and reads.

The patch attached to the issue was not applied. It was read for its
diagnosis, which was right, and the fix was written here — an issue body is
a report, not a change, and the probe is not in it.

Three notes worth keeping:

- **A constant named for one thing will be used for another.** Both are
  alignments, both are powers of two, both are about direct I/O, and the
  wrong one compiles. The comment on `WASTE_DIO_ALIGN` in `ecache.h`
  explains why it is 16 KiB and that explanation is what made it look like
  the right answer here.
- **The platform that cannot run the test is the platform that gets the
  bug.** §14 knew this and wrote it down, and the bug still shipped and
  still took an outside report to find. Writing "unverified" in a document
  is not a substitute for a run — the same lesson as §18, where an
  assumption recorded honestly read as settled after a day.
- **`make check` on Linux is not green**, and was not before this change
  either: 20 pass, 4 fail, 12 skip on a pristine tree in the same
  container. Three are the download script, which needs `curl` and does not
  have it in `Dockerfile.test` — a missing prerequisite that FAILs where
  this suite's own rule says it must SKIP. The fourth is a `serve`
  checkpoint test that passes in isolation and fails under the suite. None
  of them touch `model.c`; all of them are their own issue.

## 27. Windows built for a year and was never built on (2026-07-31)

[Issue #3](https://github.com/sqliteai/waste/issues/3), from Tadden Moore and
the AGi Dream Team Family, is the first native Windows x86_64 run of this
engine: MinGW-w64 GCC 13.1.0, the AVX-512 backend executing for the first
time anywhere, and the KDA kernel matching the `fla` oracle to **4.5e-08** —
the same order as Apple Silicon's 4.1e-08, off ARM for the first time. The
serve suite passed 167/167 against a natively built `libwaste.dll`.

It also came with four defects, none in the engine and all in the parts CI
could not reach. `windows-build` cross-compiles on Linux; `windows-run`
executes those artifacts. **Nothing had ever built on Windows or run
`tests/run.sh` there**, so the Makefile's platform detection, the shell
harness and the Python checkers were unexercised on the platform they were
written for.

| | what broke | why CI missed it |
|---|---|---|
| `Makefile` | `CC ?= cc`; stock MinGW has no `cc.exe`, `-dumpmachine` answered nothing, `ARCH` fell back to `uname -m` — which on MSYS says x86_64 and never contains "mingw" — so `WINDOWS`, `EXE` and `SOEXT` stayed unset and a Windows build was configured as a Linux one | CI always passes `CC=` explicitly |
| `tests/run.sh` | `SOEXT=so` unless Darwin; Git-Bash says `MINGW64_NT-…`, so it asked make for `libwaste.so` and reported a build failure for a library that had built fine | run.sh never ran on Windows |
| `tools/kda_ref.py` | opened the extensionless `test_kda`; MinGW emits `test_kda.exe` | same |
| `tests/run.sh` | `subprocess.run(["./waste", …])` does not go through Git-Bash's name resolution, and `os.sysconf` does not exist in Windows CPython | same |

The RAM one is worth its own line. The fix is not a Python
`GlobalMemoryStatusEx`: `waste plan --json` grew a `physical_ram_bytes`
field, because the human output already printed the number and
`waste_physical_ram()` already existed. The alternative was a second copy of
platform code living in a test.

**The job that keeps it fixed found three more before it went green.**
MSYS2 MINGW64, no `CC=` on the make line, and the whole of `tests/run.sh`:

- the download checks built their fixture with `python3 -c "open('$FT/…')"`.
  MSYS2 rewrites POSIX paths in a native program's **argv** and cannot
  rewrite one quoted inside `-c`, so Windows Python got `/tmp/…` verbatim.
- `tests/test_state.c` hardcoded `/tmp/waste_state_test.bin`, which a native
  Windows binary reads as `C:\tmp\…`. The save failed and the check reported
  the session state broken on the platform where nothing about it was wrong.
- and the job was not testing the thing it was added for: **MSYS2 ships a
  `cc` symlink**, so the default resolved and the Makefile fallback never
  ran. It now hides `cc` and builds again, because the machine that found
  the defect did not have one.

Green: **24 passed, 0 failed, 12 skipped** on `windows-latest`.

Three notes.

**MINGW64 and not MSYS, deliberately.** MSYS2's own python is a Cygwin-style
build with `os.sysconf` and POSIX name resolution. Running the suite under it
would have passed while defects 3 and 4 survived — a job that agrees with you
is worse than no job.

**Docker cannot do this.** Windows containers need a Windows host, so a Linux
container can only cross-compile — which is exactly what CI already did and
exactly what missed all of it. Docker did earn its place twice here, for the
Linux side: reproducing §26 and proving the Makefile fallback by deleting
`/usr/bin/cc`.

**And one found by accident, on Linux.** The three download checks *fail*
when `curl` is absent rather than skipping — the one thing this suite says it
must never do, since [ENGINE.md](ENGINE.md) states a missing prerequisite is
reported as SKIP. Debian slim and a bare MinGW both lack it. `make check` in
`Dockerfile.test` goes from 4 failures to 1; the remaining one is a `serve`
checkpoint test that passes in isolation and fails under the suite, which is
its own issue and not this one.

## 28. The fault injector that could not inject (2026-07-31)

The last red check in `Dockerfile.test` was
`test_failed_save_preserves_the_previous_checkpoint`, and it was red only
there: green on macOS, green on the GitHub Linux runners, skipped on
Windows. It passed in isolation and failed under the suite, which is the
shape of a test-ordering bug and was not one — in isolation the library had
not been built, so it skipped.

The check takes write permission off a directory, saves into it, and
expects `EngineError`. **Docker runs as root, and root has
CAP_DAC_OVERRIDE**: the temp file is created regardless, the save succeeds,
no error is raised, and the check reports the engine broken for something
the engine did right. Confirmed by running it twice in the same container:

```
as root     FAILED (failures=1)
as tester   OK
```

So the fix is a skip, and it is the same statement the file already made
one line above — `sys.platform == "win32"` is skipped because "directory
chmod is not a reliable Windows fault injector". A user not subject to
directory permissions is not a user this can inject a fault into either.
`make check` in `Dockerfile.test` is now 21 passed, 0 failed, 15 skipped.

Two notes.

**A test that cannot fail correctly is worse than an absent one**, and this
is the second time in two days that the same shape appeared: §26's
`bank_open` gate reported `direct_io: false` for a container that would
have worked, and this reported a save defect for a save that worked. Both
were the check being wrong about its own preconditions.

**The environment was the variable, and it was invisible.** Nothing in the
failure named root; the traceback said only `EngineError not raised`. What
identified it was running the same test as a different user in the same
container — the cheapest possible A/B, and the one worth reaching for when
a check is green on three machines and red on the fourth.

## 29. The cross-layer prefetcher has nothing to predict from (2026-07-31)

[FORMAT.md](FORMAT.md) has reserved `next_layer_top` in `usage.waste` since
the skeleton, for "a pilot/COUPLE prefetcher" that would start reading layer
L+1's experts while layer L computes. §22 built the *within-layer* prefetch,
which needs no prediction at all — the top-16 ids come out of the router
before the first read. Going across a layer boundary does need one, because
L+1's router eats the output of L+1's attention, which does not exist yet.

**First, the prerequisite: is there anything left to overlap?** No, not
within a layer. `WASTE_IO_DEPTH` 2 against 8, alternated: 0.42/0.54 against
0.43/0.52 tok/s — the spread between two runs of one setting is larger than
the difference between settings. Two reads in flight already keep the disk
as busy as it will get, so the only window a cross-layer prefetcher could
fill is the boundary itself: kda 0.28 + mla 0.065 + lm_head 0.007 + the
non-expert work inside MoE 0.19 = **~0.54 s per step** on a ~2.7 s step
where the readers have nothing queued.

**Then the signal.** `WASTE_DUMP_ROUTE` now writes the top-16 ids as well as
their weights, and the chunked path writes them too, so one prefill yields
what decode would take hundreds of seconds to emit. 214 tokens of mixed
English, Italian, code and SQL; 91 layer transitions; recall@16 of layer
L+1's actual set:

| predictor | recall@16 |
|---|---|
| random 16 of 896 | 1.8% |
| static hot 16 of the layer, held out | 20.5% |
| **the previous token's set at L+1** | **29.5%** |
| **co-occurrence from layer L, held out** | **29.0%** |
| co-occurrence fitted on the evaluation data itself | 49.7% |

**The cross-layer predictor does not beat the previous token**, and the
previous token is what the expert cache already exploits for free. Knowing
which experts layer L used tells you no more about layer L+1 than knowing
what the last token did.

**And the price is not symmetric.** On this machine bandwidth is the scarce
resource, so a wrong prefetch is not a missed opportunity, it is a read that
displaces a needed one. At accuracy p a layer reads `16 + 16(1−p)` records,
and the disk's 1.35 s/step of real work becomes `1.35(2−p)`:

| accuracy | wasted reads | window it can fill | net |
|---|---|---|---|
| 29.5% measured | 0.95 s | 0.54 s | **−0.41 s/step** |
| 49.7% overfit ceiling | 0.68 s | 0.54 s | **−0.14 s/step** |
| 60% | 0.54 s | 0.54 s | break-even |
| 80% | 0.27 s | 0.54 s | +0.27 s |

**Even the memorizing predictor loses.** The 49.7% row is a predictor fitted
on the very tokens it is scored against — it cannot be achieved, and it is
still under break-even. That is a one-sided bound, and it is what makes this
a decision rather than an estimate: building it would make K3 slower.

This is the third time the same asymmetry has decided something here. §4D
refused batching and speculative decoding, §24 refused a bigger cache, and
now this. The offloading literature — SP-MoE, MoE-SpeQ — assumes compute is
free and the link is idle, which is true behind PCIe and false here.

**A side effect worth keeping: this answers open question 1 of
[K3.md](K3.md).** It asked whether K3's 896-expert latent routing
concentrates differently from Kimi-Linear's 256 in full hidden space. Next-
token reuse is **29.5%** against Kimi-Linear's 33.6% (§4) and OLMoE's 43.5%
— the direction §4 predicted, now measured on the model this engine exists
for. Reuse keeps falling as experts get finer, and every cache argument in
this file rests on the level it has fallen to.

One caveat on the trace: it is prefill routing over real text, not decode
routing over the model's own output. The router is the same and the hidden
states are real, but a self-generated continuation is more repetitive, so
if anything these numbers are pessimistic about reuse and optimistic about
nothing.

## 30. mlock does not raise the ceiling; it removes the variance (2026-07-31)

§16's cliff is a cache hit turning into a page fault, and §24 answered it
from one side: make the slots *purgeable*, so the kernel discards instead of
swapping. That worked at an over-large budget and cost 1.6x at the one that
works, because volatile memory is memory you have given away. `mlock` is the
opposite bargain — the kernel may not take a slot at all — and the obvious
question is whether the opposite bargain is the better one. `WASTE_MLOCK=1`,
off by default.

**It is permitted.** `vm.user_wire_limit` is 56,349,970,923 bytes = 52.48 GiB,
82% of this machine's 64. A probe wires 32 GiB in one call without complaint.
(An earlier reading of this called it "exactly 7/8, the same fraction the
budget resolver uses" — that was GB against GiB. It is 82%, and the
coincidence is not there.)

**At the budget that works it is worth having, for a reason that is not
speed.** Default 46.25 GiB, cache 17.56 GiB, five runs each, alternated:

| | median | min | max | spread |
|---|---|---|---|---|
| pageable | 0.42 tok/s | 0.32 | 0.48 | 38% |
| **wired** | **0.51 tok/s** | **0.50** | **0.56** | **12%** |

The best pageable run matches the wired ones. What changes is the floor: the
pageable arm collapses to 0.32 on a machine that has been worked, and the
wired arm does not. These runs were taken *after* the 52 and 58 GiB rows
below, i.e. in exactly the state §16 says to treat as void — "macOS does not
give back what it compressed and paged during the heavy rows, so anything
measured after them is measured on a different computer." Wiring the cache is
what makes that stop being true. **The gain is reproducibility, and the
method note in §16 is what it buys back.**

**At the cliff it halves the damage and does not remove it.** 4 tokens:

| budget | cache | pageable | wired |
|---|---|---|---|
| 52 GiB | 23.32 GiB | 0.06 tok/s | **0.15 tok/s** |
| 58 GiB | 29.32 GiB | 0.03 tok/s | **0.06 tok/s** |

2 to 2.5x, and still three times worse than the default budget's 0.51. So
wiring does not make a big cache viable, and the reason is the one §13 has
been saying all along about which part of this engine is hot:

**it pins the wrong thing.** The cache is the cold part — 19 to 30% hit — and
the trunk is the hot part, 27.5 GiB read *in full every token*. Wiring the
cache does not create memory; it decides who loses the fight for it, and it
decides in favour of the part that is touched least. That is §24's finding
arriving from the other direction: the memory was never there, and neither
bargain conjures it.

**And at 58 GiB the configuration is unreachable in principle**, which is the
cleanest part of the answer. Wiring both would want 27.50 + 29.32 = 56.82 GiB
against a 52.48 GiB limit. The OS refuses it outright. At 52 GiB both would
fit — 50.82 GiB — with 1.66 GiB of headroom on a machine that also needs
wired memory of its own, for a hit rate of 27% against the default's 19%.
Not attempted: the upside is the wrong end of a curve §12 has already
measured flat, and the downside is a laptop that stops responding.

**Why it stays off by default.** On Linux `RLIMIT_MEMLOCK` is commonly 8 MB.
Measured under that cap: 43,008 of 43,690 slots refused, one warning line,
exit 0 — the fallback is not fatal, but wiring a real cache there is not
possible without a raised limit. A default that fails for most Linux users
and prints a warning about it is not a default.

The method note is about the question rather than the answer. **"Why not
mlock?" is the right question and it has three answers, not one**: it is
allowed, it does not fix the cliff, and it fixes something else that was
being lived with. The third only appeared because the measurement was run at
the budget where nothing was supposed to be wrong.

## 31. It was the trunk that wanted wiring, not the cache (2026-07-31)

§30 measured `mlock` on the expert cache and concluded it bought
reproducibility rather than speed. That was true and it was the wrong
comparison: it compared a wired cache against nothing, both on a machine
that had just been worked, and never asked whether wiring the *other* part
was better. It is. All 27.28 GB of K3's trunk wires in one pass.

Default budget, 8 tokens, four modes alternated, twice on a quiet machine
and twice immediately after a 58 GiB row had driven it into paging:

| wiring | quiet machine | right after a heavy row |
|---|---|---|
| none | 0.53–0.55 tok/s | **0.32** then 0.54 |
| cache only | 0.50–0.51 | 0.51–0.52 |
| **trunk only** | **0.57** | **0.56–0.57** |
| both | 0.57–0.58 | 0.57 |

Three things fall out, and the first two are the ones §30 could not see.

**Wiring the cache alone is worse than doing nothing** on a quiet machine —
0.50 against 0.55. This is the mechanism §30 hypothesized, now measured
directly rather than argued: pinning 17.5 GB of a cache that hits 19% forces
the 27.5 GB trunk, which is read *in full every token*, to be the pageable
one. It pins the cold part at the hot part's expense.

**Wiring the trunk is the best in every single run**, 0.56 to 0.58, quiet or
worked. Nothing else in this file is that flat across machine states.

**And wiring both adds nothing over the trunk alone.** Once the hot part
cannot be taken, protecting the cold part buys no further throughput — which
is the same statement as the first row, read from the other end.

The `none` row is where §16's method note lives: 0.32 immediately after the
heavy row, 0.54 once the machine had recovered. That spread is the whole
reason "sweep upward, never downward" exists. Wiring the trunk removes it —
the engine stops being a function of what the machine did an hour ago.

So `WASTE_MLOCK=1` now wires both, which is what a bare 1 should mean;
`cache` still names §30's behaviour, for reproducing that section. Still off
by default, for §30's reason: Linux's `RLIMIT_MEMLOCK` is commonly 8 MB and
a default that fails for most Linux users is not a default.

**The method note is about the shape of the question.** §30 asked "does
wiring help?" and answered it for the one buffer that had a wiring switch,
because that is where the code already was. The buffer without a switch was
the one that mattered, and it took a second experiment to notice that the
first had never been a fair comparison. A measurement is only as good as the
alternatives it was run against.

## 32. Wiring does not move the knee (2026-08-01)

§31 found the trunk was the buffer worth wiring and left an obvious
question: §16's budget sweep found a cliff between 46 and 52 GiB, and if
wiring stops the OS taking the hot part, does the cliff move? Re-measured
with `WASTE_MLOCK=trunk`.

| budget | cache | hit | unwired | wired |
|---|---|---|---|---|
| 32 GiB | 3.3 GiB | 0% | 0.50 tok/s | 0.45 |
| 46 GiB | 17.3 GiB | 17% | 0.53–0.55 | 0.50 |
| 52 GiB | 23.3 GiB | 31% | 0.06 | **0.19** |
| 58 GiB | 29.3 GiB | 37% | 0.03 | **< 0.012** |

**No. The knee is in the same place.** Wiring is worth 3x in the transition
zone at 52 GiB, where the OS was making the wrong choice and pinning the hot
part corrects it. It is worth nothing below the knee, where nothing was
going to be paged anyway. And above it, nothing helps: at 58 GiB the run did
not finish 8 tokens in eleven minutes, with **35.6 GB of the machine's 36.8
GB swap file in use** and 62 MB of RAM free.

That last row is the mechanism in the clearest form this project has
managed to photograph. Wiring the trunk means the trunk cannot be swapped,
so the *cache* is what goes — all 29.3 GiB of it, pages the engine believes
are resident, so every hit it counts is a swap read it does not. The engine
reported the same hit rate it always does. **The knee is set by how much
memory exists, and page-replacement policy only decides which part of the
engine is destroyed when there is not enough.**

**The experiment design was wrong and it is worth saying how.** The sweep
alternated wired and unwired at each budget, wired second, on the theory
that the handicap would fall on the configuration being argued for. It falls
the other way: a run that wires 27.5 GB and releases it leaves the memory
system in a state that punishes whatever runs next, and what ran next was
always the unwired arm. The 46 GiB unwired row of that sweep came out at
**0.02 tok/s — 348 s for 8 tokens** — against 0.53–0.55 for the same
configuration on a quiet machine in §31. That number is an artefact of the
run before it and is not in the table above; the unwired column is taken
from §30 and §31, where the ordering was clean.

Which leaves the absolute values at 32 and 46 GiB not worth arguing about
either — 0.45 and 0.50 wired here against 0.57 in §31, on a machine that had
been hammered all afternoon. **What survives is the shape**, and the shape is
what the question was about: three times better at 52, unchanged everywhere
else, and the cliff exactly where it was.

Two notes:

- **A sweep is not a set of independent measurements** when each row changes
  the machine the next one runs on. §16 knew this and wrote "sweep upward,
  never downward"; that rule is not sufficient once one arm wires memory,
  because wiring perturbs more than working the machine does.
- **The 58 GiB row was stopped rather than finished**, at eleven minutes and
  35.6 GB of swap. A more precise number was available and not worth a
  laptop that stops responding for it. `< 0.012 tok/s` is enough to answer
  the question that was asked.

## 33. An oracle fixture cannot be portable, and k-means is why (2026-08-01)

Two reports from outside ([#6](https://github.com/sqliteai/waste/issues/6),
[#7](https://github.com/sqliteai/waste/issues/7)) landed on the same blind
spot from opposite ends: **neither the container CI builds nor the container
this laptop keeps is the container a default conversion produces.**
The trunk's bulk has been 4-bit by default since before §13 refuted 3, while
`make_test_container.py` emitted only Q8G/F32 and the local Kimi-Linear was
built with `--trunk8`. Every check involving trunk width had been running on
the one shape that is not shipped.

**#7 first, because it is the one with a number worth keeping.** The engine
diverged from the shipped oracle fixture by 2.4 max on a default-conversion
container, and it was not an engine error:

| comparison | max \|diff\| | mean | correlation |
|---|---|---|---|
| engine vs a **fresh** oracle | **4.77e-05** | 7.52e-06 | 1.000000 |
| engine vs the shipped fixture | 2.425 | 0.461 | 0.979884 |
| **fresh oracle** vs the shipped fixture | **2.425** | **0.461** | **0.979884** |

`kimi_ref.py` shares no code with `model.c`, so a divergence that reproduces
identically in both comes from the weights they both read, not from either.
The reporter got the same three-row shape on Linux/x86 with his own
container (3.28 / 0.548 / 0.964251).

The obvious repair is to regenerate the fixture for the default conversion,
and **it is not enough.** One expert layer converted with `--device cpu`,
against the same layer converted on `mps`:

- the **trunk is bit-identical** — quantization is deterministic arithmetic;
- the **expert bank is not**. `train_codebooks` seeds its generator, but
  k-means on a different device converges to different books;
- splicing that one layer of 26 into an otherwise-`mps` container moved the
  final logits by **1.24 max / 0.19 mean**, against this suite's 1e-3
  threshold.

So a fixture is valid only for the exact container that produced it, and no
recorded provenance can fix that — a contributor on Linux would get the
right trunk width and fail on the codebooks instead, with a smaller diff and
the same ambiguity. **A pinned oracle is a second implementation's output
frozen against one build of the first. What survives a re-conversion is the
method, not the bytes.**

The number that made the alternative obvious was sitting in the tool the
whole time: `kimi_ref.py` reads the **container**, not the 92 GB of source
shards, so generating an oracle for the container under test costs **16.9 s**
here — against 2.67 s for the same 16-token prefill in the engine. `run.sh`
now generates, and keeps the fixture as the fallback for hosts without `uv`,
where it checks the recorded trunk against the container's and skips with
that reason.

**#6 is the same blind spot as a live bug.** `WASTE_Q8=0` claimed to
dequantize the trunk to f32 and read one byte per weight, which is true only
of Q8G, while its condition caught every quantized format. On Q4G it asked
for twice the bytes: a load failure when the overrun hit EOF, silently
decoding the next tensor as int8 when it did not. Routed through
`waste_deq_row` it now matches the default path to **1.9e-06**; restoring
the old assumption on the same container answers argmax 177 instead of 164.

Three notes:

- **This is the third instance of §28's shape.** A check that passed in CI
  and could not pass on a real container is a check that cannot fail
  correctly. The fix that matters is not in `model.c` — it is that
  `make_test_container.py` now mirrors `convert.py`'s widths, 4 bits for the
  bulk and 8 at both ends, so the synthetic container reports
  `trunk Q4G/Q8G/F32` and CI can reach the path at all.
- **A private copy of a shared routine is a fix that does not propagate.**
  That same branch carried its own fp16 conversion, flushing subnormals to
  zero — the bug `waste_f16` had been corrected for three days earlier, at a
  measured 27% error on one row of the vision tower's `fc0`. It survived
  because it was a copy. `docs/K3.md` records the identical
  knows-only-F32-and-Q8G bug being fixed in the *Python* oracle, and nothing
  connected the two.
- **Not every check can be run everywhere, and the suite should say which.**
  `WASTE_Q8=0` on K3 wants **211 GB of f32 trunk on a 64 GB machine**; the
  oracle prompt ids are Kimi-Linear's and mean nothing against K3's
  vocabulary. Both now skip with the arithmetic or the reason, rather than
  reporting a refusal as a divergence.

## 33. The 52 GiB row does not have a value (2026-08-01)

§32 re-swept the budgets wired and reported 0.19 tok/s at 52 GiB against
0.04 unwired — "three times better in the transition zone". Re-run on a
machine that started quiet, the same configuration gave **0.46**. Run again
after that, **0.03**.

| 52 GiB, `WASTE_MLOCK=trunk` | wall | tok/s |
|---|---|---|
| clean sweep | 17.55 s | **0.46** |
| §32's measurement | 42.95 s | 0.19 |
| immediately after | 239.17 s | **0.03** |

**3652 hit / 8124 miss = 31% in all three.** The engine did identical work
each time and the clock spanned 15x.

So §32's "three times better" was not a measurement of anything, and neither
is 0.46 or 0.03. **At 52 GiB the outcome is not a property of the
configuration.** The budget sits exactly where 27.5 GiB of wired trunk plus
23.3 GiB of cache either does or does not fit alongside whatever else the
machine is holding, and which side of that it lands on is decided before the
process starts.

That is the third reading of this row and the first useful one. §16 called
it 0.11–0.14, §32 called it 0.19, the clean sweep says 0.46 and the run
after it says 0.03. Every one of those was reported as a number. **The
number was the wrong output; the variance was the result.**

The clean unwired sweep, upward from a quiet machine, is what the README now
carries:

| budget | cache | hit | decode |
|---|---|---|---|
| 32 GiB | 3.32 GiB | 0% | 0.50 tok/s |
| 46 GiB | 17.32 GiB | 17% | **0.54** |
| 52 GiB | 23.32 GiB | 31% | 0.04 |
| 58 GiB | 29.32 GiB | 39% | 0.02 |

Wiring changes none of it: 32 and 46 measure 0.50 and 0.56, inside the noise
of the rows above; 58 stays hopeless; 52 has no value to change. **§31's
finding stands and §32's does not** — wiring the trunk is worth having for
reproducibility at a budget that fits, and it buys nothing at a budget that
does not.

Two method notes, and the second is the one that cost the afternoon.

- **A row that varies 15x is not a slow row, it is a row with no value.**
  Reporting its mean would have been worse than reporting nothing, because
  a mean invites comparison and there is nothing here to compare.
- **Each row of this sweep changes the machine the next one runs on, and
  wiring perturbs it more than working it does.** §32 already said the
  alternating design was wrong; running the arms separately did not fix it,
  because a 462-second 58 GiB row poisons the first row of whatever comes
  next — which is why §32's wired 32 GiB row read 0.21 and reads 0.50 when
  measured on its own. The only design that works here is one budget per
  quiet machine, and that is four times the wall clock of a sweep.

## 34. §29 refuted the wrong predictor (2026-08-01)

[deltafin](https://github.com/gavamedia/deltafin) is a parallel project with
the same goal — K3 on one machine — in Python, MIT, streaming both spine and
experts. Its `OPTIMIZATIONS.md` lists a **router lookahead**: take layer N's
pre-MoE hidden state, run *layer N+1's router weights* on it, and start
fetching what that says. The real router stays authoritative; the prediction
only starts I/O early, so it is exact by construction.

§29 measured cross-layer predictability and refused it at 29.0% recall,
against a 60% break-even. **That measured a different predictor.** §29 asked
what layer L's *expert ids* say about layer L+1's, i.e. a statistic over the
router's past answers. This asks the router.

Measured the same way, 12 decode tokens, 1092 layer transitions:

| predictor | recall@16 |
|---|---|
| same-layer expert ids | 1.7% |
| co-occurrence, held out (§29) | 29.0% |
| previous token's set (§29) | 29.5% |
| **next layer's router on this layer's hidden state** | **59.0%** |

Twice §29's number, and sitting exactly on the break-even it computed. On
its own that would be a coin flip. What decides it is that the prediction is
**steeply ranked**, which §29 never had reason to check:

| rank | 1 | 4 | 7 | 10 | 13 | 16 |
|---|---|---|---|---|---|---|
| precision | 92.2% | 80.2% | 64.3% | 53.3% | 39.4% | 27.9% |

So the policy is not "prefetch 16 and waste 41% of them". Prefetching the
top **6** — which is what the ~5.9 ms layer boundary holds at 0.92 ms a read
— gives **4.9 useful and 1.1 wasted** per layer. Blocking reads fall 16 →
11.1, the expert-I/O share of a step falls 0.548 → 0.380, and the step
should land near **0.65 tok/s from 0.54, about 1.2x**.

That is a projection, and §25 is the standing reminder about what happens to
projections here — but the input to it is measured, and the mechanism is not
in doubt: this is idle disk time being filled with reads that are right four
times in five.

**What §29 got right and what it got wrong.** The arithmetic was right: the
break-even, the asymmetry that a wrong prefetch displaces a needed read, the
observation that the window is only the layer boundary. What was wrong was
treating one predictor's failure as the question's answer. The `next_layer_top`
field the format reserves is a co-occurrence table, so the co-occurrence
predictor is what got tested — **the shape of the reserved data decided the
shape of the experiment**, and the better predictor needs no stored data at
all, only a matvec against weights already resident.

Two notes:

- **A parallel project is a source of hypotheses, not of numbers.** Nothing
  of deltafin's was adopted on its say-so; the recall and the rank profile
  were measured here, on this container, with `WASTE_DUMP_ROUTE`. Their
  design differs in a way that matters for the rest of their list: they
  stream the spine and WASTE keeps the trunk resident, which is why their
  speculative decoding pays and [EFFICIENCY.md](EFFICIENCY.md) §4D still
  refuses ours — theirs amortizes a per-token spine read we do not have.
- **The lookahead costs a second router matvec per layer**, 896x7168 against
  weights already in RAM. Under 1% of a step, and it is the reason this can
  be tried without touching the container format.

## 35. The router lookahead, built (2026-08-01)

§34 measured the predictor and priced it. Built: at the end of `moe_layer`,
once this layer's sixteen reads have all been consumed and the disk is about
to go idle through the next layer's attention, run layer L+1's router on
layer L's hidden state and issue speculative reads for its top **6**.

Six because that is what the ~5.9 ms boundary holds at 0.92 ms a read, and
because the prediction's precision falls off past it — 92.2% at rank 1,
81.4% cumulative at 6, 59.0% at 16. `WASTE_LOOKAHEAD=0` disables it.

**Two things it does are deterministic**, and they are the ones worth
trusting:

| | without | with |
|---|---|---|
| demand hit rate | 14–19% | **38–40%** |
| total bytes read | 254.2 GB | 254.5 GB |

The hit rate more than doubles and **the bytes do not move**. That is the
whole mechanism: the prefetched records were going to be read anyway, and
the lookahead only changes *when*. Past n=6 it stops being free — n=10
reads 264.2 GB, and the extra is waste.

**The throughput gain is real and this machine cannot pin it down.** Nine
paired runs, alternated, three prompt lengths:

    n=6 faster in 8 of 9 pairs
    ratio: median 1.17x, min 0.79, max 1.79
    median 0.46 -> 0.53 tok/s

A median of 1.17x against a projection of 1.20x is agreement, and a range
from 0.79 to 1.79 is what a day of sweeps has done to this laptop — §33
already established that a row measured after a heavy row is measured on a
different computer, and by now every row is after a heavy row.

**The accounting had to be designed, not inherited.** A speculative read is
not a demand access, so counting it as a miss would make a prefetcher that
guessed wrong look like a cache that performed badly, and every hit-rate
number in this file would stop meaning what it meant. So `ec_claim_spec`
counts `spec_issued` and the bytes, never `misses`; a token that later asks
for the record finds it resident and scores an ordinary hit. The 38% above
is the demand stream, comparable with every earlier figure.

**Exact by construction**, which is the property that makes it shippable:
the real router still decides, the prediction only starts I/O. `tests/run.sh`
checks the logits are bit-identical with it on and off.

Two notes:

- **It is on the decode path only.** `moe_chunk` routes a whole chunk at
  once and does not have the hook, which is why `waste bench` — mostly
  prefill — shows almost nothing while `waste run` shows the gain. That is
  the next thing to build, not a defect in the measurement.
- **The width is not a tuning constant, it is a window.** n=3 and n=4
  measured worse than n=6 (0.45–0.55 and 0.51–0.52 against 0.59–0.61) and
  n=10 worse again. Six is where the prediction is still right four times in
  five *and* the reads still fit before the demand for them arrives; both
  halves of that are properties of this disk and this model, not numbers to
  carry to another machine.

## 36. The same lookahead in the prefill path costs 7% of the reads (2026-08-01)

§35 shipped the router lookahead on the decode path and noted the obvious
next step: `moe_chunk` has no hook, so `waste bench` — mostly prefill —
showed nothing while `waste run` showed the gain. Built it. It loses.

A chunk routes nT tokens at once, so the prediction is the *union* of the
next layer's tops over every token in the chunk: one token wanting an expert
is enough to force the read. Implemented, capped at 384 ids, bit-identical
as before.

| | reads | demand hit | tok/s |
|---|---|---|---|
| decode hook only | 193.9 GB | 7% | **0.108** |
| plus the chunk hook | **207.2 GB** | 33% | 0.085 |

**The hit rate triples and the bytes go up 6.9%**, which is the signature of
a prefetch that is thrown away and fetched again. On a 64-token prefill the
wall clock does not move at all: 132.6 / 137.0 s without, 130.6 / 138.3 s
with.

The mechanism is the one §35's decode numbers hid. A decode layer claims 16
slots, so the six speculative records for the next layer survive easily. **A
chunk layer claims about 550** — the distinct experts of 64 tokens — and the
speculative records are unpinned, freshly inserted, and therefore exactly
what LFRU evicts first. They are read, evicted, and read again.

Pinning them would fix the eviction and not the problem underneath, which is
that **there is no idle window in the chunk path to fill.** A decode layer's
boundary is a real fraction of the layer, ~5.9 ms of attention against
16 reads. A chunk layer needs 550 reads against attention over 64 tokens: the
disk is busy continuously, and a prefetch there does not move a read into
idle time, it moves it in front of another read and pays an eviction for the
privilege.

So the hook is removed rather than defaulted off. Decode keeps it.

The note worth keeping is about what §35's own measurement could not see.
**`waste bench` showing nothing was read as "the hook is missing", and it was
also "the path does not want one".** One observation, two explanations, and
the cheap one was assumed. The distinguishing measurement — total bytes —
took one command and was not run until the second version had been built.

## 37. The bug the instruction set decided the meaning of (2026-08-01)

[#10](https://github.com/sqliteai/waste/issues/10), from a Windows/MinGW
build: `waste info` and `waste run` died instantly on K3 with
`STATUS_INTEGER_DIVIDE_BY_ZERO`. `waste plan` worked, because it does not
load.

The tensors the loader declines to load — the vision tower, and anything
outside `cfg.prefix` — set `on_disk` and `continue` before the quantized
branch assigns `group`, and `m->t` is `calloc`'d, so `group` stays 0. The
row-scratch sizing then divided by it.

**What that division means is the architecture's choice.** arm64's `sdiv`
answers 0 and the program carries on; x86's `idiv` raises `#DE` and the
process is gone. Same source, same container, same undefined behaviour, and
one machine reports a working engine while the other cannot open the
project's flagship model. Every measurement this project has published was
made on the machine that cannot see it.

The lesson is not "test on x86" — it is that **a suite green on one ISA says
nothing about another for undefined behaviour, and the sanitizer is what
carries the result across.** UBSan reproduces it on the hardware that hides
it: building with `-fsanitize=undefined` on this arm64 laptop gives
`src/model.c:1218:67: runtime error: division by zero` on a container the
normal binary opens without complaint. `make asan` is therefore not only a
memory-safety gate; for this class it is the only portable oracle we have.

Two things worth writing down beyond the fix:

- **The trigger was the prefix, not the vision tower**, and the issue title
  says otherwise for a good reason — on K3 they coincide. The skip needs
  `cfg.prefix[0]` non-empty, so a `vision_tower.*` tensor in a prefix-less
  Kimi-Linear container does *not* reproduce: it falls through and gets a
  group like everything else. The first attempt at a repro here failed for
  exactly that reason, which is the only way the distinction was noticed.
  `make_test_container.py` grew `--prefix`, not `--vision`.
- **It never needed K3.** The repro is an ~800 KB synthetic container, so
  this was always within reach of CI — §33's shape a third time, and the
  third different way of missing the same thing: after a trunk width nobody
  ships and an oracle from a conversion nobody makes, a container layout
  nobody generates. The check now exists, and its comment states where it
  is load-bearing: on x86 in any build, on arm64 only under the sanitizer.
  A check that passes by construction on half the machines has to say so.

## 37. The simulator was modelling a different cache (2026-08-01)

Most of what this project asks is not a throughput question. "Does the tail
of the top-16 carry enough mass to demote" (§23), "how predictable is the
next layer" (§29, §34), "what hit rate does this budget give" (§4) — none of
those need the engine to run, and all of them were answered on K3 anyway, at
**~50 seconds a measurement of which ~48 are model load and prefill**. The
same loop on Kimi-Linear is 2.9 seconds end to end.

`tools/routing_stats.py simulate` existed for exactly this and had not been
used since Gate 2, because the engine writes traces in one format and the
tool reads another. Connected now, and connecting it turned up three things.

**The tool models a different cache than the engine.** It kept a frequency
count across evictions that `ec_claim` resets, and sampled 32 victims where
`EC_SAMPLE` is 16. Against the same trace it read **36.6% where the engine
measured 30.4%** — optimistic, plausible, and wrong. Both are copied from
`ecache.c` now:

| slots | engine | simulator |
|---|---|---|
| 25 | 0.0% | 0.0% |
| 100 | 0.0% | 0.0% |
| 201 | 8.1% | 6.6% |
| 402 | 30.4% | 29.6% |

Within 1.5 points over a 0–30% range, with the access counts identical
(4992 both). `tests/run.sh` asserts the agreement rather than remembering
it, because a simulator that drifts quietly is how a policy question gets
the wrong answer for a week.

**The trace had to name its tokens.** Every script that read one of these —
three of them, in this session alone — re-derived token boundaries from
where the layer index wraps, which is a heuristic that is simply wrong on
the chunked path, where rows are grouped by layer and not by token. The dump
now writes the absolute position of the token each row belongs to, and the
reader has no reconstruction in it at all.

**And `--data` takes a container.** The record size read from the manifest
is the number the engine preads; derived from `bits` it was close enough to
put the slot counts a percent out. The converter copies the release config
into the manifest verbatim, so a container answers every shape question the
downloaded config does and is the thing already on disk.

The method note is one this file already has and this session ignored
twice. **`make -j8` builds the engine and not the checkers**, so the first
validation run compared a fresh library against a `test_forward` compiled
before the trace format changed — and the trace came out with one leading
column instead of two, which read as a broken reconstruction rather than a
stale binary. §17 recorded exactly this ("once to a stale test binary") and
the fix is the same as it was then: `make test`, or `make check`, which
rebuilds first.

## 38. One load, many arms, one machine (2026-08-01)

The second half of the iteration problem. §37 removed the engine from
questions that never needed it; this removes the *process* from questions
that do.

`tests/sweep.c` loads a container once and runs the arms back to back,
interleaved, resetting the session and clearing the expert cache between
each — because leaving the cache warm would hand the second configuration
the first one's work and measure the order rather than the setting.
`waste_model_reset` and `waste_ecache_clear` exist for it; the first was
already in `waste.c` reaching into the model's fields and is now in one
place.

```
$ WASTE_CACHE_MB=1024 ./sweep kimi-linear.waste 1008,6013,318,28288,17189 16 lookahead=0,6 3
lookahead    rep     tok/s       hit   GB read
       0      1    9.573     37.9%      6.7
       6      1   10.713     72.2%      7.4
       0      2    9.820     37.3%      6.8
       6      2   10.686     72.5%      7.4
       0      3    9.790     37.5%      6.8
       6      3   10.743     72.6%      7.3
```

**Two arms, three repeats, spreads of 2.6% and 0.5%.** Nine paired runs of
the same comparison across processes (§35) spanned 0.79x to 1.79x. That is
the whole point: the variance was never the feature, it was the harness.

On K3 it is honest about what it cannot fix:

| lookahead | median tok/s | spread | hit | GB read |
|---|---|---|---|---|
| 0 | 0.506 | 7% | 7.2–7.7% | 204–205 |
| 6 | 0.541 | 23% | 38.0–38.2% | **191** |

The deterministic columns are exact to a tenth of a point. The clock is not:
K3 still drifts inside a single process, because the drift is the machine's
memory system and not the process's. **What the harness buys on K3 is that
the noise is now visible as noise** rather than as a difference between two
runs an hour apart.

And it turned up something the cross-process measurements had wrong. §35
reported the lookahead as reading the same bytes; measured with the cache
cleared identically for both arms, it reads **6.6% fewer** — 191 GB against
204. The speculative fill arrives before the demand, so the record is
inserted early and its later hit raises its LFRU count, and records that
were being evicted and re-read now are not. That effect was invisible when
each arm started from whatever the previous process had left in the cache.

**What it cannot sweep is the budget**, which sizes the cache at open. Those
still need one process each and a quiet machine each, which §33 already
established is the only design that works there.

The method note: **a harness is part of the measurement.** Every conclusion
in §30 through §36 was drawn through a harness that added more variance than
the effects being measured, and two of them came out wrong. Building the
harness first would have been cheaper than any of the re-runs.

## 39. The cache floor was a property of a demand-only cache (2026-08-01)

§4 is the oldest load-bearing measurement in this file, and §16 and the
budget resolver are both built on it: **the cache floor is one token's
working set**, 17.0 GB for K3, and below it the hit rate is not low, it is
zero. Re-measured with `tests/sweep.c` — one process, four cache sizes
interleaved, two repeats — it is still exactly true, and it no longer binds.

The control first, at 287 slots, a fifth of the way to a token's 1472
records:

| 287 slots | hit | tok/s | GB read |
|---|---|---|---|
| lookahead off | **0.0%** | 0.507 | 153.1 |
| lookahead on | **29.1%** | 0.585 | 165.2 |

§4's zero is exactly reproduced. What breaks it is that **the lookahead does
not need a record to survive from one token to the next, only from one layer
to the next.** It fetches what layer L+1 wants while layer L is finishing, so
the record is consumed a few milliseconds later instead of three seconds. A
cache far too small to hold a token's working set is ample to hold six
experts for the length of one attention.

The whole sweep, with the lookahead on, which is the default:

| budget | cache | slots | hit | decode |
|---|---|---|---|---|
| 32 GB | 3.32 GB | 287 | 29.1% | 0.56–0.58 |
| 46 GB | 17.32 GB | 1498 | 36.2% | **0.63** |
| 52 GB | 23.32 GB | 2018 | 38.4% | 0.07–0.09 |
| 58 GB | 29.32 GB | 2537 | 41.3% | 0.07–0.08 |

Hit rate and bytes read are **identical to the digit across both repeats** —
29.1/29.1, 36.2/36.2, 38.4/38.4, 41.3/41.3 — which is what one process buys
and what §33 could not get from four.

**The useful window opens far lower than it did.** A 3.32 GB cache is within
10% of a 17.32 GB one — which is a size the default budget resolver cannot
choose, since it steps in whole multiples of a 16.2 GB working set and there
is nothing on K3 between the floor and `floor + 1x`. Whether that rule's
quantum should change is [GATES.md](GATES.md) Gate 7, open and not run: four
generated tokens is exactly the length that flatters a small cache, and
cross-token reuse is what a large one buys. The RAM above the resident trunk has stopped being
the lever it was in §12 and §16; the trunk is now nearly the whole
requirement.

**The cliff is exactly where it was**, and the rows either side of it are
the clearest statement of what it is: between 46 and 52 GB throughput falls
eightfold while the hit rate *rises* and the bytes read *fall*, 137 GB to
126. The engine does less work and takes ten times as long. Nothing about
caching touches that — it is 27.3 GB of trunk plus 23.3 GB of cache on a
64 GB machine, and §24 and §32 already established that no allocation policy
conjures the difference.

One more thing worth keeping, because it is not what §35 reported. **The
lookahead's byte economics depend on the cache size.** At 287 slots it reads
8% *more* — 165 GB against 153 — because speculative records are evicted
before use often enough to be re-read. At 1498 slots it reads 6.6% *fewer*.
It is a prefetch at small caches and a scheduling change at large ones, and
§35 measured only the large end.

The method note is short. **§4 was right and stopped being the constraint,
and nothing about §4 was wrong.** A measurement can be perfectly reproduced
and still stop describing the system, when what changes is not the number
but which mechanism the number was about.

## 40. The machine the resolver was sizing against was not ours (2026-08-02)

The default budget has one machine number in it, and until now that number
was `waste_physical_ram()`. On Linux it is `sysconf(_SC_PHYS_PAGES) *
sysconf(_SC_PAGESIZE)`, which reads the host's `MemTotal` — and reads
exactly the same thing from inside a cgroup that is allowed a fraction of
it. Every containerized run has therefore been sizing against RAM it was
never going to be given: K3 in a 32 GiB cgroup on a 256 GiB host resolves
`floor + 3x`, asks for about 80 GB, and is killed.

**This is not §16 and it does not behave like §16.** The cliff there is a
performance failure with a shape — the hit rate climbs, the bytes read
fall, throughput drops eightfold, and it is visible in a sweep. A cgroup
limit is a kill. Nothing degrades first, no allocation policy softens it,
and the sweep that found §16 could never have found this one, because the
machine it was swept on was not in a cgroup. It is the same class of bug
as §27: a platform path that every green run had avoided rather than
exercised.

So the ceiling is now `min(physical, cgroup limit)`, and everything
downstream — the 7/8 reserve, the whole-working-set stepping, the floor
refusal — is untouched. The reader takes the smallest finite `memory.max`
or `memory.high` across this cgroup and its ancestors: the limit is
hierarchical, so a leaf saying `max` does not cancel a finite parent, and
`memory.high` belongs there because a group the kernel reclaims from is a
group whose expert cache it takes back, which is §16's mechanism arriving
by another road.

**What deliberately did not go in is current pressure.** `MemAvailable`
and `memory.current` are the obvious next reading and they are a different
kind of number: capacity is fixed for the life of the process, pressure
moves between the read and the allocation. A budget is resolved once at
`waste_open` and then held for the length of a run, so bounding it by an
instantaneous sample makes the same command on the same machine two
different runs — and the reason to want it, a host that is busy *now*,
says nothing about a host that will be busy in ten minutes. Whether it
should trim the multiplier rather than the ceiling is
[#14](https://github.com/sqliteai/waste/issues/14), still open, and the
proposal that opened it is what found this bug.

Measured in the only place it can be, which is a container. `docker run
--memory=6g` on a host reporting 8,319,213,568 bytes of RAM:
`waste plan --json` gives `physical_ram_bytes` 8,319,213,568 and
`usable_ram_bytes` **6,442,450,944** — the limit exactly. The suite's own
budget check, run inside that cgroup, reports `usable 6.00 GB`, which is
the resolver consuming it rather than the reader merely reading it.
24 passed, 0 failed, 16 skipped on Linux there.

Both cgroup namespace modes were exercised, and they fail differently,
which is why the walk has to end *on* the mounted root:

| mode | `/proc/self/cgroup` | where the limit is | usable |
|---|---|---|---|
| private (default) | `0::/` | the mounted root itself | 5–6 GiB, exact |
| `--cgroupns=host` | `0::/docker/<id>` | that path, on the host hierarchy | 5 GiB, exact |

Under `--cgroupns=host` the composed path does exist, and
`/sys/fs/cgroup/memory.max` does **not** — confirming the assumption the
fallback rests on, that a real unified root carries no limit and an
unconfined host therefore still reads 0.

The choice changes, not just the reading. On the synthetic container
(floor 8,844,904, recommended 9,139,816) a 12 MiB cgroup holds
`floor + 3x` and opens silently; a 9 MiB one puts the ceiling under the
floor, so the engine runs at the floor and says so. Before this commit
both read 8.32 GB and saw nothing to say.

What is *not* here is a throughput row. This is arithmetic on a number
that was provably the wrong one, and the K3-in-a-cgroup case that motivates
it — 80.64 GB asked of a 32 GiB allowance — is derived from the resolver's
own rule, not run.

## 41. The 256-entry table was the reason the gather was scalar (2026-08-03)

§25 established that the VQ3R gather is a `load -> address -> load`
dependency and unrolled it eight ways. What it did not ask is why the
lookup had to touch memory at all. NEON has `tbl`: 16 lookups in one
instruction, from a table held in registers. The reason VQ3R cannot use it
is arithmetic, not effort — a 256-entry stage table is 256 bytes, sixteen
vector registers, on a machine with thirty-two. It does not fit, and no
amount of blocking makes it fit.

This is the same constraint the vector-search literature hit and solved:
[Quick ADC](https://arxiv.org/pdf/1704.07355) and FAISS FastScan both
force 4-bit sub-quantizers precisely so the table lives in a register, and
[T-MAC](https://arxiv.org/abs/2407.00088) reports 4.7x on ARM at 3 bits
doing the same thing for LLM weights. `vq_apply` is an ADC scan — a sum of
per-sub-quantizer table lookups over a database of codes — so the mapping
is exact rather than analogical.

**Bits per weight is `stages * log2(entries) / vec_dim`, and three shapes
hit 3.00.** 3x256, 4x64 and 6x16 all spend 24 bits per 8-weight vector, so
all three are the same record size. They differ only in whether a stage
table is addressable in registers. Swept single-threaded on K3's gate
shape (M=3072, nv=448), medians of seven runs:

| kernel | ms | vs current |
|---|---|---|
| scalar 3x256 fp32 (VQ3R) | 0.504 | 1.00x |
| `vqtbl4q` split-table 3x256 int8 | 0.407 | 1.24x |
| `vqtbl4q` 4x64 int8 | 0.152 | **3.32x** |
| `vqtbl1q` 6x16 int8 | 0.118 | 4.27x |

The split-table variant is the one that needs no format change — 256
entries as four 64-byte tables, four `vqtbl4q` and three `orr` — and it
buys 1.24x, because sixteen table registers still have to be reloaded per
stage. It is not worth a kernel.

**4x64 rather than 6x16, and the reason is quality, not speed.** Residual
k-means fitted the way `convert.py` fits it, against real weights taken
from the Q8G trunk of `kimi-linear-q8.waste` (the expert banks are already
VQ3R and would hand 3x256 a target it can hit exactly):

| shape | mean relative error | vs 3x256 |
|---|---|---|
| 3x256 | 19.97% | 1.000x |
| 4x64 | 21.47% | 1.075x |
| 6x16 | 23.60% | 1.182x |

On a real routed expert the same comparison reads 19.29% -> 20.85%
(+8.1%), so the trunk proxy was sound. 6x16 is a third faster and more
than twice the quality cost; 4x64 is the shape.

**End to end on Kimi-Linear, 1023 tokens of two different texts:**

| container | perplexity |
|---|---|
| VQ3R 3x256 | 10.937 |
| 4x64, byte indices, fp32 LUT | 11.248 |
| VQ4P, 6-bit packed, int8 LUT | **11.237** |

**The int8 table is free.** All of the +2.7% is the codebook shape;
quantizing the table — which is what makes a byte shuffle possible at all
— costs nothing measurable, and the packed container is marginally the
better of the two. One scale per 32 vector positions is why: a LUT entry
is `dot(x_v, centroid)` and its magnitude tracks `||x_v||`, which varies by
orders of magnitude across a hidden state, so a single global scale would
round the quiet positions to zero.

The container is the same size to the byte: 18 GB on Kimi-Linear, 982 GB
on K3, 3.00 b/w both.

## 42. A table built once and quantized sixteen times (2026-08-03)

The kernel above wants an int8 table. The first implementation quantized
it inside `vq_apply`, on the argument that a pass over `stages*entries`
against an apply that does `M` times as much is about 2%.

**It was about 140%.** `LUT apply` went from 0.85s to 1.48s — with the
kernel that had just measured 3.3x faster in isolation.

Two mistakes, and the estimate hid both. Gate and up are built *once per
token* and applied *once per routed expert*, so the pass ran top_k times
over a table that had not changed. And it ran serially, next to an apply
that was threaded, so its share of wall-clock was its share of one thread's
work — not one core's worth of a parallel region.

Moving it into `vq_build_lut` fixed the redundancy and left `LUT build` at
0.51s against VQ3R's 0.28s. Vectorizing the pass with NEON changed nothing
(0.52 -> 0.52), which is the measurement that identified what it actually
was: **not arithmetic, dispatch.** 260 builds a token, each 4 to 9 scale
blocks, ~79us of fork-join for ~7us of work.

Serial and vectorized, `LUT build` is 0.28s — parity with VQ3R while doing
the extra pass. `vq_quant_lut` is therefore deliberately not threaded, and
that is the note that keeps someone from "fixing" it.

**The general form:** a pass that is 2% of a kernel's *work* is not 2% of
its *time* if it runs once per caller instead of once per input, or on one
thread instead of the pool. Both were visible in the profile within a
minute of looking, and neither was visible in the estimate.

## 43. An int8 table makes the engine discontinuous (2026-08-03)

Weighting each expert inside its own task (`part[i] = w[j] * acc[i]`) and
summing afterwards, instead of `ysum[i] += w[j] * acc[i]` in the caller,
moved a VQ3R logit by 5.7e-06 and a **VQ4P logit by 0.68**.

Both paths were correct. Per layer they agreed to 1.9e-06 — *identically*
for the two formats. The asymmetry is downstream: `vq_quant_lut` takes its
scale from `max|lut|`, so the int8 table is a step function of its input. A
perturbation of 1e-8 moves the scale, and every entry sitting near a
rounding boundary moves by one LSB. VQ4P amplifies float noise about five
orders of magnitude harder than VQ3R does.

The perturbation itself was FMA contraction: `ysum[i] += w[j] * acc[i]`
fuses into a single rounding, and rounding the product first does not. The
fix is to leave `acc` unweighted and let the caller apply `w[j]` in the
same expression the serial loop uses, so the compiler contracts it the
same way.

**For VQ4P, "numerically equivalent" is not a good enough standard for two
code paths — they have to be bit-identical**, and that is now checked:
row-parallel against expert-parallel, one thread against eight, NEON
against `-DWASTE_P6_SCALAR`, all `cmp`-clean. The int8 table is what
raises the bar, and it will raise it for any future path that touches
these kernels.

## 44. A batch is both the parallelism and the barrier (2026-08-03)

At Kimi-Linear's expert shapes one `vq_apply` is a few microseconds of
arithmetic against a fork-join that costs tens, and a token spends ~900
dispatches. Giving each routed expert its own task instead — one dispatch a
layer, 26 a token — is worth 1.15x on VQ3R and 1.18x on VQ4P, and it
removes the thread-count cliff: row-parallel was *worst* at the default
thread count and needed `WASTE_THREADS=6` to look good, expert-parallel is
best at the default.

**On K3 it is a regression, and no batch size fixes it.** Holding a
layer's records before computing is a barrier against the read-ahead; the
hint has already queued all sixteen reads, and waiting for the last one
before starting the first expert stops them overlapping the multiplies they
were issued to hide behind. Batching the holds bounds the barrier — but the
batch is also how many experts can run at once. 15 steps, K3, VQ3R,
internal disk:

| mode | expert I/O | expert mm | accounted | s/token |
|---|---|---|---|---|
| row-parallel | 3.71 | 12.78 | **37.24** | **1.61** |
| expert, batch 1 | 0.78 | 51.44 | 81.80 | 4.00 |
| expert, batch 2 | 1.11 | 27.20 | 49.01 | 2.46 |
| expert, batch 4 | 4.10 | 15.62 | 42.16 | 1.87 |
| expert, batch 8 | 9.66 | 10.68 | 46.25 | 2.03 |
| expert, batch 16 | 15.69 | **6.15** | 56.33 | 1.87 |

Read the two interior columns against each other: batch 1 overlaps the I/O
perfectly and leaves one thread working, batch 16 does the arithmetic
2.1x better than row-parallel and pays 15.7s of stall for it. Row-parallel
declines the trade — it pipelines one expert at a time while putting every
thread on that expert's rows.

**So `WASTE_XPAR` is off by default.** The two strategies suit opposite
regimes and the regime is set by which of dispatch and disk is the budget,
which is a property of the model and the machine, not something the engine
can read off the container.

What would get both is a task granularity of (expert, row range) with a
small expert batch, staged so the gate/up applies, the down LUT builds and
the down applies are three dispatches per batch instead of one. That
decouples the parallelism width from the barrier width. It is not written,
and it is the obvious next thing here.

## 45. The disk contaminates the buckets that do not touch it (2026-08-03)

The K3 VQ4P container went to the external disk because the internal one
had 691 GB free against a 982 GB container. The reasoning for accepting
that was: `WASTE_PROFILE` separates expert I/O from arithmetic, so
`LUT apply` and `expert mm` stay comparable even at 0.94 GB/s instead of
12.78. **That reasoning is wrong, and `kda` is the control that shows it.**

10 decode steps, row-parallel, same prompt, same step count:

| container | LUT build | LUT apply | kda | mla | expert I/O | expert mm | accounted |
|---|---|---|---|---|---|---|---|
| VQ3R, internal | 1.03 | 7.55 | 9.14 | 1.61 | 2.47 | 8.79 | 28.66 |
| VQ4P, external | 1.37 | 8.77 | **15.60** | 2.96 | 131.38 | 10.36 | 171.63 |

`kda` reads no expert record at all — it is trunk arithmetic — and it is
71% slower. A profile taken while the disk is saturated is not a profile of
the arithmetic with one column swapped out; the whole run is slower and
every bucket carries some of it.

**"71% slower than one other run" is not the evidence, though, and saying
so was sloppy.** Five repetitions of the same internal baseline put `kda`
between 8.65s and 14.52s over 15 steps — a 68% spread on identical work,
this being a desktop with a window server on it. A single pair of readings
cannot clear that band. What clears it is the rate: 0.58–0.97 s/step across
six internal runs against **1.56 s/step external**, 61% above the highest
internal observation rather than 71% above one of them. The conclusion
stands; the argument for it needed a noise band, and the first version
would have gone in the file as a fact that the next five runs contradicted.

The rest of that spread is worth knowing on its own: `s/token` is stable to
±1.5% (1.62–1.67) while `accounted` swings 38% and `kda` 68%. **Compare
decode s/token; treat a single bucket reading as an estimate** unless it is
a median of several.

**So there is still no clean number for the VQ4P kernel on K3.** The
normalized estimate (LUT apply relative to `kda`: 0.826 -> 0.562, about
1.47x) is arithmetic on two contaminated readings and is recorded here as
the reason not to quote it. Getting the real one means the container on the
same class of storage as its baseline, which on this machine means deleting
the 982 GB VQ3R container first — and §46 is why that is now worth doing.

**The rule.** Before measuring a container on different storage than the
one it is compared against: don't. Copy it, free space, or measure
something else. If it cannot be avoided, put a bucket in the profile that
touches no expert bytes and read that one first — that check costs one
column and would have saved this run.

## 46. The stall bucket is not the disk floor (2026-08-03)

**Do not derive the disk floor from the `expert I/O` bucket.** That was
tried twice in one afternoon, in two different ways, and both were wrong in
opposite directions.

The first read §10's 9.9 GB/s — a throughput *observed during a run*, not
the device's capability — and concluded the floor was ~1.43s against a
1.87s step, so any compute win was capped near 1.3x.

The second reasoned that if reads pipeline behind arithmetic then the stall
is exactly the excess, so `disk = compute + stall`. On the K3 row-parallel
run (34.64s accounted, 3.11s stall) that gives disk 34.64s against compute
31.53s, i.e. the disk already binding and *any* arithmetic win worth
nothing. It is a tidy derivation and it inverts the moment the pipelining
is imperfect, which is the case it was invented to reason about.

**Measured instead, with `tools/diskbench.c`, at the pattern the engine
actually uses** — 12 MB records, random, cache-bypassed, 8 reader threads,
on the internal SSD:

| | |
|---|---|
| seq write / seq read | 11.10 / 10.55 GB/s |
| random, 1 thread | 10.82 GB/s |
| random, 8 threads | **12.87 GB/s** |

That settles it. The same run reads 223.86 GB over 15 steps, so the disk
owes 17.4s against 31.5s of arithmetic: **K3 decode at a 17.7 GB cache is
compute-bound, by about 1.8x.** The 3.11s of stall is not the disk running
out of headroom, it is reads that failed to hide behind arithmetic there
was plenty of.

Per token that is a 1.16s floor under a 1.61s step — **1.39x of headroom**,
and §41's kernel takes `LUT apply` from ~0.51s to ~0.15s, which would land
at ~1.28x. That prediction was worth the measurement it cost, because it
was also wrong.

**Measured: 1.09x.** The VQ4P container was copied onto the internal disk
(the VQ3R baseline moved out to make room, measured first and restored
after), so both sides sit on the same storage with the same build. Medians,
13 VQ4P runs against 5 VQ3R:

| | VQ3R | VQ4P | |
|---|---|---|---|
| **s/token** | **1.65** | **1.52** | **1.086x** |
| LUT apply | 11.21 | ~9.6 | 1.17x |
| expert I/O | 3.15 | 3.69 | slightly more exposed |
| `kda` (control) | 10.35 | 10.52 | unchanged |

`kda` unchanged is what says the two are comparable this time, which is the
check §45 was written about.

**The kernel is not the problem, and neither is cache residency.** The
obvious suspicion was that §41's 3.32x lived in a 3.94 MB index buffer hot
in L2 while the engine streams ~17 GB of index per token. Sized up, one
pass, no repetition:

| index working set | VQ3R | VQ4P | speedup |
|---|---|---|---|
| 3.9 MB | 1.268 ms | 0.290 ms | 4.37x |
| 63 MB | 9.882 | 2.478 | 3.99x |
| 252 MB | 37.178 | 9.630 | 3.86x |
| 1008 MB | 151.029 | 38.880 | **3.88x** |

It holds at 3.88x against a gigabyte. So that hypothesis is dead too — the
third of the day, and the reason this section says what is measured and
stops. The instrument is `tools/lutbw.c`, which exists so the next person
to suspect a cache artefact can settle it in a minute instead of a day.

**What is left is a scaling failure nobody has explained.** Converting the
buckets to throughput over the ~17.4 GB of index a token touches:

| | one thread | in-engine, ~10 threads | scaling |
|---|---|---|---|
| VQ3R | 6.5 GB/s | 23.3 GB/s | 3.6x |
| VQ4P | 25.3 GB/s | 27.2 GB/s | **1.07x** |

VQ4P in the engine runs at its single-thread rate. It is not the chunk
size — `WASTE_P6_CHUNK` was swept 1..16 on K3 and every value landed inside
the run-to-run noise — and 25 GB/s is far too low to be a memory ceiling on
this machine. Recorded as an open question rather than a guess; §47
answers it.

**So the standing recommendation is unchanged, for a new reason.** VQ4P is
worth having where the apply is dispatch-bound and small (Kimi-Linear,
1.18x) and is worth 1.09x on K3 — real, but not a reason to reconvert
982 GB, and nowhere near what the kernel does in isolation. The gap between
3.88x on a bench and 1.17x in place is the whole finding.

**This also dents the standing model.** "Disk I/O is the budget" and "~53%
of a K3 decode step is expert reads" were true when they were written and
are not true here: §35's lookahead, the reader pool and a 17.7 GB cache
have moved K3 to the other side of the line. The claim is worth re-checking
against `diskbench` whenever it is leaned on, rather than inherited.

**The rule.** Before claiming anything is disk-bound, run `diskbench` and
divide. The stall bucket says how much I/O failed to overlap; that is a
different question and it does not answer this one.

## 47. The fast kernel is the one the E-cores hurt (2026-08-04)

§46 left a hole: the VQ4P kernel is 3.88x standalone and 1.17x in the
engine, it is not cache residency and it is not memory bandwidth, and the
throughputs said it runs at its single-thread rate however many threads the
pool has. Three things, and the third inverts between models.

The instrument is `tools/lutmt.c` — the same two kernels driven through the
engine's own `waste_parallel_for`, with thread count and chunk from argv,
so the dispatch under test is the real one and the engine's noise is not.

**One: 3.88x is a single-thread ratio, and the slow kernel parallelizes
better.** K3's gate shape, index buffers rotated so no pass re-reads the
last one's bytes:

| threads | VQ3R | VQ4P | ratio |
|---|---|---|---|
| 1 | 6.6 GB/s | 26.1 | **3.94x** |
| 4 | 20.0 | 74.5 | 3.72x |
| 6 | 27.9 | **91.3** | 3.28x |
| 8 | 28.7 | 88.4 | 3.08x |
| 10 | 29.8 | 73.2 | 2.45x |

So the ceiling in a threaded engine was never 3.9x. It is ~3.3x, and only
at one thread count.

**Two: this machine is 6 P-cores and 12 E-cores, and the pool takes all
18.** VQ4P peaks exactly on the P-cores and *degrades* past them — 91.3 down
to 73.2. VQ3R does not: it keeps improving to 8-10. The asymmetry is that
VQ3R is latency-bound, so a slow core costs it proportionally little, while
VQ4P is wide and fast and an E-core running the same chunk is a straggler
the barrier waits for. `waste_parallel_for` cuts work into `ceil(n/nthreads)`
— one task per thread, nothing left to steal — so the straggler is
structural rather than unlucky. Oversubscribing helps (12 tasks on 10
threads: 73.1 GB/s against 56.3 at 48 tasks and 66.7 at 4) and does not
recover the 6-thread peak.

Not bandwidth: a 788 MB working set, cold, measures 91.3 GB/s at 6 threads
against 90.5 warm. Not cache residency either — that was already dead in
§46.

**Three: an apply can be too small to pay for a fork-join.** At
Kimi-Linear's shapes one task is ~9 us of arithmetic against tens of us of
dispatch. `WASTE_P6_CHUNK=16` was worse than useless there in a way worth
naming: `min_chunk` of 1024 against M=1024 hits
`if (n <= min_chunk) { fn(0, n, arg); return; }`, so **the apply ran
serially**, and it won its own sweep because with 18 threads every
alternative was worse. A default chosen that way is a default chosen by a
bug in the setup.

**What is actually achievable**, Kimi-Linear, three runs each, stable to the
last digit:

| | accounted |
|---|---|
| VQ3R, engine default today | 1.84 |
| VQ3R, best (`WASTE_XPAR=1`) | **1.31** |
| VQ4P, row-parallel, 6 threads, chunk 1 | 1.30 |
| VQ4P, best (`WASTE_XPAR=1`) | **1.06** |

Best against best is **1.24x**, and against what the engine does untouched,
**1.74x**. Both are better than the 1.18x §41 recorded, and the difference
is entirely configuration.

**And then K3 inverts it.**

| K3, VQ3R | s/token |
|---|---|
| default (18 threads) | **1.68** |
| 6 threads | 2.25 |
| 8 threads | 2.35 |
| `WASTE_XPAR=1`, 6 threads | 1.89 |

Fewer threads is 34% *worse* on K3. Its applies are 4.7x larger, so they
amortize the dispatch and genuinely use every core the machine has,
E-cores included. "Cap the pool at the P-cores" is a 25% win on one model
and a 34% loss on the other.

**So no default changes.** `WASTE_P6_CHUNK=16` is wrong at 6 threads and
right at 18; `WASTE_XPAR` is right on Kimi-Linear and wrong on K3; the
thread count that is best on one is worst on the other. Any default picked
here is tuned for one model against the other, which is why these are
switches and why the tuning table above is the deliverable rather than a
commit that moves a constant.

What would deserve building, if this comes back: a pool that knows which
cores are performance cores and sizes SIMD-heavy work to them while leaving
the latency-bound kernels the whole machine. That is a real change to
`threads.h` and it is not justified by one kernel on one laptop.

## 48. Gate 1 answered, on hardware this repo does not have (2026-08-04)

**Third-party measurement. Not reproduced here, and it cannot be** — there
is no x86 server and no NVIDIA card on this machine, which is the reason
issue #11 was written as a set of gates for someone else to run rather than
as a plan.

`ssarthak15` ran gate 1 on one Oracle bare-metal DenseIO node, dual EPYC
7J13 (Zen 3, 32 cores, two-socket NUMA interleave), K3 with direct expert
I/O from NVMe. `lm_head` medians over 33 measured decodes after 16 warm
ones, in each of three fresh processes, against a matched host scan on the
same cores, affinity mask and NUMA policy over 32 GiB — 64x the node's
512 MiB aggregate LLC:

| | `lm_head` | effective | matched host | ratio |
|---|---|---|---|---|
| 1 | 7.100 ms | 168.0 GB/s | 170.2 GB/s | **98.7%** |
| 2 | 8.827 | 135.1 | 171.7 | 78.7% |
| 3 | 8.688 | 137.3 | 164.9 | 83.2% |

Threshold declared before the run was 70%. **The x86 path is bandwidth-bound
too.**

**Why it is trustworthy without being reproducible.** The byte accounting
is what would give away a number that had not been measured, and it
reconciles exactly with this repo's own format. 1,174,405,120 payload bytes
against 18,350,080 bytes of fp16 scales is 128 elements per scale, which is
`quantize_q8g(W, group=128)`; the payload is 163840 x 7168, K3's vocab by
its hidden, so it is that tensor and not a stand-in; and `lm_head` does
keep 8 bits in a default conversion while the trunk goes to 4, so it is
also the same width `docs/BACKENDS.md` measured Metal against. Every
derived figure divides back out, including 1/1947 ms against the pooled
0.5139 tok/s.

**The AVX2/AVX-512 gap is smaller than it looks.** Gate 1 asked about
AVX-512 and this is AVX2. A wider ISA moves the same bytes with fewer
instructions, so a kernel already at 79-99% of achievable bandwidth has
nowhere to go but 100% — the answer's *direction* does not depend on the
ISA, only its exact value does. Zen 3 has no AVX-512 at all, so it was not
measurable on that node regardless.

**Against the Metal row it replaces**, same tensor, same width, same one
dispatch per token:

| | `lm_head` | effective |
|---|---|---|
| Apple silicon, NEON (2026-07-28) | 6 ms | 195 GB/s |
| dual EPYC 7J13, AVX2 | 7.10 ms | 168 GB/s |

The laptop beats the 16-channel dual-socket server on this kernel. Both are
at their machine's bandwidth, which is the finding.

**What it settles.** The clause in `docs/BACKENDS.md` — "the CPU path is
already running at the machine's memory bandwidth, and this is a
bandwidth-bound matvec" — was an Apple-silicon observation being asked to
carry an argument about accelerators in general. It now has an x86 leg. So
filling the `waste_backend` slots with CUDA kernels reproduces the Metal
result on different hardware, and issue #11's gate 1 branch where the host
had headroom to reclaim is closed.

**What it does not settle.** Nothing about the "different engine" — one
dispatch per layer, residual resident in VRAM, which is where a discrete
card's bandwidth advantage would actually live. That is gate 2, the
end-to-end cost of one dependent matvec over PCIe against this 7.10 ms, and
it remains unmeasured.

**And gate 3's premise has moved since the issue was written.** It bounded
an accelerator at roughly 2x by Amdahl on "~53% of a K3 decode step is
expert reads". §46 measured the disk at the engine's real access pattern
and found K3 decode compute-bound by about 1.8x at a 17.7 GB cache. Gate 3
is more open than it was posed, not less.

**One decode number worth keeping**, from the same run and the same
caveats — K3, 99 tokens in 192.645 s, one serial stream, 32 threads, greedy,
automatic budget:

| | |
|---|---|
| pooled | 0.514 tok/s |
| median forward | 1947 ms/token |
| suite range | 0.47-0.55 tok/s |

It stays here and not in `README.md`. Every number there was measured on
the commit it ships with, and dual EPYC is a class of machine this repo
cannot verify on.

## 49. The bench that certified the disk was reading RAM (2026-08-05)

§14 found `O_DIRECT` in a comment and nowhere in the code, and fixed the
engine. It did not look at `tools/diskbench.c`, which carries the same
sentence in its own header — "with the page cache bypassed (F_NOCACHE /
O_DIRECT)" — and had the same hole: `nocache()`'s body was `#ifdef
__APPLE__` with nothing else in it, and all three opens were unqualified.

Reported by `fab2s` as PR #22, **on hardware this repo does not have** —
Samsung 970 PRO, PCIe Gen3 x4, Ubuntu 26.04, 16 GB file, 3 MB records:

| | before | after | link ceiling |
|---|---|---|---|
| seq read | 44.67 GB/s | 3.15 GB/s | 3.94 GB/s |
| random, 1 thread | 36.75 | 2.91 | |
| random, saturated | 65.72 | 3.33 | |

11x and 17x over the link. The tell was there in every run and nobody
divided: a Gen3 x4 drive cannot deliver 65 GB/s whatever the benchmark
says, and after the fix it saturates at two threads and 85% of the
ceiling, which is what that drive should do.

**What it cost.** §46 ends with a rule — before claiming anything is
disk-bound, run `diskbench` and divide. On Linux that rule returned a
fiction from 2026-07-28 until now. No published number moves: every
`diskbench` figure in `docs/GATES.md`, `docs/EFFICIENCY.md` and §44/§46
was measured here, on macOS, where `F_NOCACHE` did work. But the rule had
no force on the platform most users are on, and Gate H is exactly the
class of decision — 1.5 TB onto the wrong device — it exists to protect.

**The general form: the engine's rules bind the tools that measure the
engine.** `bank_open` bounds its bypass, probes it with a real transfer
and reports when it did not get it. `diskbench` asserted one in a header
comment. That is now three instances of one bug class in this repo —
issue #4 (an alignment test that was false for every container that
exists), §14 (the flag that lived only in a comment), and now the tool the
disk-bound claim rests on.

**`F_NOCACHE` does not evict, and that is not a detail.** Reviewing the
fix, the write looked like it should stay buffered: row 1 stands for the
download and the conversion landing, and those write through the page
cache like everything else. On Linux that holds — a subsequent `O_DIRECT`
read writes back and invalidates the range first, so the leftovers cannot
flatter the read rows (reasoned, not measured; no Linux here). On macOS it
is wrong, and measurably so. `F_NOCACHE` stops *new* pages being cached; it
does not evict resident ones. A buffered write leaves the whole file in
the UBC and every read row below then measures RAM. Same binary, 1 GB
working file, 4 MB records, internal SSD, differing only in whether the
write fd got the bypass:

| | write bypassed | write buffered |
|---|---|---|
| seq read | 7.9-8.1 GB/s | **26.04 GB/s** |
| random, 1 thread | 6.8-7.0 | **24.34** |

3.2x and 3.5x of pure fiction, on the row that sets tok/s. The original
`nocache()` on the write fd was load-bearing and looked ornamental. The
bypass covers the whole file's lifetime or it covers nothing.

**So the fix is not the flag.** `O_DIRECT` is accepted at open and refused
at transfer — tmpfs does this, and so would a device wanting a bigger
block than the tool aligns to — so a bare flag turns a refusing filesystem
into `short read -1` and a table of zeroes with no cause given. It now
does what `bank_open` does: probe with one aligned transfer, fall back to
a plain open plus `POSIX_FADV_RANDOM`, and label every row `(cache
bypassed)` or `(PAGE CACHE, not the disk)` with a trailer explaining it.
A measurement that quietly means something different is worse than one
that is missing — §14 said that about the engine and it is truer of the
tool, because the tool is what the claim rests on.

**What is verified, and what is not.** macOS is unchanged against `main`
within noise. The Linux body compiles and runs here only against stubs for
`O_DIRECT` and `posix_fadvise` — covering both the probe-succeeds and the
probe-refused paths, and confirming the write probe restores the file byte
for byte — which is the same limitation §14 recorded for the engine, for
the same reason. The three-column table above is the reporter's. **The
platform still has not been measured from here.**

## 50. Gate 2 measured: the GPU wins the matvec and loses the transfer (2026-08-05)

**Third-party measurement, second contributor, and again not reproduced
here.** `fab2s` ran gates 1, 2 and 3 on a consumer desktop — Ryzen 9 9900X
(Zen 5, AVX-512, two 6-core CCDs with separate 32 MB L3) and an RTX 5060 Ti
(sm_120, 36 SMs, 15.5 GB usable, 448 GB/s theoretical, PCIe Gen5 x8
confirmed under load), on Kimi-Linear-48B with a default VQ3R container.
§48 could not cover AVX-512 (Zen 3 has none), could not cover a second
model, and — the part that changes its conclusion — measured one isolated
kernel rather than the aggregate step.

### Gate 1, re-answered with levers: the *step* is not bandwidth-bound

Instead of a ratio against a STREAM ceiling, one resource varied at a time
over byte-identical work — same container, same pinning, same `-n`, with
`bench --json` reporting identical `bytes_read` and hit/miss counts, and
clocks sampled *during* the load over the pinned cpuset:

| lever | change applied | throughput | Amdahl f |
|---|---|---|---|
| core clock, max-freq cap, 3629 → 5327 MHz in-load | +46.8% | 11.73 → 15.99 (+36.4%) | **0.84** (0.79-0.84) |
| DRAM, JEDEC 4800 → EXPO 6000 | +25.9% bandwidth | 14.90 → 15.63 (+4.9%) | **0.23** (0.21-0.26) |

Solving `1/(1+y) = (1-f) + f/(1+x)` on each side. The two fractions come
from independent levers on different boots and approximately partition the
step. Samples were medians of 3-5 with cooldowns and the first run
discarded; the DRAM sides are non-overlapping with under 0.6% spread each,
and the measured bandwidth change matches the nominal DIMM change (63.4 →
79.8 GB/s).

Four corroborations, none of which rely on a cross-session absolute:

- **Throughput does not scale with parallelism** past one CCD's physical
  cores: 6 threads 15.94 tok/s, 12 threads 15.54, 24 threads 12.67.
- **One core cannot saturate DRAM** — the same DIMM change moves STREAM read
  +1.8% at 1 thread and +25.9% at 6, so the aggregate is not bounded by a
  single core's outstanding-miss capacity.
- **The fractions are complementary**, 0.84 + 0.23, from two levers measured
  on separate boots.
- **The ratio against the streaming ceiling gets *worse* as bandwidth
  improves.** Per-token traffic is ~1.61 GB (1.04 GB of trunk re-read every
  token, plus 214 expert records at 2.54 MiB, measured at `-n 256` with
  prefill and read-ahead included against a nominal 26 x top-8 = 208):
  24.0 GB/s of 63.4 (38%) at 4800, 25.2 of 79.8 (32%) at 6000. If bandwidth
  were binding, raising it would pull the workload *toward* the ceiling.

The 84% is not SIMD arithmetic. It is the LUT path's dependent
load → address → load gather chains — cache-hit latency counted in core
cycles, the same mechanism §7 and §41 kept arriving at from the ARM side.
That is work a wider vector unit does not touch.

**And the isolated kernel reproduces §48 exactly.** `lm_head.weight` here is
383,385,600 B (Q8G group 128, `[163840, 2304]`, 0.321x K3's tensor and
matching hidden 2304/7168): **6.600 ms/call, 58.1 GB/s, 73-76% of ceiling**,
identical across three repeats, clearing the 70% bar §48 declared in
advance — now on AVX-512.

**So §48's kernel number stands and its generalization does not.** "The x86
path is bandwidth-bound too" was inferred from a kernel the profiler puts at
4.9% of decode at `-n 5`, 7.3% at `-n 45`, and `docs/TECHNICAL.md` puts at
0.2% on K3. Both hold at once: **the kernel is bandwidth-bound, and it is
0.2-7.3% of the budget.** The other 93-99% is core-clock-scaled. Read §48 as
answering the question about `lm_head` and this as answering it about the
step.

That cuts both ways, and it is worth being explicit because issue #11
predicted otherwise. #11 said a host that is *not* bandwidth-bound has
headroom to reclaim, which weakens the accelerator case. But the host cannot
reclaim it: threads stop scaling at one CCD, and the bound is gather latency,
not width. Abundant thread-level parallelism is exactly what a GPU has. The
gate-1 branch that closes is "fill the `waste_backend` slots" — the same
branch §48 closed, for a different reason.

### Gate 2 — the first end-to-end PCIe measurement this project has

Correctness checked against a CPU reference at rel L2 1.02e-07 before any
timing. Empty-kernel dispatch floor: **4.39 us** launch+sync, 1.30 us
launch-only.

| | `lm_head` 383 MB | one expert matrix [1024x2304], 2.4 MB |
|---|---|---|
| kernel only | 0.9940 ms → 385.7 GB/s | 0.0059 ms → 405.3 GB/s |
| full round trip | 1.0561 ms | 0.0194 ms |
| dependent chain | 1.0011 ms | 0.0114 ms |
| queued chain | 0.9962 ms | 0.0059 ms |
| dependency cost | (noise floor) | 5.5 us |

85-90% of theoretical VRAM bandwidth, and against the CPU's 6.600 ms the
**full GPU round trip is 6.25x faster**. The clause carried over from Metal —
that the round-trip eats the win — does **not** transfer to a discrete card.
Dispatch is not the obstacle for a restructured engine either: 27
dispatches/token is 0.1 ms. It only binds the backend-shim shape, at 624
dispatches x 11.4 us = 7.13 ms/token.

**The deciding term is the expert stream:**

| | |
|---|---|
| H2D pinned, 544 MiB = one token's routed experts | **19.807 ms → 28.8 GB/s** |
| H2D pageable | 20.812 ms → 27.4 GB/s |
| share of a measured 62.7 ms CPU token | **31.6%** |
| expert set vs usable VRAM | 16.5 GiB vs 15.5 GiB — does not fit |

28.8 GB/s is 90% of Gen5 x8, so the link is behaving. **It is also slower
than this CPU's own RAM at 63-80 GB/s.** The card is structurally on a worse
path to the same bytes: read experts into host RAM, then push them across a
link at half the speed the host already had them at.

**The expert matmul was implemented and measured, not substituted.** Taking
the term from `matvec_q8g` would have been wrong — experts are residual VQ,
not int8, and 16.53 GiB of 3-bit experts is ~88 GB at f16, so they must be
decoded every token. One expert's gate+up+down, indices and codebooks
straight out of the container, checked against a CPU reference at 3e-07:

| per token, 26 layers x top-8, kernels only | decode-then-matvec | LUT, as the engine amortizes it |
|---|---|---|
| **VQ3R** (stages=3) | **13.86 ms** | 15.85 ms |
| VQ2R (stages=2) | 12.48 ms | **9.70 ms** |

`vq_apply` scales with stages (0.0428 → 0.0712 ms per expert, +66% for +50%
lookups); reconstructing the weights and doing an ordinary matvec is nearly
flat (0.0600 → 0.0666, +11%) because its M x N MAC term does not depend on
stages. **The two cross between 2 and 3 stages, and the GPU picks the
opposite algorithm from the CPU.** The reason is §41's, seen from the other
side: the LUT exists to save FLOPs, its table is 864 KB at three stages and
cannot leave L2, while the codebook is 24 KiB and sits in shared memory. On
a GPU the FLOPs are free and the gathers are not.

**Scope, and it matters: that is a 256-entry result, and `WQ_VQ4P` likely
inverts it.** VQ4P's table is 288 KB fp32 and **72 KB int8** against VQ3R's
864 KB, and 72 KB fits shared memory — which is the only reason
reconstruction won here. The crossover is a property of the codebook shape,
not of the device. Measuring it needs a 64-entry path in the benchmark, a
fresh conversion and 0.6.4. Not run.

Against the CPU's expert-matmul time — the profiler's share (48.7% VQ3R,
52.9% VQ2R) applied to the 62.7 / 67.9 ms bench medians, approximate because
profile and bench ran different read-ahead settings — that is **2.2x and
3.7x**. Real, and far from the 6.25x the contiguous `lm_head` matvec gets.

**Which corrects the projection.** With the expert term measured at 13.9 ms
instead of the 3.7 ms an int8 stand-in implied:

| term | ms/token |
|---|---|
| expert H2D | **19.8** |
| trunk read at 379 GB/s | 2.7 |
| expert decode + matvec, measured | **13.9** (was 3.7) |
| dispatch, 27 x 4.39 us | 0.1 |
| `lm_head` | 1.0 |
| total | **~37.5 ms/token, ~27 tok/s** |

**~1.7x** over the measured 62.7 ms / ~16 tok/s, not the ~2.3x the stand-in
implied, with transfer falling from 73% to 53% of the budget because the
compute term grew. Still a projection — there is no CUDA backend to measure
— and it ignores KDA and MLA, a further 24% of CPU time that would need
kernels of their own.

### Gate 3 — no, and it would not have helped

`gdscheck -p`, GDS 1.16.1.26, reports compat mode on all transports:
disk → host RAM → `cudaMemcpy`. The hop is not avoided. Three reasons that
is settled rather than pending:

1. **Not silicon.** The card reports `supports GDS`, BAR1 at the full
   16384 MiB, platform verification passes. `nvidia_fs` (min 2.12) is simply
   absent.
2. **Hostile to obtain.** `nvidia-fs-dkms` pulls a driver DKMS package at a
   different version than the prebuilt driver in use; prebuilt
   `linux-modules-nvidia-fs-*` target `-nvidia` kernel flavours rather than
   `-generic`; and GDS is not guaranteed with `iommu=on/pt`, which that
   machine needs.
3. **Amdahl-bounded anyway.** 98.1% of expert reads are served from the RAM
   cache, so GDS could touch ~2%, and on a miss the disk is the slow link
   (3.4 GB/s there), not the bounce buffer.

Note this is the opposite regime from the one §48 flagged: "~53% of a K3
decode step is expert reads" is cold-cache and K3-scale, which is also the
scale a 16 GB card cannot serve at all. The two do not overlap.

### The configuration that would invert gate 2, built and killed

The expert set misses VRAM by 1 GB. If it fit, the PCIe hop would become a
one-time load and the per-token transfer term would vanish. So a VQ2R
container was built with `convert.py --stages 2`, everything else default.

**It fits, with room to spare:**

| | VQ3R | VQ2R |
|---|---|---|
| expert bank | 16.53 GiB | **11.04 GiB** |
| per-expert record | 2.543 MiB | 1.699 MiB |
| resident (trunk + state + scratch) | 1.24 GiB | 1.21 GiB |
| total on device | 17.77 GiB | **12.25 GiB** |
| fits 15.5 GiB usable | no, by 2.27 GiB | **yes, by 3.25 GiB** |

The bank ratio is 0.6682 against a bits-per-weight ratio of 0.6667; the
difference is per-row f16 scales and index-block padding, which do not scale
with stages.

**On the CPU it is slower, by 7.7%**, medians of 3, 6 threads pinned to one
CCD, `-n 512`, 96.5% hit rate in every case:

| container | budget | median tok/s |
|---|---|---|
| VQ3R | 22G | **15.9601** |
| VQ2R | 22G | 14.7337 |
| VQ2R | record-scaled 15.78G | 14.6873 |

The scaled budget holds cache capacity constant in *records* — without it
the smaller container simply gets a bigger cache and the comparison measures
hit rate instead of format. Both VQ2R runs agree to 0.3%, so it is intrinsic.
`WASTE_PROFILE=1` says where it goes: expert I/O falls 22% as expected
(0.69 → 0.54 s, 15.1% → 11.5%) and expert mm rises 11% (2.22 → 2.47 s,
48.7% → 52.9%) and cancels it. That was an implementation artifact on 0.6.3
— `vq_rows` had an `if (st == 3)` fast path and `st == 2` fell through to the
generic one-row loop, so VQ2R issued 33% fewer lookups and still lost. §41
has since reworked that path around a 64-entry table, so treat it as context
for the numbers above rather than a standing gap.

**Quality is what actually closes the loophole, and it reproduces
`docs/GATES.md` Gate 3 independently.** Reconstruction error against source
weights, 312 tensors each: VQ3R median **19.51%** (19.39-22.08), VQ2R median
**33.19%** (33.05-37.63) — against the 19.4% at 3 bits and "2-bit VQ stays
unsafe at 33%" recorded there, on different hardware and a different model.
`verify_container.py` FAILs the VQ2R container at its 0.30 threshold, which
is that same operating point; the parse itself is clean. The logit proxy
agrees the damage is real without being dramatic at one step: top-1 agrees,
KL 0.0179 nats, but top-10 overlap is 7/10, logit rel L2 is 9.42%, and greedy
continuations diverge at the third token.

**So gate 2's answer holds for every configuration that meets this project's
own quality bar.** The only shape whose expert bank fits 15.5 GiB is the one
Gate 3 rules out; the shape that passes Gate 3 misses VRAM by 2.27 GiB. The
two do not overlap — the same structure as the gate 3 answer above.

### Method notes worth keeping, independent of CUDA

- **On `amd_pstate` in active mode the governor is not a clock lever.** Under
  sustained load powersave boosts to 5332 MHz against performance's 5327 —
  identical. A governor toggle compares idle-clock labels on same-speed runs.
  A clock lever must be a max-frequency cap, verified in-load, on both sides.
- **Pin threads within one CCD.** Splitting 6 threads across both CCDs costs
  **16-25%** at identical thread count and identical work. That is `--cpus` /
  `WASTE_CPUS` measured from the outside by someone who did not know it was
  landing, and it is the strongest argument yet that the flag is not
  optional tuning.
- **Hold `-n` fixed** when sweeping — `--threads` also sizes the reader pool,
  so at low token counts the ranking between thread counts inverts.
- **Use `-n 1024`+.** Misses are unique-expert first touches, a fixed cost,
  so hit rate rises with length: 94.0 / 96.5 / 98.1% at 256 / 512 / 1024.
- **Check `bytes_read` matches** before comparing throughput at all.
- Over a long back-to-back campaign (32 runs, ~30 min) that machine produced
  occasional ~30% low outliers on byte-identical work — not thermal, not
  competing processes, not huge-page fallback, not fragmentation, cause
  unidentified. Short series showed none. Anything measured over a long
  campaign needs medians and within-round ratios.
- Gate 2 needs no host CUDA install if a CUDA >= 12.8 image is available
  (12.8 added sm_120).

### What it settles

A discrete consumer card is answered, and for a **different reason than
Metal was**. Metal died on the round trip; here the round trip is 6.25x
favourable and the kernels hit 85-90% of VRAM bandwidth. What kills it is
expert-transfer bandwidth plus VRAM capacity: PCIe is slower than the host's
own RAM, and the experts do not fit.

That names the condition under which it flips, which is the useful part.
**VQ3R's 17.77 GiB fits a 24 GB part comfortably** — and then the H2D term
becomes a one-time load rather than 19.8 ms every token, which is the whole
deciding row. The GPU VQ-decode throughput measured above applies unchanged
to that case.

Not measured: bandwidth against DRAM latency separately, GDS in GDS mode,
KDA and MLA on a GPU, the VQ4P 64-entry crossover, VQ2R quality on a real
eval, gate 4, contexts beyond 4096, prefill as distinct from decode.

## 51. Four numbers that describe one prompt, and only one is the disk (2026-08-09)

Written down because the confusion is easy to have and expensive to act on:
"how many experts does a prompt read" has four different answers, and the
one that sets tok/s is the smallest of them.

On `kimi-linear.waste` (26 MoE layers, 256 experts, top-8), a 23-token
Italian prompt, `WASTE_DUMP_ROUTE` for the trace and `test_forward … 0` so
prefill is the whole run:

| | |
|---|---|
| routing decisions (tokens x MoE layers) | 598 |
| expert **activations** (x top-8) | 4784 |
| **distinct** `(layer, expert)` records | 1803 |
| records read on the best measured run | 1803 |

The gap between 4784 and 1803 is intra-prompt reuse, and it is not evenly
spread: 898 of the 1803 are touched exactly once, while 8 of them are
touched on 22 of the 23 tokens.

Same prompt, four configurations, changing only cache size and lookahead:

| config | hit | misses | evictions | GB read |
|---|---|---|---|---|
| 8 GB cache, lookahead off | 62.3% | **1803** | 0 | 4.48 |
| 4 GB cache, lookahead off | 62.1% | 1812 | 202 | 4.50 |
| 4 GB cache, lookahead 6 (default) | **83.0%** | **814** | 404 | **5.00** |
| 64 MB cache, lookahead off | **0.0%** | 4784 | 4759 | 11.88 |

Two rows are worth keeping for what they confirm. The first: with the cache
above the working set and no speculation, misses land on **1803 exactly**,
the distinct count computed from the trace — the counter is not
approximating anything. The last: 64 MB holds 25 records against the 208 one
token needs, every reuse has been evicted before it is reused, and the hit
rate is **0.0%**, which is §4's zero reproduced on a second model and a
second architecture.

**The row that matters is the third, and it reads more bytes than the
minimum while stalling half as often.** 5.00 GB against 4.48 — +11.6% of I/O
— for 814 blocking reads against 1812, **-55%**. That is not a contradiction
between two counters, it is the reason they are two counters: `misses`
counts accesses that waited on the disk, `bytes_read` counts I/O including
the lookahead's speculation, and `src/ecache.c` keeps
`spec_issued` out of `misses` on purpose so a wrong guess cannot be read as
a cache that performed badly. On a design where the disk is the budget, the
stall is the currency and the byte is the price.

This is a third point on the curve §39 drew, and the sign is the other way
round: there, on K3 decode, the lookahead read 6.6% *fewer* bytes at 1498
slots and 8% more at 287. Here, at 1610 slots — 89% of the prompt's distinct
records — it reads 11.6% *more*. Consistent with the mechanism §39 named
rather than against it: speculation costs bytes whenever the demand stream
would have hit anyway, and a short prefill with heavy intra-prompt reuse is
the regime where it would have. It buys stalls, not bytes, and here that is
all it buys.

**Pool size, not model size, is what governs the reuse.** The same counters
on `k3.waste` (92 MoE layers, 896 experts, top-16), 4 prefill tokens, 8 GB
cache, lookahead at its default:

| | Kimi-Linear, 23 tok | K3, 4 tok |
|---|---|---|
| activations | 4784 | 5888 |
| distinct records | 1803 (**37.7%**) | 4492 (**76.3%**) |
| misses | 814 | 3919 |
| GB read | 5.00 | 66.56 |

Three quarters of K3's activations touch a record never seen before in the
prompt, against a bit over a third on Kimi-Linear. That is the whole reason
the budget resolver reasons in whole working sets on K3 and nobody has
needed it to on Kimi-Linear — and note the K3 run is deep in the thrashing
regime it describes, 692 slots against 4492 distinct records and 5068
evictions.

Not measured: the same counters on decode rather than prefill, prompts long
enough for the hot set to saturate, and whether the 37.7 / 76.3% split
tracks `n_experts` or `n_experts / top_k`.

## 52. There is no expert for history (2026-08-09)

The intuition that a MoE routes a question to a topical specialist — "this
one is maths, that one is history" — is worth refuting with numbers, because
it licenses an optimization that does not exist: pinning or prefetching a
per-domain expert subset.

Two Italian prompts on `kimi-linear.waste`, one historical and one
mathematical, 23 and 22 tokens, deliberately sharing their first token
(`13724`, "La"). The control comes free from that: at position 0 there is no
context, so the hidden state is a function of the token id alone.

```
token 13724, position 0
  L1  history  8 117 155  13 233 164 248 236
      maths    8 117 155  13 233 164 248 236
  ...
  -> all 26 layers, 8/8 identical
```

**The two prompts route the first token to the same 208 experts, all of
them.** The router is a matvec on the hidden state; it has no channel
through which "the question is about history" could reach it except the
hidden state, and at position 0 there is nothing in the hidden state but the
token. Mechanically obvious, which is the point of using it as the control:
it isolates the input.

Context does move it, and by a lot. Token `2694` ("di"), four occurrences
across the two prompts, experts in common over the 26 layers:

| | hist@4 | hist@20 | math@3 | math@19 |
|---|---|---|---|---|
| **hist@4** | 100% | 45% | 40% | 19% |
| **hist@20** | 45% | 100% | 32% | 13% |
| **math@3** | 40% | 32% | 100% | 27% |
| **math@19** | 19% | 13% | 27% | 100% |

Identical token, 13% to 45%. So the topical signal is real but second order,
and the ordering says how much:

| overlap of top-8 sets | |
|---|---|
| two tokens within the history prompt | 26% |
| two tokens within the maths prompt | 32% |
| a history token against a maths token | **16%** |

16% against 26-32% is a domain effect that exists and does not dominate: two
tokens of the *same* sentence already share only a quarter of their experts.
Over the whole prompts the pools are 69.3 and 56.2 distinct experts per
layer out of 256, overlapping at **29.8% Jaccard**.

There is also a trained force pushing the other way. The
`e_score_correction_bias` reorders the selection against the raw scores on
90% of Kimi-Linear's routing lines and on **368 of 368** of K3's — the
weights logged next to the ids are not monotone decreasing, and that is the
load-balancing term at work. A model with a maths expert would be a model
whose balancing had failed.

**What this rules out and what it leaves open.** Ruled out: any prefetch or
residency policy keyed on the topic of the request, and any offline
partition of the bank by domain. The exploitable locality is sequential —
the same experts recurring across nearby tokens, §51's 4784-to-1803 — which
is exactly what `waste_ecache_hint` and the lookahead already take.

Left open, and weaker than it needs to be: `usage.waste` warm assumes a
*global* hot set. Measured across these two prompts, records used by >=80%
of a prompt's tokens number 25 (history) and 34 (maths), sharing 17 — **40%
of the union**, falling to 27% at a >=50% threshold. So a global hot set
exists on this evidence and is the *minority* of either prompt's: between a
quarter and two fifths shared, the rest prompt-specific. That is neither the
warm list's premise nor its refutation, and 23 tokens is far too short to
settle it.

Not measured: whether longer or more homogeneous corpora separate the pools
further, whether the effect grows with `n_experts` (K3's 896 were not run
this way), and any of it on decode rather than prefill.

## 53. k3-mini: the experts do not average, because they are orthogonal (2026-08-09)

Asked directly: collapse every MoE layer's experts into one, find the best
weighting to do it with, and measure what the result costs and what it can
still say.

**The tooling for this and the next two sections lives on the `k3-mini`
branch, not on main** — `tools/merge_experts.py` picks the weights,
`tools/merge_layer.c` does the terabyte, `tests/check_mini.sh` reports both
halves of the trade, and `src/model.c` there carries a no-router path that a
merged container needs. None of it was merged, because what it measured is
that the direction is dead; the numbers are here so nobody builds it twice.
What did come to main is the resolver fix §57 describes, which this
experiment is what uncovered.

    Wbar = sum_e alpha_e * W_e            sum_e alpha_e = 1

The systems half is exactly what arithmetic says it must be. The model half
is a total loss, and the reason is one measured number.

### What a merged container costs and buys

Same machine, same prompt, `waste run --temp 0 -n 16`, default budget:

| | K3 | k3-mini |
|---|---|---|
| container on disk | 982 GB | **30 GB** |
| expert bank | 952.48 GB | 1.06 GB |
| parameters | 2.78 T total, 104.19 B active | 59.78 B total, 58.61 B active |
| `waste plan` floor | 29.19 GB | 28.80 GB |
| recommended budget | 80.77 GB (capped to 46.39) | **32.02 GB** |
| peak RSS | 45.43 GB | 28.91 GB |
| decode | 0.60 tok/s | **1.58 tok/s** |
| expert reads / token | ~17-19k | 92 |
| cache | 9023 hit / 14529 miss = 38% | 1472 hit / 0 miss = **100%** |
| what it says | "The capital of Italy is **Rome**." | `<\|close\|><\|close\|><\|close\|>…` |

Building it: 92 layers, 1.02 TB read and 2.7e12 weights decoded, **301 s**.

**The RAM floor barely moves — 29.19 to 28.80 GB — and that is the point
most likely to be misread.** The floor is the resident trunk, 27.28 GB, and
merging the experts does not touch it. What merging removes is the
*pressure*: K3 wants 80.77 GB and cannot have it, k3-mini is finished at
32.02 GB and more RAM buys it nothing. Same floor, no ceiling.

**The speedup is mostly arithmetic, not I/O**, which was not the
expectation. `WASTE_PROFILE=1`, 16 steps, K3 at a full working set of cache:

| bucket | K3 | k3-mini |
|---|---|---|
| expert mm | 14.49 s (35.2%) | 0.98 s (10.3%) |
| LUT apply | 12.48 s (30.3%) | 0.67 s (7.0%) |
| expert I/O | 2.96 s (7.2%) | 0.11 s (1.1%) |
| accounted | 41.15 s | 9.58 s |

Expert mm falls 14.8x and the LUT apply 18.6x, both close to the 16x that
top-16 to top-1 predicts, while expert I/O — the bucket this whole engine is
built around — was only 7.2% to begin with at that cache size. (`kda` also
falls, 12.29 to 4.58 s, for work that is nominally identical in both. Not
explained; most likely pool contention with the MoE phase, and it is not
what the table is about.)

### The weighting does not matter, and that is a property of the model

Four schemes, Kimi-Linear, against the unmerged model's next-token
distribution on the same 128-token prompt:

| alphas | KL (nats) | logit rel L2 | pearson | top-10 |
|---|---|---|---|---|
| `trace` — routed weight over 248 tokens | **2.668** | 0.328 | 0.751 | 2/10 |
| `uniform` — 1/E | 2.687 | 0.330 | 0.749 | 2/10 |
| `bias` — softmax(e_score_correction_bias) | 2.687 | 0.330 | 0.749 | 2/10 |
| `top1` — the single busiest expert | 3.710 | 0.493 | 0.512 | 1/10 |

The three averaging schemes agree to 0.7%, because **the alphas are nearly
the same vector under every principled rule**. The trained bias prior is
flat to three digits — perplexity 895.8 of 896 on K3, 255.9 of 256 on
Kimi-Linear, no expert above 1.08x uniform — which is what a load-balancing
term is trained to produce. There is no informative weighting to find,
because the model was trained not to have one. And `top1` is the control
that matters: concentrating the mass is *worse*, so the damage is not "the
average is too blurry".

### The gain sweep, which settles it

If the merged expert were a weak but correctly-aimed version of the layer,
scaling it up would help. `routed_scaling_factor` is one number in the
manifest, so the sweep costs only the runs (Kimi-Linear):

| gain | 0x | 1x | 2x | 4x | 8x | 16x | 24x |
|---|---|---|---|---|---|---|---|
| KL | **2.684** | 2.687 | 2.690 | 2.696 | 2.708 | 2.737 | 2.769 |

**Gain 0 is the best row**, and every increase is monotonically worse.
Reproduced on K3: gain 1 is KL 4.241, gain 0 is KL **4.065**. Deleting the
merged expert outright beats using it, on both models. The merged routed
path is not weak signal, it is noise, and what k3-mini actually runs is
"trunk + shared experts + attention" with a small harmful perturbation.

### Why: the experts are mutually orthogonal

Decoded straight out of the container, Kimi-Linear layer 1, 16 experts
sampled:

| | gate | down |
|---|---|---|
| cos(expert_i, expert_j), i != j | **0.0006** | **-0.0001** |
| \|merged\| / mean \|expert\| | 0.0646 | 0.0606 |
| 1/sqrt(E), E = 256 | 0.0625 | 0.0625 |
| cos(merged, expert), mean | 0.0653 | 0.0613 |

Distinct experts are orthogonal to measurement precision, and everything
else follows from that one line. The average of E orthogonal matrices has
1/sqrt(E) of their norm — measured 0.0646 against a predicted 0.0625 — and
cosine ~1/sqrt(E) with each of them. **The merged expert is 99.8%
orthogonal to every expert it was built from.** It is not a compromise
between specialists; it is a direction none of them points in, at a
sixteenth of the magnitude.

This also kills the repair that looks obvious. An FFN's intermediate
dimension is permutation-symmetric, so averaging unaligned neurons is
normally the bug, and the model-merging literature (git re-basin, OT
fusion) exists to fix exactly that. It does not apply: alignment presupposes
that two networks compute something similar up to a permutation, and
cos = 0.0006 says these compute nothing similar under any permutation.
There is no correspondence to recover, which is why no alignment mode was
built rather than built and found wanting.

§52 measured the same structure from the routing side. Orthogonality is what
"there is no expert for history" looks like in the weights.

### What it settles

- **Collapsing a MoE to one expert is not a compression of it, it is a
  deletion of it.** The honest description of k3-mini is "K3 with the routed
  path removed", and the gain sweep is proof rather than interpretation.
- **The speedup is real and belongs to the deletion**: a 59.78 B model
  instead of a 2.78 T one, at 2.63x the tokens per second and 16.5 GB less
  peak RSS.
- **No convex combination does better** — not a measured one, not the
  trained prior, not the busiest expert, not any gain. The failure is
  geometric, so it cannot be fixed by choosing alphas.

Two method notes. The merge tool refuses a job with no `alpha=`; it used to
default to uniform, which silently produced three byte-identical containers
from three different policies because the driver built the alpha files and
forgot to name them. And `merge_layer --check` reports the re-quantization
error separately from the merge, so a bad result has one suspect instead of
two: it lands at 0.195 rel L2, the same VQ3R figure §50 records, which is
how the decode path was shown to agree with `verify_container.py` (cosine
0.992 on a one-hot merge) before any of the above was believed.

Not measured: k > 1 clustering (merging 896 experts into 16 groups is one
token's working set held resident, and is a different geometry), merging
only the head of the usage distribution, distillation or fine-tuning of a
merged model back toward usefulness, and whether any trained MoE has experts
that are *not* orthogonal.

## 54. k = 16 clusters: routing was the exclusion, and residency is not free (2026-08-09)

§53 left one door open — "merging into more than one expert is a different
operation with a different geometry, and at k = 16 it would be exactly one
token's working set held resident. Not run." Run now, with `--clusters k`
and three partitionings. It closes the door twice: once on quality, and
once, unexpectedly, on speed.

### A correction to §53 first

§53 explained the merge schemes' agreement by saying "the alphas are nearly
the same vector under every principled rule". **That is wrong.** The trained
bias prior is flat, as §53 says, but the *measured* usage is not:

| Kimi-Linear L1 | busiest expert | top-16 share | perplexity |
|---|---|---|---|
| `trace`, 248 tokens | **15.1x uniform** | **34.5%** | 124.1 / 256 |
| `bias` | 1.05x uniform | 6.5% | 255.9 / 256 |

`trace` and `uniform` are genuinely different vectors — one substantially
concentrated, one flat by construction — and they still land at KL 2.668
and 2.687. The result is *stronger* than the explanation §53 gave it: a
correctly measured, concentrated weighting buys nothing over 1/E, which
rules out "you used the wrong alphas" instead of sidestepping it. Every
number in §53 stands; that one sentence of reasoning does not.

### The sweep

Kimi-Linear, KL from the unmerged model on the same 128-token prompt:

| k | merge, roundrobin | merge, random | prune (keep k busiest) |
|---|---|---|---|
| 1 | 2.668 | — | 3.710 |
| 4 | 2.663 | — | — |
| 8 | 2.662 | — | — |
| 16 | 2.721 | 2.545 | **2.200** |
| 32 | — | — | 2.292 |
| 64 | — | — | 2.396 |

Merging is flat from k=1 to k=8 and then gets *worse*. **Pruning — which is
not a merge at all, just "keep the k busiest experts and run all of them on
every token" — beats every merge at every k tried.** Both curves are
U-shaped, both minima are catastrophic, and every one of these containers
answers "The capital of Italy is" with **Paris**.

The norm law from §53 holds at the second point once the right n is used —
not the member count but the effective one, `1 / sum(alpha^2)`:

| | members | effective | measured ratio | 1/sqrt(effective) |
|---|---|---|---|---|
| k=1, uniform | 256 | 256 | 0.065 | 0.063 |
| k=16 roundrobin, cluster 0 | 14 | 2.99 | 0.561 | 0.578 |

and cos(member_i, member_j) is **0.0005 inside a cluster** — the same
orthogonality §53 found across the whole layer. Clustering does not find
similar experts to merge, because a layer does not contain any.

### What the shape says: the value of routing is the exclusion

The two curves cross in a way that names what is lost. Merging destroys the
experts, by the norm law. Pruning keeps k of them intact and still fails —
and gets *worse* as k grows past 16, while its logit rel L2 and pearson
improve monotonically over the same range (0.326 / 0.766 at k=16, 0.308 /
0.779 at k=64).

That combination is the result. The aggregate output moves closer to the
original as experts are added back; the next-token distribution moves away.
What a prune-k layer cannot do is **leave an expert out**. Adding the
17th-busiest expert to a permanently-on set adds its contribution to every
token, including all the tokens the real router would have excluded it
from, and past k=16 the wrongly-included mass grows faster than the
correctly-included mass.

So what routing buys is not which experts exist, nor how they are weighted
on average — §53 ruled out both — it is the per-token exclusion. That is
§52 seen from the weights: a token's top-8 shares 26% of its members with
its neighbour in the same sentence, and most of what a layer decides is
what *not* to run.

### K3 at k = 16, which is slower than K3

The systems half was supposed to be the easy win: 16 experts x 92 layers is
exactly one token's working set, so the bank is 18.26 GB instead of 952 GB
and never misses. Built in 635 s; 105.36 B parameters, 104.19 B active,
which is the same active count as K3 — k = top_k means the arithmetic is
unchanged and only the I/O should differ.

Same prompt, same 46.39 GB budget, `waste run -n 12`:

| | K3 | k3-k16 |
|---|---|---|
| decode | **0.52 tok/s** | **0.27 tok/s** |
| cache | 39% hit, 10731 misses | **100% hit, 0 misses** |
| bytes read (16 steps) | 218.83 GB | 17.01 GB |
| peak RSS | 48.52 GB | 47.76 GB |
| KL from K3 | — | 4.705 (k=1 was 4.241) |

**Zero expert I/O, identical arithmetic, less RAM, and it takes twice as
long.** `WASTE_PROFILE=1` at a matched 17.6 GB cache says the slowdown is
not in the MoE:

| bucket | K3 | k3-k16 | ratio |
|---|---|---|---|
| kda | 12.34 | 19.70 | **1.60x** |
| mla | 2.42 | 4.06 | 1.68x |
| expert mm | 15.01 | 25.99 | 1.73x |
| LUT apply | 12.99 | 23.97 | 1.85x |
| expert I/O | 3.51 | **1.50** | 0.43x |
| accounted | 42.31 | 63.36 | 1.50x |

**`kda` never touches an expert, and it is 1.6x slower.** So is `mla`. Every
bucket that does byte-identical work in both containers slows by the same
factor as the buckets that do not. That is a machine-wide slowdown, not an
algorithmic one, and it is the only reading the numbers allow.

The mechanism this is consistent with — and it is consistent rather than
isolated, since nothing here instrumented the OS — is the one §24, §32 and
§39 keep arriving at from other directions. k=16 touches **all** of its
18.26 GB of cache every single token and evicts nothing (0 evictions), on
top of a 27 GB trunk it also touches every token: ~45 GB permanently hot on
a 64 GB machine. The streaming container has the same RSS and the same
budget, but 17418 evictions — its slots are constantly rewritten, so at any
instant its genuinely hot set is the trunk plus the few GB it is cycling,
and the rest can go cold.

**That refines §39's cliff in a way worth keeping: the cliff is not a
property of the budget number, it is a property of how much of the budget is
touched per token.** Two runs at 46.39 GB and within 1 GB of the same peak
RSS differ 2x, and the one that is *resident* is the slow one. "A paged
cache hit is slower than the disk read it replaced" is in CLAUDE.md already;
what is new is that a container can be built whose hit rate is 100% by
construction and which loses to streaming anyway.

### What it settles

- **k > 1 does not rescue the merge.** Quality is flat to k=8 and worse at
  k=16, on both models; on K3, k=16 (KL 4.705) is worse than k=1 (4.241).
- **Pruning beats merging everywhere**, and still does not work. The best
  static approximation of a top-8-of-256 layer found here is a fixed 16
  experts, at KL 2.200, which still says the capital of Italy is Paris.
- **Holding one working set resident is not automatically a win.** The
  configuration §53 named as the interesting unrun case turns out to be
  slower than the streaming engine it was meant to replace.

A method note. The k=16 container also exposed a budget bug: the resolver
recommended `floor + 3x a token's working set` without noticing that for a
merged container the working set *is* the whole bank, so it asked for
80.77 GB where 46.39 GB holds every expert byte. Now capped at the
container's total expert size; unmerged containers are unaffected (K3's
952 GB bank against a 3x working set of 52 GB, Kimi-Linear's 16.5 against
1.6).

Not measured: whether the slowdown is macOS's compressor specifically (no
OS instrumentation was taken), the same comparison on Linux or on a machine
where 45 GB is not 70% of RAM, k=16 with top_k < k and a rebuilt 16-row
router, and whether any of these containers fine-tunes back to usefulness.

## 55. §54's speed claim was measured wrong, and k=16 is faster (2026-08-09)

§54 reported that K3 at k=16 runs at 0.27 tok/s against unmerged K3's 0.52,
called it "slower rather than faster", and built an explanation on top of it
about resident memory. **The measurement was taken with one process per arm,
at the end of a session that had just written 80 GB of containers and read
several terabytes. Re-run with `tests/sweep.c` — one load, arms interleaved,
two repeats — it inverts.**

| container | cache | slots | tok/s (rep 1, rep 2) | hit | GB read |
|---|---|---|---|---|---|
| K3 | 4096 | 346 | 0.579, 0.600 | 29.9% | 291.7 |
| K3 | 9000 | 760 | 0.597, 0.623 | 34.3% | 260.7 |
| K3 | 17736 | 1498 | 0.595, 0.612 | 41.2% | 219.9 |
| k3-k16 | 4096 | 346 | 0.506, 0.508 | **0.0%** | 272.1 |
| k3-k16 | 9000 | 760 | 0.455, 0.503 | **0.0%** | 272.1 |
| k3-k16 | 18000 | 1521 | **0.718, 0.689** | 93.8% | 17.0 |

**k = 16 with its whole bank resident is ~1.15x faster than streaming K3,
not 2x slower.** Both repeats agree, and the two containers were measured in
the same process state as each other.

The error is the one this repo has now made three times. `tests/sweep.c`
exists because of §32 and §33; its header says two arms in two processes are
two computers; §50's method notes record ~30% unexplained low outliers on
byte-identical work over long campaigns on this machine class. All of that
was read during the session that then produced the bad number anyway. The
rule that would have caught it is not "be careful", it is **never compare
two arms across two process lifetimes, and never at the end of a campaign** —
which is exactly what `sweep` is for and what it was not used for.

**What §54 got wrong, precisely**: the throughput table, the profile table
built from the same paired runs, and every sentence explaining why a
resident cache would be slow — including "the cliff is a property of how
much of the budget is touched per token", which was an inference from
`kda` being 1.6x slower in one process than in another. The K3 arms above
are flat from 346 to 1498 slots (0.579-0.623, spread 7%, repeats differing
2-4% at the same setting), so cache residency does not cost what §54 said
it costs. `WASTE_PURGEABLE` is off by default, so the volatile/nonvolatile
syscalls that were the other candidate never ran either.

**What §54 got right, and why it survives**: everything about quality. The
k-sweep, prune beating merge at every k, the U-shape, the orthogonality and
the norm law are all logit and weight measurements — deterministic
functions of the container, independent of machine state. Those tables
stand unchanged, and so does the conclusion that no static combination of
experts reproduces the routing.

### What the corrected number actually says

Removing **93% of the expert I/O buys 15%.** That is the useful figure, and
it independently confirms from the outside what `WASTE_PROFILE` says from
the inside: at a full working set of cache, expert I/O is 7-8% of a K3
decode step, and the step is bound by the VQ apply. A container engineered
so that expert reads are impossible gains about what the I/O bucket was
worth, and no more.

Which is also the answer to "would an accelerator help here". The bytes are
not the budget once the cache is full — the dependent gather chains are, and
`docs/BACKENDS.md` measured Metal losing on this engine (22% slower overall,
53 GB/s against the CPU's 195 GB/s on the one matvec it implements) and CUDA
finding ~84% of the step tracking core clock rather than DRAM.

One detail worth keeping from the small-cache arms. **k3-k16 hits 0.0% at
346 and 760 slots where K3 hits 29.9% and 34.3% at the same sizes.** A
merged container accesses its 1472 records in the same deterministic cycle
every token, which is LRU's worst case exactly; a routed container's access
order varies, so LFRU catches some reuse. Making the working set fixed makes
it degrade *harder* when it does not fit.

Not measured: whether the 1.15x holds at longer generations, on a machine
where 46 GB is not 72% of RAM, or against a k=16 container whose quality was
not already destroyed — which is the only reason none of this is a
recommendation.

## 56. Truncating top_k: 1.49x, and the knee is not where the KL says (2026-08-09)

§53-§55 established what cannot be done: no static combination of experts
survives, because what routing buys is the per-token exclusion. The
complement is the lever nobody had priced — **keep the selection, take fewer
of it.** `num_experts_per_token` is a manifest field; nothing is
re-converted, and every consumer adapts (`cfg_sane` bounds it,
`waste_plan_memory` sizes floor and recommendation from it, the scratch
shrinks, `moe_layer` renormalizes the survivors to one).

**`docs/EFFICIENCY.md` lever C does not already answer this.** C was refused
because ranks 9-16 carry 33.3% of the routed mass where it needed under a
tenth — but C *demoted* the tail to a cheaper precision, worth 16.7%, while
this drops it and takes the compute with it. And C was priced against a
profile where expert I/O was 54.8% of a step; it is now 8.3%, and §55
measured that removing 93% of it buys 15%. The bottleneck moved, so every
I/O-shaped lever needs repricing. Nobody had measured the **KL** of
truncation end to end — only §20's reconstruction error on the weights.

`tests/sweep.c` gained a `topk=` arm (lowering only: the scratch is sized at
load from the manifest's top_k). One load, arms interleaved, two repeats,
cache fixed at 17736 MB, K3:

| top_k | tok/s | speedup | KL vs top-16 | hit | GB/token | working set |
|---|---|---|---|---|---|---|
| 16 | 0.594 | 1.00x | — | 41.2% | 13.7 | 17.01 GiB |
| 12 | 0.696 | **1.17x** | **0.007** | 50.1% | 10.1 | 12.76 |
| 8 | 0.885 | **1.49x** | 0.037 | 62.8% | 6.9 | 8.50 |
| 6 | 0.931 | 1.57x | 0.046 | 71.0% | 5.6 | 6.38 |
| 4 | 1.055 | 1.78x | 0.118 | 78.5% | 4.8 | 4.25 |

Against the bar this repo applies to containers — §50 records VQ2R at KL
**0.0179** as damage real enough for `verify_container.py` to FAIL —
top-12 does *less* damage than a quantization the repo rejects, and top-8
twice as much while halving the working set.

**The knee is 16 to 8 and it is a regime change.** A token's working set
falls 17.01 to 8.50 GiB, and the 26.8 GiB this machine has above the floor
goes from holding 1.6 of them to 3.2 — the `floor + 3x` window §39 called
good and which K3 at top-16 could never reach on 64 GB. Below top-8 the
cache is already ample and only the linear compute saving is left, which is
why 8 to 6 buys almost nothing.

### The KL at one position does not certify an operating point

top-4 has KL 0.118 and the correct argmax, and it is **broken**. Greedy
continuation, 60 tokens, same prompt:

| top_k | what it writes |
|---|---|
| 16 | "1. **The 'Red Planet'**: … iron oxide … 2. … largest volcano: Olympus Mons …" |
| 8 | "1. **The 'Red Planet'** … iron oxide (rust) on its surface. 2. It has the largest volcano …" |
| 4 | "It looks like your message got cut off after '1.' …" then `<\|close\|>` forever |

top-8 reproduces top-16's content, facts and order with different
formatting. top-4 does not follow the prompt at all. **A single-position KL
measures the head of the distribution; the top-10 overlap falling 7, 6, 5, 5
was the signal that the tail was reordering, and the tail is what a
continuation walks into.** §50 used exactly this test to reject VQ2R; it is
the gate, and KL is only the screen.

### The lever moves the budget into the cliff

Reducing top_k lowers the recommendation from 80.77 GB to 54.77 GB, which
for the first time **fits** under the resolver's 7/8-of-RAM ceiling — so the
resolver takes it, and walks into the 46-52 GB cliff §39 measured:

| top-8 | tok/s | peak RSS | hit |
|---|---|---|---|
| default budget (54.77 GB) | **0.08** | 40.71 GB | 67% |
| `--budget 46G` | **0.77** | 47.29 GB | 62% |

**10x slower at the larger budget**, with lower RSS and a higher hit rate —
§39's signature exactly. The hazard was previously masked: K3 at top-16 asked
for 80.77 GB, never got it, and was forced down to a good value. Making the
recommendation reachable made it dangerous. The resolver's ceiling is too
generous on this machine and nothing in it knows about the cliff.

### The ceiling on this whole direction

MoE is 64.2% of a decode step; kda is 29.2% and mla 5.7%. So **2.8x is the
absolute limit of anything that only touches experts**, even if they were
free. top-4 at 1.78x has already removed 68% of the MoE time, and the rest
of the way down (top-2, top-1) is where the model breaks — Kimi-Linear
measures KL 1.327 and 2.568 there. Going past 2.8x means attacking kda and
mla, which is `docs/BACKENDS.md`'s "a different engine, not a backend".

Recommended operating point on this machine: **top-8 with an explicit
`--budget 46G`** — 1.49x, KL 0.037, continuation intact.

Not measured: longer generations (§50's note that hit rate rises with
length applies to both arms and may move the ratio), other prompts and
languages for the continuation gate, top-8 on Kimi-Linear as a deployment
rather than a control, and the natural composition — **fewer experts stored
more precisely**, where a 4-stage container at top-4 reads a quarter of the
records at +33% each and could buy back truncation error with per-expert
accuracy (§20's `err² = err3² + mass(S)·delta` prices the trade). That one
costs a re-conversion.

## 57. The resolver's headroom was an eighth because nobody had measured it (2026-08-09)

§56 found the default budget walking into §39's cliff the moment `top_k`
came down. The cause is one constant: the automatic budget stepped down
until it fit under **7/8** of usable RAM, and 7/8 of 64 GB is 56 GB — inside
the 46-52 GB band §39 had already measured as an eightfold collapse.

**The eighth was never measured.** It was a plausible margin, and it stayed
harmless for one reason: K3 at top-16 asks for 80.77 GB, cannot have it, and
the step-down lands on `floor + 1x` = 46.39 GB — the measured optimum,
reached for the wrong reason. Lower `top_k` to 8 and a token's working set
halves to 8.50 GiB, three multiples now fit under 56, and the default takes
54.77 GB.

| K3 at top-8, default budget | budget | tok/s | peak RSS | hit |
|---|---|---|---|---|
| before (7/8 ceiling) | 54.77 GB | **0.08** | 40.71 GB | 67% |
| after (3/4 ceiling) | 46.18 GB | **0.88** | 48.57 GB | 62% |

**11x, on the path a user gets by typing nothing.** Note which way the other
columns move: the slow one has the *higher* hit rate and the *lower* RSS,
which is what paging looks like from inside the process — the OS took the
cache, the engine re-read less because it was waiting more, and every
counter the engine keeps says it was doing better.

K3 at top-16 still resolves to 46.39 GB, unchanged. Kimi-Linear, whose
recommendation fits many times over, is untouched. A 128 GB machine still
gets the full `floor + 3x`.

Two smaller things came out of the same edit.

**`waste_memplan` now reports `working_set_bytes`.** The resolver used to
recover it as `(recommended - floor) / 3`, which stopped being true when §55
capped `recommended` at the container's whole expert set — on a merged
container the two are no longer three times apart. A quantity the rule is
built on should be reported, not re-derived from something that has since
grown a special case. `tests/run.sh` mirrors the rule and now reads the
field instead of recomputing it; `serve/engine.py`'s struct follows.

**The rule is now stated the same way in four places** — `waste.h`'s doc
comment, `docs/ENGINE.md`, `tests/run.sh`'s comment and the resolver itself.
All four said 7/8. Three of them were prose that had been true.

Not measured: where the cliff actually is on this machine (46 works, 52
does not, and nothing between them has been run), whether it is a fraction
of RAM at all or an absolute headroom the OS needs — 3/4 is the safe side of
the only two points there are, not a fitted value — and any of this on
Linux, where the page-cache behaviour under `O_DIRECT` is not the same.

## 58. §4D re-priced: batching is worth less now, not more (2026-08-09)

§56 and §57 both turned on the same thing — a measurement priced when
expert I/O was half a step, still being quoted after the cache made it 8%.
So §4D was re-run rather than assumed, and it comes out the other way from
the guess that prompted the re-run.

`EFFICIENCY.md` §1 measured chunked prefill at **1.62x** over sequential and
derived a 1.63x ceiling for any batching scheme, on the model that grouping
removes I/O and none of the compute. Same measurement today, 56-token
prompt, cache at a full working set:

| top_k | mode | tok/s | | GB read |
|---|---|---|---|---|
| 16 | sequential | 0.43 | | 860.8 |
| 16 | chunked | 0.69 | **1.60x** | 307.7 |
| 8 | sequential | 0.67 | | 449.0 |
| 8 | chunked | 0.86 | **1.28x** | 179.3 |

**§1 reproduces exactly at top-16 — 1.60x against its 1.62x — and collapses
to 1.28x at top-8.** Both levers take their gain from the same place, so
they overlap instead of composing: 0.43 to 0.86 is 2.0x end to end where the
product of the two would be 2.38x.

**And the ceiling holds for batching across independent streams too, which
§4D never separated from grouping within one.** The reason §1 gave is
structural and does not care where the tokens come from: `vq_apply` costs
one pass per (token, expert) pair, and the pair count is `T * K * 92`
however they are grouped. That work is 64.2% of a step, so as B grows the
per-token cost tends to it and no batching scheme beats **1/0.642 = 1.56x**.
Aggregate throughput on this engine tops out near 1.4 tok/s, not the tens
that a bytes-only argument suggests.

The bytes-only argument is worth naming because it is the one that misleads.
Per token this engine touches ~36.4 GB — 27.3 of resident trunk, 9.1 of
experts at top-8 — and at the 195 GB/s this machine reaches on a streaming
quantized matvec that would be 187 ms, i.e. 5.4 tok/s. It measures 0.88.
**The engine runs at 16.4% of its own memory bandwidth**, because the step
is dependent gather chains rather than streaming, which is the same thing
`docs/BACKENDS.md` found on CUDA (84% of the step tracks core clock, 23%
DRAM). Batching does not fix that; it amortizes bytes, and bytes are not
what the clock is going into.

### What that leaves, arithmetically

60 tok/s is 68x from 0.88. The factor splits cleanly and neither half is
optional:

| | factor | built? |
|---|---|---|
| bytes per token, 36.4 GB to 3.25 | **11.2x** | no |
| bandwidth efficiency, 16.4% to 100% | **6.1x** | no |
| top_k truncation | 1.49x | yes, §56 |
| batching | ≤1.56x, and it overlaps top_k | yes, and already default for prefill |

11.2 x 6.1 = 68. The two levers that exist are, between them, worth about
2x and are nearly spent; the whole remaining factor is in the two that do
not. And the byte half cannot come from the experts: they are 9.1 GB of the
36.4, so deleting them outright is 1.33x. **It has to come from the trunk,
which is 75% of the bytes and 100% unconditional.**

Which names the next measurement precisely, and it is a geometry question of
exactly the kind that killed the merge in §53 before anything was built:
**does K3's trunk have contextual sparsity at all** — is there a per-token
prediction that keeps most of a layer's output while reading a tenth of its
weights? If the answer is as flat as the merge alphas turned out to be, 60
tok/s on this machine is closed and the honest ceiling is about 2x from
here. Not run.

Method note, and it is the third time today: the guess that prompted this
re-run was that the new regime would make batching *more* valuable. It makes
it less. A regime change invalidates a measurement's conclusion in whichever
direction the arithmetic says, and the arithmetic has to be redone rather
than re-argued.

## 59. The trunk is not FFN, so contextual sparsity is aimed at 16% of it (2026-08-09)

§58 left one question standing: 60 tok/s needs 11.2x fewer bytes per token,
the experts are only a quarter of them, so it has to come from the trunk —
**does K3's trunk have contextual sparsity?** The literature that makes this
work (Deja Vu and its descendants) targets FFN blocks, where a ReLU-family
activation produces true zeros and a small predictor can say in advance
which neurons will fire.

The first thing to measure was not the activations. It was where the bytes
are:

| per token, top-8 | GB | share | if it vanished entirely |
|---|---|---|---|
| attention (MLA + KDA) | 18.92 | **49.9%** | 2.00x |
| routed experts | 9.10 | 24.0% | 1.32x |
| shared experts (FFN) | 6.27 | **16.5%** | 1.20x |
| routed latent projections | 2.44 | 6.4% | 1.07x |
| lm_head | 1.19 | 3.1% | 1.03x |

**Half of K3's per-token bytes are attention, and the FFN the technique
addresses is a sixth.** Nothing here, deleted outright, reaches 2x. A
generous compound — half the attention heads skippable *and* the FFN 90%
sparse — is **1.66x** against the 11.2x required.

### The sparsity is real, and it is worth 1.14x

Measured anyway, because "is it harvestable at all" is worth knowing. Real
hidden states from a K3 run (`WASTE_DUMP_HIDDEN`), shared-expert weights
decoded from `trunk.bin`, SiTU applied as the container declares it
(beta 4.0, linear_beta 25.0). Keeping only the largest intermediate channels
for that token and zeroing the rest, relative L2 of the layer's output:

| keep | L5 | L20 | L45 | L88 |
|---|---|---|---|---|
| 50% | 0.034 | 0.036 | 0.055 | 0.016 |
| 25% | 0.094 | 0.103 | 0.148 | 0.041 |
| 12.5% | 0.163 | 0.194 | 0.238 | 0.069 |
| 6.25% | 0.230 | 0.267 | 0.317 | 0.092 |

**A quarter of the channels carry 99% of the output energy** (rel L2 0.094 →
1 − 0.094² = 0.991), consistently across depth, and the last layers are
sparser than the middle ones. Against a flat activation, keeping a quarter
would leave error 0.87; this is 0.09. The concentration is not marginal.

And it is worth **1.14x**, because it applies to 16.5% of the bytes. Keeping
half rather than a quarter — which is what a 93-layer error budget would
more likely allow, since 9% per layer compounds — is **1.09x**.

### What it settles

**60 tok/s on this machine is closed.** Not for want of a trick: the bytes
are too evenly spread for any single-component saving to matter, the one
component the technique fits is a sixth of them, and the half that is
attention has no "read a tenth of the weights" formulation at all — every
output dimension of a projection depends on every input.

The honest ceiling from 0.88 tok/s, adding everything measured today:
top_k truncation 1.49x (taken), batching ≤1.56x and overlapping it (§58),
FFN sparsity 1.09-1.14x if built. **About 2x, total.** Past that needs
different hardware or a different model — the 6.1x of bandwidth efficiency
is real but it is a kernel rewrite of the whole forward pass, which is
`docs/BACKENDS.md`'s "a different engine, not a backend".

Two caveats on the sparsity numbers, both in the direction of making them
weaker rather than stronger. One token, four layers — the dump truncates per
token, so this is a single position measured at four depths. And existence
is not exploitability: skipping a column of `down_proj` requires knowing
which channels matter *before* computing the activation that reveals them,
which is the predictor half of Deja Vu and is not measured here at all.

Not measured: attention-head-level sparsity (the 49.9%), whether the
channel set is stable enough across tokens for a cheap predictor, and any
of this on a model whose activation is ReLU-family rather than SiTU.

## 60. The sparse channels are not the same channels (2026-08-09)

§59 measured that K3's shared-expert FFN is genuinely sparse per token — a
quarter of the intermediate channels carry 99% of the layer's output — and
left the half that decides whether it is usable: **is it the same quarter
next token?** Skipping a column of `down_proj` requires knowing which
channels matter *before* computing the activation that reveals them.

Eight tokens, each the last position of a growing prefix of one sentence
(the dump truncates per token, so this is eight runs), top 25% by magnitude:

| layer | Jaccard between tokens | random baseline | in all 8 | in at least one | a static set covers |
|---|---|---|---|---|---|
| 20 | **0.271** | 0.143 | 56 (4%) | 4289 (70%) | 57% |
| 45 | **0.206** | 0.143 | 25 (2%) | 4934 (80%) | 49% |
| 88 | **0.196** | 0.143 | 31 (2%) | 5083 (83%) | 48% |

For sets of size p drawn at random the expected Jaccard is `p/(2-p)` =
0.143. Measured is 0.20-0.27 — **1.4x to 1.9x chance, not 5x or 10x.**
There is no stable core: 2-4% of the set is common to all eight. A fixed set
chosen on the mean activation captures about half of any single token's.

And the number that kills the batched form of the idea too: **70-83% of all
channels appear in the top quarter of at least one of eight tokens.** A
group of eight would have to read four fifths of the matrix regardless.

The eight tokens are prefixes of the same sentence, which is the most
favourable case for stability that could have been chosen. It still comes
out close to noise.

### Which closes it

The shape is exactly §53's, from a different direction. There, the experts
were individually meaningful and mutually orthogonal, so no static
combination of them existed. Here the channels are individually meaningful
per token and their identity is nearly unstructured across tokens, so no
static selection of them exists either. **Both times the structure is real
and local, and has no regularity to exploit above it.**

So the ranking, complete, from 0.88 tok/s on this machine:

| | worth | state |
|---|---|---|
| top_k truncation to 8 | 1.49x | measured, taken |
| batching | ≤1.56x, overlaps the above | already default for prefill |
| FFN contextual sparsity | 1.09-1.14x if it worked | **and it does not: §60** |
| bandwidth efficiency 16.4% → 100% | 6.1x | a rewrite of the forward pass |
| bytes per token, 36.4 GB → 3.25 | 11.2x | no mechanism found |

**About 2x is the honest ceiling from here**, and 60 tok/s needs different
hardware or a different model. That is the answer to the question §58 posed,
arrived at without building anything.

Two caveats, both real. This is magnitude ranking of the output, and Deja
Vu's predictors are trained on the input rather than thresholded on the
output — a learned predictor could in principle find structure that
magnitude does not expose, since the activation is a deterministic function
of x. What the 2-4% common core and the 83% union say is that it would have
to find nearly all of it, which is the regime where the predictor costs what
it saves. And the input used is the post-layer residual from
`WASTE_DUMP_HIDDEN`, not the normalized state the engine hands the shared
expert — a proxy, chosen because the byte arithmetic in §59 caps the whole
direction at 1.14x regardless of how exactly it is measured.

Not measured: a trained predictor, attention-head-level sparsity (the 49.9%
of bytes §59 found there), and stability across genuinely unrelated prompts
rather than prefixes of one.

## 61. §50's flip condition was met from the other side (2026-08-13)

**Third-party, third contributor, and again not reproduced here.**
`mccoyspace` ran an incremental CUDA offload on an NVIDIA **GB10** — a
coherent unified-memory part, not a discrete card — on K3 and on
Kimi-Linear. Reported in issue #11; the raw archives and the operating
profile are linked from there. Nothing below was executed on this machine,
and the same caution applies as to §48 and §50.

§50 ended by naming the condition under which the discrete-card answer
flips: *"VQ3R's 17.77 GiB fits a 24 GB part comfortably — and then the H2D
term becomes a one-time load rather than 19.8 ms every token, which is the
whole deciding row."* That condition has now been met, by a route §50 did
not consider. Coherent unified memory does not make the H2D term a one-time
load; it means **there is no H2D term**. The deciding row is not reduced,
it is absent.

### What paid, and by how much

K3, against that machine's own CPU baseline:

| stage | K3 decode | contract |
|---|---:|---|
| CPU baseline before GPU work | 0.34-0.35 tok/s | deterministic CPU reference |
| KDA CUDA pilot | 0.476 tok/s | bounded logit tolerance, routes and tokens unchanged |
| accepted dense Q4 projections | 0.673 tok/s | same route/token contract |
| VQ3R gather, strict 64-token capture | **0.902 tok/s** | byte-identical logits, routes, tokens |
| qualified held-out default | **0.637 tok/s at 121.35 W** | 12 unseen 64-token prompts |

**Read the held-out row, not the strict one.** 0.637 tok/s is the figure
with an out-of-sample contract behind it, and this machine does 0.45-0.62
tok/s on the same model — so it clears the top of that band by about 3%,
which is the same order as the band's own width. The honest summary of the
K3 result is therefore not "2.6x" but "a coherent-memory GPU part reaches
roughly what an M5 Pro laptop reaches, having had to write CUDA to get
there." The 2.6x is real and it is measured against that host's own CPU,
which is the correct denominator for asking whether the GPU paid, and the
wrong one for asking whether the hardware buys anything.

### What it refutes

`docs/BACKENDS.md` said, as of 2026-08-04, that filling the `waste_backend`
slots with CUDA kernels *"would still reproduce the Metal result on
different silicon."* On this class of vehicle that is now false. The offload
was incremental and kernel-class — KDA, then dense projections, then the VQ
gather — with **attention state, routing and final expert accumulation left
on the CPU**. It is much closer to filling slots than to the whole-forward-
pass rewrite Metal's conclusion demanded, and it paid anyway.

The reason the Metal conclusion over-generalised is now visible: it
observed a *mechanism* — synchronous launch and round-trip per call on
several hundred small dependent matvecs — and stated a *shape*. Coherent
memory removes the round trip without removing the dependency. §50 had
already found the same seam from the discrete side (6.25x on the round trip,
killed instead by transfer bandwidth and VRAM capacity); this closes it.

### What it does not touch

**The discrete-card answer of §50 stands unchanged.** PCIe is still slower
than the host's own RAM, and a 16 GB card still does not hold a 16.5 GiB
expert bank. GB10 is not evidence about a 5060 Ti, and nothing here should
be read as reopening that case.

The **VRAM floor** in issue #11 also stands: 27.28 GB resident trunk and a
17.19 GiB working set is a hard 32 GB minimum for K3, realistically 48. GB10
clears it. Most parts people mean when they ask for CUDA do not.

### The design detail worth keeping regardless

CUDA produces a separate partial per selected expert; the **CPU applies the
router weights and reduces those partials in original router order.** That
is what made byte-identical logits reachable at all — zero changes across
10,649,600 logits and 5,888 routed rows — and keeping the router itself on
the CPU is what makes route invariance an *interpretable* gate rather than a
tolerance. §43's bar (an int8 table makes the engine discontinuous, so an
approximate match is not a match) was cleared rather than negotiated.

### VQ4P, including the negative half

§50 listed "the VQ4P 64-entry crossover" as not measured. It has been, on
the same vehicle, one container and identical runtime settings, 16-token
medians:

| path | tok/s |
|---|---:|
| scalar CPU | 1.471 |
| NEON CPU | 2.293 |
| CUDA, CPU-built LUT read coherently | 3.771 |
| CUDA, GPU-built and quantized LUT | **9.138** |

3.98x NEON — and **the mechanism is not a saved copy**. In the coherent
CPU-LUT arm, building the LUT on the CPU cost 1.205 s against 0.195 s of
CUDA apply, 619.76%, which fired a preregistered 10% trigger for a GPU LUT
builder. Moving construction on-device is the whole difference between
3.771 and 9.138. That is the same lesson as §42 from the other end: the
table is not free, and where it is built decides more than how it is
applied.

The negative result inside it is worth as much. **VQ4P is not a throughput
upgrade over VQ3R on that vehicle**, despite complete and exact coverage:
7.192 against 7.081 tok/s without Q0, 12.731 against 12.326 with it. VQ4P
followed a differing quantized trajectory that read 8.11% more expert bytes,
and its profiled CUDA VQ phase was 13.82% slower. Complete, exact, and not a
win — reported rather than buried, which is why it is here.

This is a Kimi-Linear result, not a K3 throughput claim, and no K3
conversion decision follows from it.

### What is still not measured

Whether any of it is maintainable. This project cannot execute a line of it:
no NVIDIA hardware, no way to regress-check a numerical contract, and — per
issue #36, filed the same week — a suite whose non-synthetic path does not
run in CI on any platform either. That is the open question about CUDA now,
and it is not a question about kernels.

## 62. The seam is a property of the engine, not of one vehicle (2026-08-23)

**Third-party, same contributor as §61, and again not reproduced here.**
`mccoyspace` extended the GB10 coherent-memory CUDA work from K3 and
Kimi-Linear to **Kimi-K2-Instruct**. Reported in issue #11; the report and
the experimental branch are linked from there. Nothing below ran on this
machine.

§61 closed by saying the open question about CUDA *"is not a question about
kernels"* — it was whether any of it is maintainable, and underneath that,
whether the arrangement was about the engine's structure or about one
model. This is the entry that answers the second half, and the answer
decided the first.

### K2 has no KDA layers, and that is the finding

Both models the GB10 work had used have KDA, and §61's offload arc *began*
with a KDA pilot. K2 is 61 MLA layers and no KDA at all, with a different
expert geometry — 384 experts top-8 against K3's 896 top-16.

What that composition needed was **a geometry allowlist, not a second
implementation.** The existing dense and VQ3R machinery was reused
unchanged; the patch adds exact K2 qualification and permits the zero-KDA
composition, while continuing to reject unknown models and VQ formats.

A seam that holds across {KDA, top-16-of-896} and {no KDA, top-8-of-384} is
a seam about the engine rather than about the vehicle. That is worth more
than the speedup below, and it is what the maintenance decision turned on.

### The numbers, and what can be checked from here

| | K2 on GB10 |
|---|---:|
| CPU fallback, mean | **1.223 tok/s** |
| CPU fallback, best | 1.227 tok/s |
| CUDA, mean | **2.748 tok/s** |
| CUDA, best | 3.001 tok/s |
| speedup | 2.25x |

Nothing here could verify a throughput figure — `README.md` says in as many
words that nobody on this project has a K2 container. What could be checked
is whether this is the same model this project validated for correctness in
0.6.8, and it is: 61 layers, 384 experts top-8, 1.026 T total against the
1.03 T recorded there, and **31.69 B active per token, agreeing to the
digit** with @fab2s's independent conversion on different hardware. Two
conversions converging on the active count is what would have caught a
neighbouring shape.

**So the CPU row is the more useful half of the result.** 1.223 tok/s is
the first K2 throughput figure this project has from anyone, and it is the
only K2 denominator that exists. It is an ARM CPU on a GB10, not a laptop,
so it is not a substitute for one either — which is why it is here and not
in `README.md`, under the same rule as §48, §50 and §61.

**The reading that inverts from §61.** For K3, 2.6x was the right
denominator for *"did the GPU pay on this host"* and the wrong one for
*"does this hardware buy anything over a laptop"*, because 0.45-0.62 tok/s
is published for K3 on an M5 Pro. For K2 the second reading is simply
unavailable: there is no laptop number to hold 2.748 against. The 2.25x is
uncontested because nothing exists to contest it, which is a reason to want
a K2 container here — not a reason to doubt it.

**A caveat on the mean.** It is two CUDA runs per prompt family across
three families, against one CPU reference each, 64 tokens per run. That is
a sound shape for a qualification and a thin one for a throughput figure:
issues #37 and #44 both report machines where the first repeat of an arm is
reproducibly the slowest, on byte-identical work, and §50 hit a third that
produced occasional 30% low outliers with no cause identified. Read 2.25x
as indicative.

### The contract, per stage rather than in aggregate

Across 195 causal comparisons: generated tokens, argmax, top-10 ordering
and ordered routes unchanged, with zero route changes and zero fallbacks.
Within that, the two paths hold **different contracts**, and the
distinction is load-bearing:

| path | contract |
|---|---|
| VQ3R expert apply | **byte-exact** |
| dense fast path | bounded, worst max abs logit diff `4.9114e-5` |

Both are right and neither should be read onto the other. §43's argument is
that an int8 lookup table makes the engine *discontinuous* — one entry
rounding the other way propagates rather than averaging out — so on the VQ
path an approximate match is not a match. That is the path that is
byte-exact. The dense projections are ordinary floating-point, where a
bounded difference is a legitimate contract and not a weakened one.
Summarising the pair as "byte-identical" would carry the strong claim onto
the weaker half; summarising it as "within tolerance" would throw away the
half that had to be exact.

A 4,608-token soak over 24 requests and four deterministic cycles produced
identical SHA-256 hashes on repeated outputs, with no request failures,
restarts, swap reads or writes, CUDA errors, or post-warm RSS growth;
maximum GPU temperature 67 C.

### What it does not establish

- **Not a discrete-card result.** §50 stands untouched: PCIe is still
  slower than the host's own RAM. GB10 has no H2D term to pay.
- **VQ4P is rejected on K2**, and unqualified shapes fail closed. The path
  refuses unless the complete K2 fingerprint and VQ3R geometry match.
  That is the right instinct and the reason an out-of-tree accelerator is
  safe to have at all: it cannot silently engage on a shape nobody
  qualified.
- **`/v1/chat/completions` is unavailable** on that branch.
- It remains an experimental-branch result, not an upstream-supported
  profile, and this repository has executed none of it.

### The decision this settled

Issue #11's item 3 — *what a maintainable arrangement looks like* — was the
question left after §61, and it is now answered:

1. **The implementation stays out of tree.** `src/cuda.cu` does not exist
   here and `WASTE_ENABLE_CUDA=1` still refuses.
2. **The seam is upstream and lives in `waste.h`**, with its shape designed
   from a review branch rather than agreed in the abstract. Private symbols
   a fork reaches past would be the permanent private fork this arrangement
   exists to avoid.
3. **A named hardware owner runs the real-model contract per adopted
   release**, and **a release where that validation did not run carries no
   current qualification claim.** A release that says "not validated this
   cycle" is better than one carrying a stale claim.

The reason is in this section rather than in §61: a backend that needed an
allowlist and not a second implementation to reach a model with an entire
kernel class missing is a backend whose interface can be reviewed here,
even though its kernels cannot be run here.

**What has not changed.** #36's larger point stands — no job on any
platform runs the suite against a real container. Two pieces closed this
week (`diskbench` is built by CI natively and cross since cf3ecc6/0016b1c;
a missing tool reports SKIP rather than accusing the engine since #42), but
the real-container gap is exactly the one that matters for a numerical
contract, and it is open. That is the argument for carrying a seam and not
kernels: the seam is something this project can review and keep honest.

## 63. Gate 7: §39's window is a four-token window (2026-08-23)

**Third-party, and the first Gate 7 data of any kind.** `Lrrr908` ran the
grid on a 128 GB Ryzen AI Max+ 395 (Strix Halo, Zen 5), K3, `tests/sweep.c`,
one process, interleaved arms, 200 generated tokens and 3 repeats per arm,
on commit 2056d44f. Reported in issue #37. Not reproduced here — this
project's machine is 64 GB and cannot select half the arms below.

### The kill criterion fired

Gate 7 was posed with the criterion declared in advance: *if at 200+ tokens
a 3–4 GB cache is no longer within ~10% of `floor + 1x`, the rule stands as
written and §39 becomes a note about short sessions.*

| cache | slots | median tok/s | vs `floor + 1x` |
|---|---:|---:|---:|
| 3072 MiB | 259 | 0.082 | 72.6% |
| 3400 MiB | 287 | 0.073 | **64.6%** |
| 4096 MiB | 346 | 0.087 | 77.0% |
| 17606 MiB (`floor + 1x`) | 1487 | 0.113 | — |
| 17740 MiB | 1499 | 0.114 | 100.9% |
| 35212 MiB (`floor + 2x`) | 2975 | 0.136 | 120.4% |
| 52818 MiB (`floor + 3x`) | 4463 | 0.154 | 136.3% |

23–35% below, against a threshold of 10%. **So the quantum stands.** The
budget resolver keeps stepping in whole multiples of one token's working
set, no code changes, and §39's *"a 3.32 GB cache is within 10% of a
17.32 GB one"* is now a statement about four-token sessions specifically.

§39 named its own failure mode in the sentence that opened this gate —
*"four generated tokens is exactly the length that flatters a small cache,
and cross-token reuse is what a large one buys."* That was a prediction.
It is now a measurement.

### The mechanism, from the counter rather than the clock

The comparison confounds two things: 200 tokens against §39's four, **and**
a Strix Halo with a Kingston NVMe against an M5 Pro. No 4-token arm was run
on that host, so nothing in the dataset separates them.

It survives because the deciding quantity need not be throughput. Hit rate
falls out of the routing trace and the cache policy; it does not care what
the disk or the cores are doing, and both runs report it stable across
repeats. It is the one column that can be set beside a different machine:

| slots | §39, 4 tokens | Gate 7, 200 tokens |
|---:|---:|---:|
| 287 | 29.1% | 29.7% |
| ~1490–1500 | 36.2% | 41.0 / 41.2% |
| 2537 | 41.3% | — |
| 2975 | — | 49.8% |

**The small cache gains 0.6 points and the large one gains five.** At 287
slots essentially every hit is the lookahead, which works within a token and
therefore cannot know how many follow; everything above that is cross-token
reuse, which is exactly what four tokens cannot exhibit. Put most sharply:
**the 200-token run reaches at 1,499 slots the hit rate the 4-token run
needed 2,537 slots to reach.**

**Caveat, because it matters more than the agreement does:** the two runs
use different prompts — a 13-token code prompt against §39's id list — so
the routing traces differ and these are not the same experiment. Read the
shape, not the digits. It corroborates a mechanism from a quantity the host
cannot move; it is not a controlled length sweep, and nobody has run one.

### What only a >64 GB host could produce

The resolver takes the largest of `floor + 3x/2x/1x` that fits under **3/4**
of usable RAM. On the 64 GB machine here that is always `floor + 1x`, so
`floor + 2x` and `floor + 3x` had never been selected by anything.

(Three quarters, not the seven eighths this gate was written against and
which issue #37 quoted on both sides. §57 moved it on 2026-08-09 and
`src/waste.c` has read `phys - phys / 4` since. It does not change any arm
above: `floor + 3x` is 80.77 GB and clears either ceiling on a 128 GB host.)

They are smooth. tok/s rises and GB read falls **monotonically across the
entire grid**, hit rate climbing to 56.6% at 4,463 slots with no knee and no
inversion. That is new information rather than a replication.

### "No cliff observed" is correct, and narrower than it reads

§16's eightfold collapse is a **fraction of physical RAM**, not a cache
size. On the 64 GB machine it sits between the 46 GB arm (~72% of RAM,
0.63 tok/s, healthy) and the 52 GB arm (~81%, 0.07 tok/s, dead).

Gate 7's largest arm peaked at **79.3 GiB on a 128 GB host, about 66%** —
below the fraction that was still healthy here. So the monotone curve is
what should have happened, and it is **not** evidence that the cliff is
absent at 128 GB; it is evidence that the grid stopped short of it. The arm
that would test it is roughly `cache=70000`, which puts total RSS near 81%
of that host's RAM. Not run.

Both outcomes would be worth having. If it collapses, §16 is a fraction and
travels. If it does not, the fraction is wrong and §57's headroom — an
quarter, and an eighth before §57 went looking — is holding back machines
that do not need it.

### Two smaller things this turned up

**The first repeat of an arm is reproducibly the slowest, on a third
machine.** Both smallest arms have an r1 off their own r2/r3 (0.091 against
0.066/0.082; 0.054 against 0.073/0.075). Issue #44 reports the identical
shape on a Zen 2 Windows box and cites this one as the second instance.
Three hosts, three vendors, two operating systems — past the point where it
should be filed as each reporter's local noise. It is why the table above is
medians.

**`waste plan`'s working set is computed over 93 layers, not 92.** It is
`rec × top_k × layers` with the total layer count, including the dense layer
that has no bank, so a reader computing `FORMAT.md`'s record size × 1,472
gets 17.01 GiB and does not match the 17.19 GiB printed at them. The
arithmetic is deliberate and `src/waste.c` says so — *"about 1% loose on K3,
in the direction that recommends slightly more rather than less, and
changing it would move a figure `tests/run.sh` asserts."* The gap is
documentation, not arithmetic: the explanation sits where the number is
computed and the person who trips on it is reading a terminal.

### Read the shape, not the absolutes

Gate 7's throughput runs 0.05–0.16 tok/s against 0.45–0.62 here, on a
Kingston NVMe serving every expert miss. The ratios and the curve are the
contribution; the absolute figures are a different machine and should not be
set against this project's.

## 64. Gate 1: two K3 opens do not degrade, they take the machine (2026-08-24)

Issue #31 asked for the collapse to be reproduced deliberately rather than
inferred from a sample that was being used to argue something else. It was,
on the 64 GB laptop this project develops on, and **the gate has an answer
that is not a throughput number: neither process ever generated a token.**

### The baseline, which is the only throughput figure here

One auto-budget K3 open, 30 generated tokens, `waste bench`:

| | |
|---|---:|
| decode | **0.39 tok/s** (2556 ms/token) |
| peak RSS | **45.39 GB** against a resolved 46.39 GB |
| expert cache | 17.56 GB |
| hit | 30.9% — 17,393 hit / 38,868 miss |
| read | 573.26 GB total, 19.109 GB/token |
| wall, load included | 79.9 s |

The hit rate is low because 30 tokens is a short session and §63 is the
reason: at 200 tokens the same cache size measures around 41%, and the
difference is cross-token reuse that has not had time to accumulate. So this
denominator is *pessimistic*, which matters only for a ratio that in the end
could not be computed.

### What the pair did

Two of those, started together, same container, no `--budget` on either. The
resolver is deterministic and machine-wide, so both selected the same
46.39 GB: **92.78 GB of intent on 68.72 GB of RAM, 135%.**

At 93 seconds — by which point the single-process run had finished
everything — neither had reached decode:

```
PID    ELAPSED  %CPU     RSS      COMMAND
16459  01:33     87.9    9.4 GB   ./waste bench k3.waste -n 30
16460  01:33    211.6    9.9 GB   ./waste bench k3.waste -n 30
```

9.4 and 9.9 GB resident against 27.28 GB of trunk each: both still loading.
Swap went from 15.06 GB used to 38.50 GB used between two samples minutes
apart, the allocation growing 15.36 → 39.94 GB as the OS tried to keep up.

Shortly after, the machine stopped responding and needed a power cycle.
**Zero tokens on either side, no partial result, and the scratchpad holding
the run's output was on `/private/tmp` and did not survive the reboot.**

### The finding

§16 priced the neighbourhood at "8x slower", and that framing turns out to
understate this configuration by the wrong kind of margin. 8x was one process
sitting near the paging cliff. Two processes past it is not slow — it is a
**liveness failure that arrives before generation starts**, in the trunk load,
where the resident set is being touched for the first time and there is
nothing optional left to evict.

That kills the null hypothesis outright. *"Do nothing, and say so — the engine
already prints what it chose; a host that runs two of them can read both
lines"* assumes there is a host left to read them on.

### The arithmetic that should have been done first, and settles gate 3

| | | share of RAM |
|---|---:|---:|
| physical | 68.72 GB | |
| resolver ceiling, 3/4 (`src/waste.c`) | 51.54 GB | 75% |
| one automatic open | 46.39 GB | 68% |
| **two automatic opens** | **92.78 GB** | **135%** |
| **two opens at the absolute floor** | **58.38 GB** | **85%** |

The last row is the one that matters and it needed no experiment. K3's floor
is 29.19 GB, almost all of it the 27.28 GB resident trunk, and it is not
negotiable — it is the memory the engine refuses to open without. **Two of
them exceed the ceiling this machine applies to one**, and land at 85% of
physical, above the ~81% that §39 measured as already dead and well past the
~72% that was still healthy.

So on this machine **no budget policy makes two K3 opens coexist.** A
reservation ledger would drive the second process to its floor and the pair
would still oversubscribe; trimming the multiplier has the same ceiling. Both
candidates in #31's gate 3 are mechanisms for dividing the cache *above* the
floor, and the floor alone is what busts the budget.

What is left is therefore not a budget mechanism at all:

- **Refuse the second open** — which #31 lists as a non-goal in as many words
  ("anything that refuses an open that would otherwise have succeeded"), and
  which is what `exclusive_open` already does opt-in since #29, on the key
  #31 correctly criticised.
- **Stop duplicating the trunk** — one resident copy shared between contexts,
  which is a different engine shape (a server, or shared mappings) and not a
  resolver change.

A ledger still covers the cases where the floors *do* fit and only the caches
contend — K3 beside Kimi-Linear, or two Kimi-Linear opens on a 128 GB host.
It does not cover the case this issue opened on.

### What this does not establish

- **No tok/s ratio, because there is none.** Nothing here says how much
  slower two K3 processes are; it says they do not get that far.
- **macOS only.** Linux with an OOM killer would have shot one process rather
  than freezing, and the surviving one might well have completed. That is a
  different result and worth having from anyone who can produce it safely.
- **Gate 2 as written must not be run on this machine.** *"Same pair, second
  process pinned to `--budget` floor by hand"* is 46.39 + 29.19 = 75.58 GB,
  110% of RAM — the same experiment with a smaller number. The table above
  answers it by arithmetic instead.

### Operationally

This experiment costs a reboot and should be run in a memory-limited cgroup
or on a machine with room, not on a bare laptop. The estimate that preceded
it — "20-30 minutes of paging, with a real chance the OS starts killing
things" — was wrong in kind rather than in degree: macOS grew swap to 40 GB
and froze instead of killing anything.

## 65. GLM-5.3-Flash is this engine with three things added (2026-08-27)

`zai-org/GLM-5.3-Flash` was published the day before this was written: 313 B
parameters, 328 GB of fp8, `Glm5NextForConditionalGeneration`. The question
was how much of it this engine already runs. The answer turned out to be
most of it, and the interesting part is which of the differences were
*loud* and which were silent.

### What was already there

Nothing had to be written for any of this. The layer mix is 34 KDA and 11
MLA, one full-attention layer in four — Kimi-Linear's pattern at K3's
scale. The KDA is the same recurrence with the same parameterization:
`f_a`/`f_b` forget gate, per-head `A_log`, `gate_lower_bound` -5.0, short
conv 4, `g_a`/`g_b` output gate, gated RMSNorm. The router is DeepSeek's,
which is K3's: sigmoid scores, an `e_score_correction_bias` for selection
only, renormalized top-8, `routed_scaling_factor` 2.5. The tensors are
named the way K2 and K3's are, the config nests under `text_config` with a
`language_model.` prefix exactly as K3's does, and the weights are fp8 with
block scales, which the converter has read since K2.

Even the MLA needed nothing: `qk_rope_head_dim` is **0** and `mla_use_nope`
is set, so `mla_layer` runs with a zero-width rope slice and the absorption
is unchanged. GLM's own positional signal is in the KDA layers and in the
indexer's rope, not in MLA at all.

### What was new, and what it cost

| | where | resident cost |
|---|---|---|
| mHC, 4 residual streams | the layer loop, both sites | 35 B params, 8-bit |
| clamped SwiGLU, limit 10 | seven activation sites | none |
| DSA k-pool indexer | `mla_layer` | 128 B/token/layer |

The third is the one worth recording. DeepSeek Sparse Attention here scores
*pools* of four adjacent tokens rather than tokens, keeps
`index_topk / index_kpool` = 512 of them, and always appends the tail that
has not filled a pool. Two consequences fell out of writing it:

- **A pool's compressed key is computed once and is 32x smaller than the
  keys it summarizes.** One 128-wide vector per four tokens per
  full-attention layer, appended when the pool's last token arrives. The
  per-step state is that array plus a four-slot rolling buffer. Against the
  latents already cached (512 floats per token per layer) it is a
  fourteenth. The obvious implementation — cache the raw keys and pool them
  at selection time — would have cost 32x more and recomputed the same
  softmax every step.

- **Below `index_topk` tokens the sparse path *is* the dense path**, not an
  approximation of it. With no more complete pools than the query may keep,
  the selection is every visible token. So the branch costs nothing at
  short context and the selection arithmetic appears exactly where the
  saving does. `dsa_select` returns -1 for it and the head loop takes the
  path it always took.

### The two that were silent, and one that was found by checking

Loud failures need no discipline. These three did:

**`kda_layers` is 0-based on GLM and 1-based in a WASTE manifest.** Copied
through it puts KDA on the wrong layers. Every tensor is found, every shape
checks out, `cfg_sane` passes, the container loads, and the model answers
noise. The converter rebuilds the list from `layer_types` instead, which
states the same thing without an origin convention.

**`eos_token_id` is a list of three.** Read as an integer it stops on
nothing.

Neither is visible to the oracle diff, because the oracle reads the same
manifest and would be wrong in the same direction. That is the general
shape of a converter bug and the reason `tests/test_convert_glm.py` checks
against what the *release* says rather than against the container.

**The pre-tokenizer has no Han branch.** This one was not deduced, it was
measured, and it is the concrete result of this section. Kimi's pattern
starts with `[\p{Han}]+`; GLM's does not, so on GLM a Han run and the Latin
run touching it are one pre-token where on Kimi they are two. Sixteen
tokens in GLM's vocabulary span that boundary and they are ordinary words:

| | with the Han branch | GLM, and the release |
|---|---|---|
| `A股` | `32 98963` | `111321` |
| `维生素C` | `103261 34` | `121569` |
| `C罗` | `34 99209` | `126152` |
| `QQ音乐` | `47724 99908` | `126724` |

A tokenizer that is right on 22 English and CJK strings and wrong on `A股`
is the failure mode this family keeps producing: correct on everything the
corpus happens to contain. The corpus now contains it, the container states
which pattern it wants in `tokenizer_han_split`, and `tools/tokdiff.py`
compares against whichever reference the source actually ships — tiktoken
for a rank file, `tokenizers` for a `tokenizer.json`. 21 of 21 identical
against the real GLM vocabulary.

A fourth thing was expected to be silent and was not: the merge list.
Re-encoding `tokenizer.json` into a tiktoken rank file only reproduces the
release's encoder if the merge list is ordered by the id of what each merge
produces — merge-by-rank and merge-by-list-position are otherwise two
encoders sharing one vocabulary. GLM's is so ordered (0 descents in 321648
merges), and `hf_tokenizer.py` refuses rather than approximates when a
release's is not. Its first version demanded something stricter — that both
halves of every merge rank below the result — and rejected this tokenizer
over 56373 merges that are simply redundant spellings (`Ġ`+`th` beside
`Ġt`+`h`, both producing `Ġth`). The check was wrong, not the tokenizer;
what matters is the order of the results, not the ranks of the inputs.

### The size, which is why this is worth having

Arithmetic from `config.json`, not a conversion:

| | GLM-5.3-Flash | K3 |
|---|---:|---:|
| trunk, resident | ~4.8 GiB | 27.3 GiB |
| experts at VQ3R | ~106.5 GiB | 982 GiB |
| one token's working set | ~3.2 GB | ~17 GB |

On the 64 GB machine §64 measured K3 at 0.39 tok/s on, GLM's trunk leaves
room to cache something on the order of 40% of its entire expert set, where
K3 gets a working set and a half. §63's four-token window and §39's
hit-rate curve both say what to expect from that, and neither has been run
here — no GLM container has been converted yet. Everything in this section
that is not from `config.json` was measured on a synthetic container at
1/32 scale, whose shapes are the ones the engine branches on and whose
weights are noise.

## 66. The budget stopped three working sets short of the machine (2026-08-27)

The automatic budget walked a ladder whose top rung was `floor + 3x` a
token's working set. That is what `recommended_bytes` means — "worth
having, without knowing the machine" — and it is the wrong ceiling once the
machine is known. On this 64 GB laptop Kimi-Linear's **entire** expert set
is 17.17 GB and the default gave it **1.65 GB** of cache; K3 on a 256 GB
host would take the same 51.6 GB of cache it takes here.

The real ceiling is the container's own bank: a cache that holds every
expert never reads one twice, and a byte past that is a slot nothing will
ever fill. So the ladder now starts at however many working sets cover
`bank_bytes` and walks down, and the cache never exceeds the bank. Nothing
about §39's and §56's cliff changes, because the cap that guards it does
not — three quarters of what the process may use, with the cliff above it —
and an unfilled slot is address space, not memory.

Three things came out of that, and only the first was the plan.

### 1. The RAM, on a container that fits

Kimi-Linear, `waste bench`, defaults, this machine:

| | before | after |
|---|---:|---:|
| resolved budget | 3.07 GB | **18.48 GB** |
| expert cache | 1.65 GB | 17.20 GB |
| hit rate, 200 tok | 70.7% | **96.3%** |
| bytes read, 200 tok | 66.27 GB | **17.75 GB** |
| tok/s, 64 tok | 11.13 | **12.60** |
| tok/s, 200 tok | 12.67 | **14.81** |

K3 is untouched — 0.3251 tok/s before and after, byte for byte the same
465 GB read — because its bank is 962.83 GB and no rung above `floor + 1x`
fits under the cap. The change is only visible where the container is
smaller than the machine, which is exactly where it was meant to be.

### 2. Residency is what WASTE_XPAR was really asking about

§44 measured one task per routed expert at 1.18x on Kimi-Linear and a
regression on K3, and the reason it gave was already the answer: holding K
records before doing any arithmetic is *a barrier against the read-ahead*.
It is free parallelism when the records are in RAM and a stall when they
are on their way. The flag was standing in for a fact the cache can state.

So `moe_layer` asks it, per layer and per token, in `top_k` hash lookups
(`waste_ecache_resident_all`). Measured, same build, same container:

| | XPAR=0 | XPAR=1 | default (asks the cache) |
|---|---:|---:|---:|
| Kimi-Linear, 200 tok | 12.18 | 14.51 | **14.25** |
| K3, 20 tok | 0.3251 | 0.1729 | **0.3251** |

The automatic choice takes 89% of the forced win where forcing is right and
**all** of the loss where it is wrong. `WASTE_XPAR=0/1` still overrides.

What makes this admissible at all is that the two paths are bit-identical —
both sum per-expert results in `j` order — and that is now asserted in
`tests/run.sh` rather than believed. A scheduling choice made from cache
state that also moved the numbers would make an answer depend on how warm
the cache happened to be, which is unreproducible by construction.

### 3. A cache that can hold everything should not discover it by missing

With the bank resident, 200 tokens still read 13.2 GB of it as **3443
separate demand misses**. The slots were already allocated and the reads
were going to happen; what the demand stream adds is that they happen late,
one at a time, and that a layer is not fully resident — and so cannot take
the path above — until every one of its experts has been asked for once.

A background thread now walks the banks in file order and fills empty slots
(`waste_ecache_admit`: never evicts, never counts a hit or a miss, counts
its bytes). It starts only when every record fits, because below that
"put anything in an empty slot" competes with the demand stream for slots
it is about to need, and LFRU is a better judge than file order.

It is a win at every length measured, including the short ones that read
three times the bytes:

| tokens | fill off | fill on |
|---:|---:|---:|
| 16 | 7.19 | **7.24** |
| 32 | 9.64 | **10.02** |
| 64 | 11.79 | **12.50** |
| 200 | 14.21 | **14.98** |

### What this does not do, and one thing it is not

**It is not throughput on a machine that was already fast enough.** On this
NVMe the read-ahead was hiding almost all of Kimi-Linear's I/O: expert I/O
was 0.8% of a step at a 19 GB cache and the whole gain above comes from the
scheduling change residency enables, not from the reads themselves. The
reads matter where the disk is slower — the tested USB enclosure at
0.94 GB/s would spend 70 s on 66 GB and 19 s on 17.7 — and on K3, where
17 GB a token cannot be hidden at any queue depth.

**Where it should pay is a machine this project does not have.** GLM-5.3-
Flash (§65) is ~4.8 GiB of trunk and ~106.5 GiB of experts: on a 128 or
256 GB host the new ladder makes it fully resident and the fill reads it in
once, where the old one gave it a 9.6 GB cache. Nobody here can run that,
and this section says so rather than estimating it.

**An f32 trunk is not the next lever.** With the experts resident the
profile is 49.7% trunk matvec and 30.8% KDA, so the obvious thought is to
spend the spare RAM on the trunk too. Measured: `WASTE_Q8=0` on
Kimi-Linear is **12.74 tok/s against 15.02** — 8x the trunk RAM for a 15%
loss, because the quantized matvec is memory-bandwidth bound and dequantizing
makes it worse. The remaining time is arithmetic, and arithmetic is not
where more RAM helps.

## 67. Waking the pool cost 54 us, 150 times a token (2026-08-27)

With §66's expert set resident, a Kimi-Linear decode step profiles as 49.7%
trunk matvec and 30.8% KDA — arithmetic, not I/O. The obvious reading is
that the kernels are the wall. They are not, or not yet: **a third of the
dispatches in a step were spent waking threads.**

### The measurement that settles it

An empty kernel, dispatched in a loop, on this 18-core machine (6
performance, 12 efficiency):

| pool | empty dispatch, 4096 rows | over 128 rows |
|---|---:|---:|
| 18 threads (12 of them parked E-core workers) | **54.06 us** | 53.36 us |
| 6 threads, all performance | **1.63 us** | 1.38 us |

The row count does not matter, which is the whole point: this is not work,
it is the cost of starting one. Two causes, and they compound:

- Workers park on a condvar. Waking one is a scheduler round trip.
- The barrier at the end waits for the *slowest* participant, and on this
  machine that is an efficiency-core thread the scheduler takes tens of
  microseconds to run. It contributes nothing to a short job and delays
  every one of them.

A decode step dispatches about 150 times — nine matvecs per KDA layer, the
router, the shared experts, the recurrence, the VQ kernels. At 54 us that
is 8 ms of a 67 ms token.

### Two changes, and which one mattered

**Spin before parking.** Workers spin on the job word for a bounded number
of iterations before falling back to the condvar; the caller publishes with
a release store, the chunk cursor and the completion count became atomics,
and the condvars stayed for whoever did park. Worth about **2.4%** on its
own (14.46 -> 14.80) — because a full-pool dispatch still had to wake the
twelve parked E-core workers, and they still set the barrier.

**Do not reach for the slow group unless the job can hide the wake.** A
dispatch now goes to the fast group — which spins, so ~1.5 us — unless the
weights it touches exceed `WASTE_WIDE_MIN`, 4 MB by default. Swept on
Kimi-Linear, 150 tokens:

| threshold | tok/s |
|---|---:|
| 0 (fast group always) | 14.86 |
| 2 MB | 16.56 |
| **4 MB** | **16.62** |
| 6 MB | 15.89 |
| 8 MB | 15.84 |
| infinite (whole pool always, the old behaviour) | 14.91 |

A real optimum rather than a trend, which is what the mechanism predicts:
below it the dispatch dominates and the fast group wins, above it the work
dominates and the extra cores do. Kimi-Linear's q/k/v/o projections are
4.7 MB and sit just above the line; its low-rank KDA projections
(`f_a`, `g_a`, `b_proj`) are 37-262 KB and sit far below it.

Together: **14.41 -> 16.74 tok/s, +16.2%**, three runs each, spread 0.02
and 0.11.

### The inversion in §47 was this all along

§47 recorded that six threads beat eighteen on Kimi-Linear and that
eighteen beat six on K3, and CLAUDE.md warns not to carry the setting from
one model to the other. Measured again on this build:

| | before | after |
|---|---:|---:|
| 18 threads | 14.46 | **16.97** |
| 6 threads | 14.49 | 15.35 |

The inversion is gone, and eighteen is now the better answer on the model
that used to prefer six. Capping the pool was never buying "fewer, faster
cores"; it was buying "no parked threads to wake". Now that short jobs do
not wake them, the long jobs get to use them.

### What did not change, and what is still true

**K3 is neutral.** 0.3389 / 0.3166 / 0.2835 against 0.2955 / 0.3251 /
0.3233 — the same mean inside a +-9% run-to-run spread. It reads 465 GB
for 20 tokens; 8 ms a token of dispatch is not where its time goes. §66
said the same thing about residency, and for the same reason.

**Nothing about the numbers moved.** The split is by row, so the logits are
bit-identical across every combination of `WASTE_SPIN`, `WASTE_WIDE_MIN`
and `WASTE_THREADS` — checked, and now asserted in `tests/run.sh`.

**On a machine with one kind of core this is inert.** `n_fast` is 0 there,
`waste_pool_fast()` is the whole pool, and the threshold selects between
the pool and itself.

### One bug worth recording, because it looked like a data race

The first version seeded the job word with epoch 1 while workers start at
`seen = 0`. Every worker therefore woke immediately on creation, read `n`
and `chunk` before anything had written them, and decremented `active`
below zero — which the first real dispatch then inherited, so it could
return while its workers were still running. It showed up as two
ThreadSanitizer reports on `g_pool.n` and `g_pool.chunk` and read like a
missing barrier. It was not: it was a job that never existed. The pool is
now clean under TSan across the forward pass, chunked prefill, session
state, the ownership lock and every option combination above.

## 68. GLM-5.3-Flash, converted and run (2026-08-27)

§65 implemented GLM against a synthetic container at 1/32 scale and said,
in as many words, that the real conversion was the proof. It has now
happened. **The container loads, matches a PyTorch oracle to 2.4e-5, and
answers questions at 3.09 tok/s on a 64 GB laptop.**

    $ waste run ~/models/glm53.waste "The capital of Italy is" -n 20
    waste: no --budget, using 46.37 GB of 64.00 GB (expert cache 41.36 GB)
    The capital of Italy is Rome. Rome is the largest city in Italy and is
    known for its rich history, iconic landmarks such
    [20 tokens, 5.07 s, 3.94 tok/s | experts 5816 hit / 904 miss = 87%]

### What it cost

| | |
|---|---:|
| download, 62 shards | 306 GiB, ~2 h at 36-97 MB/s |
| conversion, `--jobs 3` | 42 expert layers at 158-163 s each, ~40 min |
| trunk | 5022 MB, 1246 tensors |
| experts | 42 x 2598 MB = 109 GB |
| container | **112 GB** against 328 GB of fp8 source |

The estimate §65 published from `config.json` alone — 4.8 GiB of trunk,
106.5 GiB of experts, 9.02 MiB per expert record — came out right: the
first converted layer was 2597.6 MiB for 288 experts, 9.02 MiB each,
before the rest had even downloaded.

### The two bugs, and the difference between them

**One was caught before the conversion and cost nothing.** GLM nests the
text model the other way round from K3 —
`model.language_model.layers.N` against `language_model.model.layers.N`,
and `lm_head` outside the wrapper rather than inside it — which no
`tensor_prefix` can reconcile. Found by listing every name and shape the
engine would demand for this config and checking them against the
checkpoint's own index: 1246 tensors, 0 missing, 0 mismatched *after* the
fix, all 1246 missing before it. Fifteen minutes of a partly-downloaded
checkpoint, against an hour of conversion and a container that would have
refused to open.

**One was not, and cost an hour.** `build_trunk` skipped any tensor whose
name ends in `_packed` or `_scale`, meaning the companions of a quantized
weight. GLM ships a real learned parameter called `hc_attn_scale` — 90 of
them, two per layer, the gains on the mHC mapping — and all 90 were
dropped. The conversion finished, the manifest published, and the load
then said `required tensor is missing: model.layers.0.hc_attn_scale`.

The check was a suffix on the wrong thing. Every companion in this family
is a `.weight` plus a suffix (`weight_packed`, `weight_scale`,
`weight_scale_inv`), and matching *that* keeps the parameter and drops the
companions — including 89 fp8 `weight_scale_inv` tensors that the old test
did not match and had been writing into K3-era trunks all along.

The general lesson is the one the first bug already illustrates: **a
converter's failures are cheap to find statically and expensive to find at
load.** The static check would have caught this one too, and did not,
because it asked "does the engine's list exist in the checkpoint" and not
"does the container contain the engine's list". Both directions now run.

### The numbers

`waste info`: **313.33 B parameters total, 16.74 B active per token**, 45
layers, 288 experts top-8, arch `glm5-next`.

`waste plan`: floor **5.14 GB**, one token's working set 3.17 GB, fully
resident 119.30 GB. On this machine the ladder from §66 resolves
**46.37 GB with a 41.36 GB expert cache** — thirteen working sets. The old
ladder stopped at three, i.e. 14.66 GB.

**Against the oracle**, 4 tokens, the real container:

| | |
|---|---:|
| max abs | 2.39e-4 |
| rel L2 | **2.41e-5** |
| argmax | 3 vs 3 |
| top-10 | identical, in the same order |

That is mHC, the clamped SwiGLU, the DSA indexer, KDA, MLA, the router and
the streaming path all agreeing with an independent PyTorch implementation
on 313 B real parameters. The DSA branch exercised here is the dense one —
`index_topk` is 2048 and the prompt is four tokens — so the sparse branch
remains verified on the synthetic container (§65) and not on this one.

**VQ3R holds on GLM's expert distribution.** Relative error on a real
`gate_proj`: **0.1951**, against the 19.59-19.77% `docs/K3.md` records for
K3 and the 0.195 of §50. Same recipe, same number, no retuning.

**Throughput**, `waste bench`, this 64 GB machine:

| | tok/s | hit | read |
|---|---:|---:|---:|
| 64 tokens | 2.40 | 87.7% | 72.7 GB |
| 200 tokens | **3.09** | 89.3% | 160.8 GB |
| 200 tokens, old §66 ladder (14.66 GB) | 3.16 | 71.4% | **481 GB** |

Five times K3's 0.45-0.62 tok/s, on a model that actually fits the
machine. And the same shape §66 found on Kimi-Linear: on this NVMe the
larger cache buys **three times less disk traffic at the same speed**,
because the read-ahead was already hiding the reads. On the 0.94 GB/s
enclosure `docs/K3.md` measures, 481 GB is 8.5 minutes of pure reading
against 2.9.

### Kimi K3 is untouched, and this is the check that says so

Not "the suite passes" — the logits. Same 16-token prompt, same container,
this build against `cbef892`, the commit this session started from:

    baseline seq vs new seq: max abs 0.000000

Two K3 checks do fail, and they fail identically at `cbef892`: chunked
prefill and the CPU backend each differ from the default path by
**rel L2 0.0176** (max abs 0.2858, argmax unchanged), against a 1e-3
threshold. The two alternatives agree with *each other* exactly, so what
differs is the default — the i8mm/SMLAL trunk kernels this branch added,
whose accuracy on K3 CLAUDE.md already records ("two kernels that agree
with the f32 path to 4e-5 on Kimi-Linear differ from each other by logit
rel L2 0.13"). Pre-existing, documented, and not this session's to
silently change.

Two more K3 "failures" were the suite reporting a missing prerequisite as
a defect, which this repo's rules forbid. `WASTE_REF_SRC` defaults to
Kimi-Linear's weights, so a K3 run round-tripped K3's container against
Kimi-Linear's safetensors and called it a converter bug; and the hotlist
check opens at a hardcoded `--budget 5G`, which is below K3's 29.19 GB
floor, so the engine refused, no cache line was printed, and that read as
"the hotlist did nothing". Both are SKIPs now, with the reason. With the
right source, K3 round-trips: 19.58-19.80% relative error, PASS.


## 69. What GLM needed to be usable, and what it did not (2026-08-27)

§68 converted GLM-5.3-Flash and showed it answering a question. That is not
the same as being usable, and the gap between the two turned out to be four
things — of which two were a format nobody had written down, one was a
verification, and one was a measurement that killed the work it was meant
to justify.

### The chat format could not be written down

`waste chat` on the fresh container did raw continuation, because a
container with no `chat.json` has no format and the CLI says so rather than
guessing. Writing one took three fields the four-string format did not
have, and each was needed because GLM's format cannot be expressed without
it:

| field | why |
|---|---|
| `prelude` | GLM opens every conversation `[gMASK]<sop>`, which belongs to no role |
| `stop` | a turn ends when the *next role marker* begins, so there is no suffix to close it with |
| `think` | the generation prompt is `<|assistant|><think>`; the model closes the channel itself |

The `stop` one is the instructive failure. Written without it, the format
loads and answers correctly and then **keeps going**:

    The capital of Italy is Rome.<|user|>What is the capital of Italy?
    Answer in one sentence.</think>The capital of Italy is Rome.<|user|>...

because the CLI took its stop from the assistant suffix, GLM has none, and
the container's `eos_token_id` is `<|endoftext|>` — which is right for a
raw continuation and is not what ends a chat turn. The release declares
three eos ids for exactly this reason and the container holds one.

With all three:

    > What is the capital of Italy? One sentence.
    The user is asking a simple factual question... </think>The capital of
    Italy is Rome.

### The reasoning channel was refused rather than parsed

`serve/chatfmt.py` said, in as many words, that a chat.json container has no
think markup and a request asking for one is refused. True of both Kimi
containers and false of GLM, whose channel is in its specials and whose
generation prompt always opens it. So `think` is now part of the format,
`PlainParser` splits on its close marker, and the server returns the two
separately:

    "content": "The capital of Italy is Rome.",
    "reasoning_content": "The user is asking a simple factual question..."

streaming included — `reasoning_content` deltas first, then `content`.
`reasoning_effort` maps to GLM's own spelling of it, a system turn
(`<|system|>Reasoning Effort: High`), through an optional `effort` field;
a format that does not say how to ask still refuses rather than dropping
it silently.

`thinking: false` is **refused** for GLM. Its template has no path that
leaves the channel closed, and answering with it closed puts a stray
`</think>` in the reply — measured, not assumed.

### The sparse attention branch, verified on real weights

§68 verified GLM against the oracle at four tokens, where `index_topk` is
2048 and the selection is therefore every visible token: the dense path.
The sparse one was verified only on a synthetic container at 1/32 scale.

Closing that needed the sparse branch reached on *real* weights, and a
2048-token oracle run is hours. So the container was cloned with symlinks
and a manifest whose `index_topk` is 16 — same 112 GB of weights, four
pools kept instead of 512 — which reaches the branch at 24 tokens.

**The selections are identical, 55 of 55**, at every (layer, position)
where the branch fires. Not "the logits are close": the engine's chosen
pools and the reference's chosen pools, compared directly through a
`WASTE_DUMP_DSA` trace on one side and the same line format on the other.
The scores agree to a median 8.4e-4 absolute, and the one 48% relative
outlier is a score of 0.0198 — cancellation, not disagreement.

The logit difference in that run is 1.95e-3, two orders above §68's
2.41e-5, and the control says sparsity is not why: **the same 24 tokens
through the dense path give 1.74e-3.** It is the KDA recurrence
accumulating over 24 steps in a different summation order, and it grows
with prompt length whether the attention is sparse or not.

And at the real setting, 2100 tokens: the branch fires on all 11 MLA
layers for positions 2051-2099, keeping 512 pools of 513 — 2048 pooled
positions plus the tail, of 2052 cached, which is what the config says it
should do. Against the same prompt with the selection disabled: rel L2
0.0909, argmax and top-10 identical. The selection is doing something and
it is not breaking the answer.

### Chunked prefill would have bought nothing, and here is why

GLM's prefill runs at decode speed — 266 tokens in 69 s — because
`waste_model_prefill` routes an mHC container through the per-token path.
The obvious fix is to teach the chunked path mHC. Measured first, on the
model that already has that path, warm cache:

| prompt | per token | chunked |
|---:|---:|---:|
| 64 tokens | 4063 ms | 4075 ms |
| 512 tokens | 27067 ms | 27231 ms |

Nothing, at either length. The profile says why: in a chunked prefill
**82.8% of the time is the VQ apply**, and the chunk expands nearly the
whole bank because 64 tokens at top-8 over 256 experts route almost
everywhere. Chunking trades disk reads for VQ decode, and on a machine
where the reads were already cached that is a pure loss.

So prefill is not slow because it is not chunked. It is slow because the
expert kernel costs the same per (token, expert) either way, and that is
§43's problem and not this one. The fallback GLM takes costs nothing
measurable, and building the chunked path for it would have been a day
spent on a 0.3% regression.

### What long context needs, and what it costs

| ctx | floor | state |
|---:|---:|---:|
| 4096 | 5.14 GB | 0.23 GB |
| 32768 | 5.79 GB | 0.87 GB |
| 131072 | **8.00 GB** | 3.06 GB |

The indexer's pooled keys are in that: one 128-wide vector per four tokens
per full-attention layer, which is a fourteenth of the latents beside them.
Nothing else in the engine grows with context. At 1M — the release's
figure — the state would be around 25 GB, which is a machine question and
not an engine one.


## 70. GLM's vision tower, and the resize that was not a rounding error (2026-08-27)

§65 listed the tower as not implemented and said why: GLM's is not K3's.
It is now, and the interesting parts are the two places where "close
enough" would have been wrong in a way nothing would have reported.

### What it is, against what K3's is

| | K3 | GLM-5.3-Flash |
|---|---|---|
| blocks | 27 x 1024, 12 heads | 24 x 1024, 16 heads |
| position | learned 64x64 grid, bilinearly resized | 2D rope only |
| q/k | as projected | RMSNormed per head, then rotated |
| biases | none | on every projection |
| MLP | GELU (tanh) | clamped SwiGLU |
| patch | Conv2d over 3x14x14 | Conv3d over 3x2x14x14, the temporal slot a copy |
| merge | 2x2 reshape, two-layer projector | Conv2d downsample, gated merger with a LayerNorm and an exact GELU |

Two towers in one file rather than a branch inside one, for the reason a
`if tower == ...` in every function is harder to check than two functions
that each say one thing.

**It matched the oracle on the first run: rel L2 3.3e-5** on the real 563.6 M
tower, 7.7e-7 on the synthetic one CI builds. Which is less a statement
about the code than about writing `tools/glm_vision_ref.py` first and the C
against it — the four details that are easy to get wrong were wrong in the
reference too, once each, and were found there.

### The patch order is not a detail

The rows arrive **block-major over merge blocks** — block row, block
column, then the 2x2 inside it — which is what lets the tower's downsample
be a reshape, and what `get_vision_position_ids` assumes when it builds the
rotary indices. Raster order, the obvious reading and what K3 uses, rotates
every patch by someone else's position. The output stays finite, the norms
stay plausible, and the model describes a different picture.

It is checked against the reference's own reshape/permute/expand, on an
image whose dimensions are already a multiple of `patch * merge` so that
the resize is the identity and only the ordering is under test: **max abs
difference exactly 0**.

### The resize was 7.7%, and that is not a rounding error

K3's loader samples bilinearly, and the first version of GLM's did too —
it was the same code and the release's config does not shout about
resampling. Measured against the processor's own choice:

| the engine's pixels vs | max abs | rel |
|---|---:|---:|
| torch bilinear | 0.0000 | **0.0000** |
| torch bicubic | 0.4172 | 0.0986 |
| torch bicubic + antialias (what the release does) | 0.2879 | **0.0773** |

The first row is the useful one: the sampling *convention* was already
exact, half-pixel centres and all. What differed was the kernel, and 7.7%
relative on the tower's input is a different image, not a noisier one.

So the kernel is implemented — separable, cubic with a = -0.5, support
widened by the scale factor on a downsample, which is what PIL does and
what torch ported. **Engine against torch's antialiased bicubic: max abs
4e-5, rel 0.00000.**

The temptation was to write "the resampling is bilinear where the release
is bicubic; see the docs" and move on. The number is what made that
untenable: a documented 7.7% is still a tower being shown something the
model was not trained on.

### What it costs, and what it does

The tower is 563.6 M parameters, 282 MB at 4 bits, and it is **loaded only
when a caller asks for images**. That needed its own fix: the skip was
written as "outside `tensor_prefix`", which on K3 is the same set as "the
tower" and on GLM — whose container has no prefix at all — is the empty
set. Its 282 MB were resident on every text-only open, and the memory plan
counted them in the floor. Both now say `vision_tower.` in as many words.

    $ waste run glm53.waste "What does this image look like?" --image x.png
    [x.png: 40 image tokens]
    This image displays a vibrant, abstract pattern of diagonal stripes in
    various colors like green, blue, purple, and pink, overlaid with fine
    vertical lines.

which is what the file actually contains. `waste plan` prices it at 805 MB
more with `--image`, against K3's 1.12 GB.

### The grid rule, transcribed rather than invented

GLM aligns to `patch * merge` = 28 pixels and then fits the result into a
budget expressed in *merged tokens*, with a binary search over the content
height when it does not fit. Two things in `smart_resize` look like
accidents and are not: the frame count rounds to a multiple of the temporal
factor before it multiplies the budget, and the search advances `low` to
`content_height + 1` rather than to the aligned height. Both are
transcribed.

## 71. The two K3 checks were measuring the router, not the arithmetic (2026-08-27)

`tests/run.sh` had two checks failing on K3 and only on K3: chunked prefill
and the CPU backend each differed from the default path by max-abs 0.2858
against a 1e-3 threshold, argmax unchanged. 0.7.0 shipped with them red, and
both §68 and the changelog blamed the i8mm/SMLAL trunk kernels. **That was
wrong, and the first measurement killed it**: `trunk_kern` defaults to
`TK_F32`, so neither kernel was in the run at all. §68's other observation —
that the two alternatives agree with *each other* exactly, so what differs
is the default — was right, and is half of what follows.

What it actually is, in the order the evidence arrived:

- A length sweep of the default path against the CPU baseline: 1 token
  3.9e-07, 4 tokens 5.0e-07, 12 tokens 1.1e-06, **13 tokens 1.4e-02**. Not a
  drift that grows with length. A step.
- The route traces at 13 tokens: 6 of 1196 routing decisions differ, **all
  at token 12**, the one the step appeared on.
- The scores behind the earliest of them, layer 56, experts 889 and 712.
  Printed at the dump's `%.6g` they are the same number. At `%.9g` they are
  0.112161167 and 0.112161085 — **a relative margin of 7.311e-07**, while
  the same expert's score moves 1.5e-08 between the two engines.

The last number is the whole finding, and it needs the distribution to mean
anything. Over the full 16-token prefill — 1472 decisions, 92 layers, 384
experts, top-16 — the gap between rank 16 and rank 17 has a **minimum of
7.311e-07**. The flip is not *a* tie, it is *the* tie: the closest call the
model made anywhere in the forward pass. The 1st percentile is 4.4e-05, sixty
times wider, and the median 7.4e-03. The 47 differences that follow the first
one average 5e-03, squarely typical — because they are computed on a hidden
state that has already moved, and are consequences rather than coin flips.

Two independent paths, chunked prefill and the CPU backend, produce
**byte-identical route traces** and differ from the default in exactly the
same 48 places. That is not each path having its own noise; it is NEON
summation order on one side and scalar on the other, disagreeing by 1e-08 on
a decision that needed 1e-07 to make.

**So the checks were wrong, not the engine.** A top-K router converts an
arbitrarily small arithmetic difference into a discrete one. Past the first
flipped expert the two paths are running different weights, and the distance
between their logits stops being a measurement of arithmetic — it measures
how much the model cares which of two indistinguishable experts it used. A
threshold on that quantity has no setting that works: 1e-3 fails on a tie
forever, and the 0.3 that would pass could not catch a kernel that was
actually broken. §44 and the `exp1` work already said this about kernel
comparison ("a logit norm cannot separate *the arithmetic moved* from *the
selection moved*"); the suite had not been told.

`tests/route_diff.py` asks the question the threshold was standing in for.
Given the reference's route trace, the other path's, and the reference's own
scores, it answers **identical**, **tie** (the first disagreement is between
two experts the reference could not separate, at or under a relative margin
of 1e-5) or **diverged**. Only the first disagreement is judged, for the
reason above. `--eps` defaults to 1e-5 because that is the empty decade
between the tie that flipped, 7.3e-07, and the tightest call that held,
4.4e-05 — 14x above one and 4x below the other, rather than a round number.

The check that replaces the threshold is **stronger**, in three ways. The
argmax is now a hard failure on every path rather than a clause on one of
them. Routing that differs on a margin the reference could resolve is a
failure that names the token and the layer, where before it was a distance.
And a difference past the threshold with the routing *unchanged* — the case
that really is a defect in the arithmetic — is now its own verdict instead
of being pooled with the tie into a single red line.

The cost is one line per (token, layer) in two traces, about 400 KB on this
prompt, written by dumps that already existed. The engine is unchanged
except for the dump's precision: `%.6g` cannot resolve a tie, and a dump
whose only job is to say how close a decision was must round-trip a float.
Six digits printed the two scores as equal and made a near-tie look like a
selection bug.

Kimi-Linear never tripped any of this. Its router does not tie at 16 tokens,
which is why the checks looked correct for as long as they did.

## 72. A third tie, and the two bugs it was hiding behind (2026-08-27)

§71 fixed two K3 checks that were measuring the router. Fixing them turned
CI green enough to reveal that `main` had two further problems, and the
second turned out to be the same finding a third time.

**0.7.0 did not build on x86_64 Linux, on Windows, or under ASan.**
`model.c` declared and called `waste_mvq4_rows_i8mm` unguarded, but the
Makefile adds `src/simd_i8mm.c` to `SRC` only for `arm%|aarch64%`. Three of
five CI jobs failed at the link, one of them `asan + fuzz` — so the
sanitizers had never run against anything 0.7.0 added. `simd_i8mm.c` was
already careful, defining the symbol twice so it exists in every ARM build
even where `-march` did not take; what was missing is that **a dispatcher
and the list of files that satisfies it have to be guarded on the same
predicate**. The Makefile's own comment records the identical failure once
before from the other direction, a `findstring` that left `kda_neon.c` out
while `backend.c` still called it. Compiling `model.c` for x86_64 leaves the
symbol undefined at `c9cd7bb` and unreferenced now; a full x86_64 build
links.

**The GLM oracle check was failing on both Linux jobs and nowhere else**,
and had been through 0.7.0, saying only "the GLM path diverges from the
oracle". Adding four numbers to that verdict answered it in one CI run and
overturned every guess that preceded it, including two of mine.

linux-x86_64 and linux-arm64 reported max abs 0.105701 and 0.105702, rel L2
0.00783997 and 0.00783999. **Two different ISAs agreeing to five
significant figures is not a platform difference**, and the engine was not
what moved: at the worst logit macOS, both Linux runs and the shipped
fixture all print -7.3257, and only the freshly generated Linux oracle
prints -7.43141.

The ruled-out list is worth as much as the finding, because each entry cost
a measurement: the fixture is not stale (the engine matches it to 5.72e-06);
macOS is not skipping the real comparison (its line says "a PyTorch oracle
built from this container"); `uv` resolves the same torch 2.13.0, fla-core
0.5.2 and einops 0.8.2 on both platforms, differing only in triton, which
cannot matter because `load_naive_kda()` is never called; `-ffp-contract`
on/fast/off all agree within 6.7e-06; `WASTE_THREADS` 1 through 18 are
bit-identical; ASan and UBSan are clean and change no digit. **And it is not
a router tie**: the tightest top-2-of-8 boundary gap in that prefill is
9.4e-03, with nothing under 1e-3, so the router cannot flip on noise. §71's
answer was the wrong answer here, and checking cost one dump.

It is a **DSA** tie. At layer 2, token 15, the four visible pools score 0,
0, 0 and 0.00164 with `keep=2`: pool 3 wins outright and the second slot is
an **exact three-way tie**, margin 0.000e+00. The engine takes pool 1;
`torch.topk`, called in `kimi_ref.py` with `sorted=False` and so free to
answer in any order, takes pool 2 on Linux and pool 1 on macOS. One pool of
four tokens attended differently is worth rel L2 0.0078 in the logits, and
it is a defect in neither implementation.

`kimi_ref.py` already wrote the trace that says this, under the same
`WASTE_DUMP_DSA` the engine uses, and its own comment says why: so the two
selections "can be diffed directly rather than inferred from a logit
difference". **Nothing was diffing them.** The instrument existed, was
documented, and was never wired to a check — which is its own lesson, and
cheaper to learn here than at 982 GB.

`tests/dsa_diff.py` reports the same three answers `route_diff.py` does —
identical, tie, diverged — over the pool ranking instead of the expert
ranking, and the GLM check now consults it. Three checks on three models now
share one shape: compare the logits, and where they part, ask which discrete
decision moved and whether anything could have resolved it.

The general form, and it is not about ties: **a check that reports only a
verdict makes the next person re-derive the evidence.** Two of these three
were investigated for hours against a bare FAIL. The fix each time was four
numbers the check already had in hand.

## 73. A protocol checked only against itself (2026-08-28)

PR #50 added Kimi K2's native tool-call rendering to `serve/chatfmt.py`,
with 438 lines of tests. Every one of them passed. The rendering was wrong.

The tests were self-consistent: they fed the renderer's output to the
parser and asserted the parser read back what the renderer wrote. That
agrees with itself whatever the format is, which is the whole problem —
`test_chatfmt.py` even *pinned* the defect, asserting
`<|im_system|>system<|im_middle|>` for a tool result turn.

K2's published `chat_template.jinja` says
`<|im_system|>{{ message.get('name') or message['role'] }}<|im_middle|>`.
So a result carrying `name` opens a turn named for the **function**, one
without it opens `tool`, and never `system` — which is what `chat.json`'s
system prefix carries and what the PR substituted for it. Rendering the
same conversation both ways, everything else was **byte-identical**: the
`tool_declare` turn with compact separators, the id alone after
`<|tool_call_begin|>`, the arguments, the section close, the generation
prompt. One line out of the protocol, and no self-consistent test could
ever see it.

The reason this needed an oracle at all is worth stating, because it is
backwards from the usual case. **Kimi-Linear's release ships the five
tool-call tokens and no `chat_template` whatsoever.** The vocabulary is
declared and the grammar is not. `chatfmt.py`'s own docstring had said so
and refused to guess ("the markup Kimi-Linear's tokenizer carries for it is
not transcribed anywhere in this repo"). The grammar is K2's, K2 publishes
it, and `tests/serve/test_chatfmt_upstream.py` now diffs against it — the
same thing `test_xtml.TestAgainstUpstream` does for K3, and the cheapest
oracle in the repo: a 2 KB template plus `examples/chat-kimi-linear.json`,
no weights and no container.

Evidence that the two formats are the same family, which is the part
nothing can prove outright: `chat-kimi-linear.json` was written
independently from Kimi-Linear's own specials, and renders plain
conversation **byte-identically** to K2's template. Two formats that agree
exactly on the part that is checkable are good reason to believe they agree
on the part that is not.

One divergence is deliberate and asserted rather than fixed: with no system
turn first, K2's template inserts Moonshot's own system prompt. A server
rendering an arbitrary container's `chat.json` has no business inventing
that, so the test states the difference instead of forbidding it.

The differential also caught a defect of *mine* within minutes of existing.
Resolving the merge left `segments: list[Segment] = []` running after the
tool declaration had been appended to it, so a declared tool rendered as
nothing at all — silently, since the conversation still came out valid.
§71 and §72 were about checks that report a verdict without evidence; this
is the other half. **A test that only compares a thing to itself is not a
weak oracle, it is not an oracle**, and the three protocols this repo
renders now each have one.

## 74. Qwen3.8-Flash-Next: what was new, and what only looked new (2026-09-04)

Four architectural pieces this engine had never run — Gated DeltaNet,
Qwen Sparse Attention, HyperConnection, and a per-layer n-gram embedding —
and the useful finding is how little of the *engine* they touched. The
container format did not change: a packed `[E, 2I, H]` gate_up beside an
`[E, H, I]` down splits into exactly the WEXP records everything else
writes, one expert per 4 KiB-aligned record, so routing still costs one
`pread` and the expert cache, the read-ahead and the expert-parallel path
were reused unmodified. What is genuinely Qwen's is five self-contained
kernel files and a loader branch.

**The 80 GiB trunk that is 2.6 GB resident.** The n-gram tables are 16
heads of ~20 M rows, 78 of the trunk file's 80.51 GiB. Held resident they
would exceed the whole RAM budget on any machine this targets; read a row
per head per token they cost one Q8G row each. The same exclusion the
embedding table has always had, for the same reason, and it is what makes
a 176.94 B model open with a 3.11 GB floor. The converter has the mirror
problem: a head is ~12 GiB as f32, so it is written 64 Ki rows at a time,
relying on Q8G grouping along the last dimension to make the batches
reconstruct what quantizing the head whole would have produced.

**8 GiB of expert cache, and nothing above it.** Measured over 48 greedy
tokens: 4 GiB gives 3.20 tok/s at an 8% hit rate, 8 GiB gives 4.97 at 64%,
16 GiB gives 4.92 at 88%. The collapse below one working set is §3's rule
again — below a multiple the hit rate is zero, not low. The flat top is
the more useful half: **24 points of hit rate bought nothing**, because at
64% the remaining reads already overlap the arithmetic. A cache sized to
the machine rather than to the knee spends RAM for no tokens.

**A tokenizer difference that no vocabulary test could see.** Qwen's
pre-tokenization pattern is `\p{N}` where Kimi's and GLM's are
`\p{N}{1,3}`: every digit is its own piece. `tools/hf_tokenizer.py` was
right to refuse the pattern rather than approximate it, and the engine now
carries `tokenizer_digit_run` the way it already carried
`tokenizer_han_split`. The trap is in the checking, not the fixing —
**Qwen's vocabulary contains no multi-digit token at all**, so on this
checkpoint the two settings produce identical ids and every parity test
passes either way. "202" has no merge to reach. The flag is therefore
tested on a synthetic vocabulary that does hold `20`, where the pre-token
boundary is directly visible. A parity test against the release would have
green-lit the wrong pattern for the next member of the family.

**An unexplained 4e-3.** The container-native oracle and the engine read
the same trunk dequantized to the same f32, so their difference should be
summation order — around 1e-6. It is 4e-3 relative per layer. Routed
expert ids and weights match exactly at every layer and the argmax matches,
so nothing observable is wrong, and end-to-end generation is coherent on
the real checkpoint. It is recorded here undiagnosed rather than absorbed
into a tolerance: 4e-3 is close to bf16's epsilon and nothing in that path
should be rounding to bf16. The suite gates at the measured value so the
number cannot grow while nobody is looking, which is the least a check can
do about a thing it does not understand.
