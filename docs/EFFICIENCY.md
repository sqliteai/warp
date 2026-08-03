# Where the remaining speed is (2026-07-31)

K3 decoded at 0.33 tok/s when this was written and does **0.49–0.53** now;
§4A is the one change that did it, and §4B–D and §5 are what was measured
and refused. *(A second change has landed since: the router lookahead of
§4F, which takes the default budget to 0.56–0.63 — see the dated correction
there, because this document refused it first.)* This document is the
account of "what is left", started after
an outside article
([AirLLM](https://github.com/lyogavin/airllm) running K3 at ~5 minutes per
token) prompted a review of the streaming path. The article itself has
nothing this engine wants — WASTE is ~100x faster and never expands an
expert at all — but one line of it does: **AirLLM overlaps loading with
compute and WASTE does not.**

Everything below was measured on the 64 GB M5 Pro with the container on the
internal SSD, unless marked as an estimate. Estimates are labelled and their
inputs named, because the point of this document is to decide what to build
next and a projection dressed as a measurement is how that decision goes
wrong.

## 1. Grouping tokens removes the I/O and none of the compute

This is the finding §4D follows from, and it is the opposite of what the
offloading literature assumes: there, compute is free and the transfer is
the wall, so putting more tokens in flight is the answer. Here the two are
the same order of magnitude and it is not.

Measured with `waste chat … /stats` (cumulative counters, so they include
prefill) at a budget on the floor, so the cache cannot flatter the count.
Chunk dedup is structural — `moe_chunk` iterates over the *distinct* experts
of the chunk — so a miss is a record, and the miss count is the number of
distinct records the chunk needed.

| prompt | cumulative reads | wall |
|---|---|---|
| N=1  | 14 794 | 57 s |
| N=8  | 18 836 | 73 s |
| N=16 | 22 289 | 88 s |
| N=32 | 27 881 | 120 s |

A standalone token costs 92 x 16 = **1472 reads**. The marginal token
*inside a chunk* costs far less — but only in I/O:

| interval | reads/token | vs standalone | I/O | compute | total |
|---|---|---|---|---|---|
| N=1→8   | 577 | **0.39x** | 0.64 s | 1.65 s | 2.29 s |
| N=8→16  | 432 | **0.29x** | 0.48 s | 1.40 s | 1.88 s |
| N=16→32 | 350 | **0.24x** | 0.39 s | 1.62 s | 2.00 s |
| *standalone* | 1472 | 1.00x | **1.62 s** | ~1.5 s | ~3.1 s |

(I/O derived from the read count at the measured 10.73 GB/s; compute is the
remainder of the wall clock, which is why it carries the noise.)

**Grouping tokens removes 70–76% of the I/O and 0% of the compute.** The
compute column is flat at ~1.5 s/token in every row, because `vq_apply`
costs one pass per (token, expert) pair — `vq_rows` does exactly `stages`
gathers per row per vector position — and the number of pairs is T x K x 92
however the tokens are grouped.

Independent check: this puts the ceiling of any batching scheme at
3.1 / 1.9 = **1.63x**, and chunked prefill measures 0.47 tok/s against 0.29
sequential = **1.62x**. The model reproduces a number nobody fitted it to.

## 2. The disk is nearly saturated at queue depth 1

`tools/diskbench.c`, 11.83 MB records (K3's real expert size), cache
bypassed, internal SSD:

```
seq write   :  8.30 GB/s
seq read    :  9.06 GB/s
rand  1 thr : 10.73 GB/s   <- what the engine gets today
rand  2 thr : 12.79 GB/s
rand  4 thr : 12.89 GB/s
rand 16 thr : 12.89 GB/s
```

One read in flight already gets 83% of the maximum. **A second read buys
20% of bandwidth and nothing more.** So the value of prefetching is not
bandwidth — it is that the read stops blocking the arithmetic.

## 3. The MoE loop serializes reads it already knows about

`moe_layer` picks its top-K, then:

```c
for (j = 0; j < K; j++) {
    rec = read_expert(m, L, idx[j]);   /* blocking 11.83 MB pread */
    ...                                 /* only then does it compute */
}
```

`waste_ecache_get` calls `fetch` inline on a miss. Queue depth 1, no read in
flight, no overlap.

But **all 16 expert ids are known before the first read**, from the routing
loop directly above. These are sixteen independent reads the engine has in
hand and issues one at a time.

Implementation constraint worth recording: the compute pool in
`src/threads.h` is fork-join with a single global job descriptor —
`g_pool_run_mu` serializes whole jobs — so the prefetcher cannot ride on it.
It needs its own thread(s) and its own queue.

## 4. What each lever is worth

Baseline **3.03 s/token (0.33 tok/s)** at the default budget (17.56 GB
cache, 13% hit). Decomposed at the margin: **I/O 1.41 s, expert matmul
1.03 s, everything else 0.50 s**.

### A. Pipeline I/O against compute — **built, ~1.6x measured**

Turns a sum into a maximum. With two reads in flight the I/O also runs at
12.89 GB/s rather than 10.73, so 1.41 → 1.17 s.

    before:     1.41 + 1.03 + 0.50 = 2.94 s
    pipelined:  max(1.17, 1.03) + 0.50 = 1.67 s   ->  1.81x projected

**Shipped, and measured at ~1.6x.** Two binaries built separately from
`main` and this branch, alternated on one machine: K3 goes 48.41 → 30.03 s
and 58.61 → 32.85 s for 16 tokens, 0.27–0.33 → 0.49–0.53 tok/s, with
`3357 hit / 20195 miss` identical in every run of both. Kimi-Linear gains
too, since 22% of its accesses still miss: `run` 8.96 → 10.81 tok/s and
`bench` 7.24 → 8.77, both about **1.21x**. Chunked prefill gains ~1.35x, because a chunk
already spreads each expert over several tokens and has less I/O to hide.
The projection was 1.81x; the gap is the per-layer LUT build, which is
serialized ahead of the applies and cannot hide the layer's first read.
[LEARNED.md](LEARNED.md) §22 has the defects it took to get there.

Two reader threads with their own queue (`WASTE_IO_THREADS`, `WASTE_IO_DEPTH`;
0 restores the synchronous path exactly). No format change, no reconversion,
output bit-identical — `tests/run.sh` checks read-ahead against synchronous
reads byte for byte.

Shape confirmed in the literature: SP-MoE ([arXiv:2510.10302](https://arxiv.org/abs/2510.10302))
reports 1.07–3.5x from asynchronous prefetch with a cutoff-layer policy;
MoE-SpeQ ([arXiv:2511.14102](https://arxiv.org/abs/2511.14102)) similar.

### B. The paging cliff — built, measured, and it is not a speedup

[LEARNED.md](LEARNED.md) §16: 17.32 GB of cache gives 13% hit at 0.32 tok/s;
29.32 GB gives 37% hit at **0.04 tok/s**. The engine caps itself at
`floor + 1x working set` to stay away from the cliff, and pays for it by
running at 13% hit when 37% is technically reachable.

If 37% were safe, I/O drops 1.41 → 0.85 s, and with (A) the step reaches
**1.53 s → 1.98x → ~0.65 tok/s**.

Idea to try: allocate cache slots as **purgeable** memory — on macOS
`vm_allocate` with `VM_FLAGS_PURGABLE` plus
`vm_purgable_control(VM_PURGABLE_SET_STATE, VM_PURGABLE_VOLATILE)` while a
slot is idle. Under pressure the kernel *discards* those pages instead of
compressing and swapping them; the engine finds the slot empty and treats it
as a miss, which is exactly the `pread` it already knows how to do. The 8x
cliff becomes graceful degradation and the default budget can go back to
being aggressive. Linux equivalent is `MADV_FREE` plus a sentinel.

**Built and measured (2026-07-31), and the hypothesis was wrong in an
instructive way.** `WASTE_PURGEABLE=1`, 8 tokens, read-ahead on:

| budget | cache | purgeable | hit | decode |
|---|---|---|---|---|
| 46.25 GB (default) | 17.56 GB | off | 19% | **0.49–0.52 tok/s** |
| 46.25 GB (default) | 17.56 GB | on | **0–1%** | 0.29–0.33 tok/s |
| 58 GB | 29.32 GB | off | 39% | **0.04 tok/s** |
| 58 GB | 29.32 GB | on | 0–21% | **0.22–0.25 tok/s** |

Read the last two rows first: at an over-large budget purgeable is **6x
faster**, exactly as designed — the kernel discards a slot instead of
swapping it, the engine re-reads, and the cliff becomes a slope.

Now read the first two. **At the budget that actually works, purgeable
costs 1.6x**, because the hit rate goes to nothing. macOS reclaims volatile
objects eagerly, not only under pressure, so a cache that would have stayed
resident is taken anyway.

Both rows have one cause, and it is the finding: **volatile memory is
memory you have given away.** The choice purgeable offers is not "keep more
cache", it is "lose it cheaply or lose it expensively". §16's cliff was
never the OS mismanaging the engine's memory — the memory was never the
engine's, and the 39% hit rate in the third row is real while every hit in
it is a page fault.

So the projected 1.98x does not exist. A cache above what the machine will
leave resident cannot be had at any price, and the default budget resolver
— which steps down a whole working set at a time — was already choosing the
only size that works.

**Kept, off by default, as an escape hatch**: it turns a 6x catastrophe from
a badly-chosen `--budget` into a 2x slowdown. Turning it on at a sane budget
is a mistake, which is why it says so here and in `waste.h`. Bit-identical
either way; `tests/run.sh` checks that.

Linux's `MADV_FREE` equivalent is not written, and this result is the reason
to expect little from it.

**The opposite bargain was tried too (`WASTE_MLOCK=1`, also off by
default).** Wiring the slots halves the cliff — 52 GiB goes 0.06 → 0.15
tok/s and 58 GiB 0.03 → 0.06 — and still lands three times under the default
budget, because it pins the cold part: the cache hits 19–30% while the
27.5 GiB trunk is read in full every token. At 58 GiB it is impossible
anyway, 56.82 GiB of trunk plus cache against a 52.48 GiB
`vm.user_wire_limit`. What it *does* buy is at the budget that already
works: five alternated runs give a median 0.51 against 0.42 and a spread of
12% against 38%, because the pageable arm degrades on a machine that has
been worked and the wired one does not. That is §16's "sweep upward, never
downward" bought back. [LEARNED.md](LEARNED.md) §30.

**And §31 corrects which buffer to wire.** The cache was the one with a
switch, not the one that mattered: wiring it alone is *worse than nothing*
on a quiet machine (0.50 against 0.55), because it leaves the 27.5 GB trunk
— read in full every token — as the pageable part. Wiring the trunk instead
gives **0.56–0.58 tok/s in every run, quiet machine or worked**, which is
the flattest number in this repository. `WASTE_MLOCK=1` wires both; still
off by default, because Linux's 8 MB `RLIMIT_MEMLOCK` would refuse it.

**It does not move the knee** (§32). Re-sweeping the budgets wired: 3x at
52 GiB, where the OS was choosing wrongly and pinning the hot part fixes it;
nothing below the knee, where nothing was being paged; and nothing above it,
where 58 GiB puts 35.6 GB into swap and 8 tokens do not finish in eleven
minutes. The knee is set by how much memory exists. Policy only decides
which part of the engine is destroyed when there is not enough.

### C. Stage-major records — measured and dropped

Today a record is `[hdr][gate][up][down][scales]`, and within a matrix
`[row_block][vec_pos][row_in_block][stage]`: **stages are interleaved at the
innermost level**, so reading 2 of 3 stages would mean reading two bytes in
every three and saving nothing.

Proposed for format v1:

    [hdr][scales][stage0: gate|up|down][stage1: ...][stage2: ...]

each plane padded to 4 KiB. Then reading `s` stages is **one contiguous
pread of a prefix of the record** — design goal 1 ("one coalesced read per
expert") survives intact, and reading all three costs ~0.1% of padding.

What it enables:

- **per-activation precision**: read two stages for the tail of the top-16,
  the ones whose renormalized routing weight is small. Error by stage count
  is 57.5% / 33.2% / 19.5% ([LEARNED.md](LEARNED.md) §20).
- **a two-stage cache**: 1.5x more experts in the same RAM, and a hit that
  needs full precision reads only the missing stage — a third of a record.
- graceful degradation under I/O pressure instead of a stall.

It cuts compute in the same proportion, because `vq_rows` does `stages`
gathers per row. If one stage in three is dropped for half the experts, both
buckets go x0.833 and the step reaches **1.36 s → 2.23x → ~0.74 tok/s**.

**This is not what §20 refuted.** §20 measured *static per-expert*
allocation and found the delta flat to 1.01–1.15x, which is correct and
settled. This is *per-activation* allocation keyed on the current token's
routing weight — the one signal §20 identifies as not flat. And it differs
from §20's closing note (demoting the cold tail saves disk and ~0% of the
reads, because cold experts are not read) in that it demotes what is being
read *now*.

**The measurement that gates it — run, and it says no (2026-07-31).**

The whole lever rests on the top-16 being top-heavy: demote the tail only if
the tail is light. `WASTE_DUMP_ROUTE=path` writes one line per (token, layer)
with the renormalized weights; 1104 rows from 12 K3 decode tokens over 92 MoE
layers:

| rank | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| mean weight | 0.149 | 0.108 | 0.077 | 0.055 | 0.043 | 0.032 |

**Rank 1 to rank 16 is a factor of 4.6, and ranks 9–16 carry 33.3% of the
mass** — a third, where the lever needed under a tenth. It is not a
per-layer effect either: the tail mass runs 21.6% to 48.4% across the 92
layers, no layer peaked enough to demote selectively.

Cross-checked on Kimi-Linear, the same way §20 did when it wanted to know
whether K3's QAT was what flattened its experts: top-8 with
`routed_scaling_factor` 2.446, ranks 1 to 8 run 0.243 to 0.064 once
normalized — a factor of 3.8 — and the bottom half carries **32.0%**. The
same number on a different model, a different expert count and a different
top-k. **The routers in this family do not concentrate**, and that is a
property of the family rather than of K3.

Priced with §20's own formula, `err² = err3² + mass(S)·delta`:

| demoted | mass | expert error | I/O and compute saved |
|---|---|---|---|
| ranks 9–16 | 33.3% | 19.5% → **24.9%** | 16.7% |
| ranks 13–16 | 14.5% | 19.5% → **22.0%** | 8.3% |

19.5% is the measured operating point and 20.3% is what a K3 expert costs at
3 bits from MXFP4; 24.9% is well past both, for a sixth of the reads. That is
the same straight line §20 found for static allocation — "the exchange rate is
fixed and both ends of it are bad" — and per-activation allocation lands on it
rather than escaping it.

**So the layout change is not worth making**, and the reason is one level
deeper than §20's: K3 is homogeneous in how hard its experts are to quantize
*and* in how much weight its router gives them. There is no tail to demote
because the router does not make one.

What would revive it: a router whose weights are actually peaked (a different
model), or a demotion whose error cost is far below the residual-stage
ladder's 19.5 → 33.2. Both are somebody else's model, not a tuning change
here. The instrument stays in the tree — the dump is four lines behind an
env var — because the question recurs for every new container.

### D. Batching and speculative decoding — refuted for this machine

The offloading literature — SpecMoEOff (2.5x), SP-MoE, MoE-SpeQ — all
assume **compute is free and the transfer is the wall**. On a GPU behind
PCIe that holds. Here it does not: at the margin compute is 1.5 s/token
against 1.62 s of I/O. They are nearly equal.

From §1, grouping tokens removes 76% of the I/O and none of the compute.
So:

- **Batching** tops out at 1.63x on its own, and it does *not* compose with
  (A), because it removes I/O that (A) has already hidden underneath the
  compute. After (A), B=8 lands on the same ~1.98x that (B) reaches alone.
- **Speculative decoding** at d=4 costs 1 + 3(0.39) = 2.17 token-equivalents
  of I/O but **four tokens of compute**. Accept two of four and it has spent
  more time than generating them in sequence. K3 also ships no MTP head
  ([K3.md](K3.md)), so it would need an external draft on top of that.

Recorded here because it is the first thing anyone who reads that literature
will try to build.

### F. Cross-layer predictive prefetch — measured and refused

Within a layer the prefetch is exact. Across a boundary it must guess, and
the guess is not there: co-occurrence from layer L predicts layer L+1's
top-16 at **29.0%** recall, against **29.5%** for simply reusing the
previous token's set — which the cache already does for free, and 20.5% for
the layer's static hot 16. Break-even needs ~60%, and a predictor fitted on
the data it is scored against reaches only 49.7%, so even the unachievable
ceiling loses. [LEARNED.md](LEARNED.md) §29 has the arithmetic and the
`WASTE_IO_DEPTH` sweep that shows there is nothing left to overlap within a
layer either.

**Refuted as written, and the lever is built (2026-08-01).** The heading is
wrong and is kept because the mistake in it is the useful part: what was
measured above is *one* cross-layer predictor, a co-occurrence table over
the router's past answers, and its failure was read as the question's
answer. Asking the router instead — running layer L+1's router weights on
layer L's hidden state — recalls **59.0%** at 16, twice the number in the
paragraph above, and the prediction is steeply ranked (92.2% at rank 1,
81.4% cumulative at 6). Prefetching the top 6 into the layer boundary is
right about four times in five.

Built and on by default (`WASTE_LOOKAHEAD=0` disables it). The real router
still decides, so logits are bit-identical and `tests/run.sh` checks that.
Measured in one process with the cache cleared identically per arm:

| lookahead | median decode | demand hit | data read |
|---:|---:|---:|---:|
| off | 0.506 tok/s | 7.2–7.7% | 204–205 GB |
| top 6 | 0.541 tok/s | 38.0–38.2% | **191 GB** |

Two consequences reach the rest of this document. It is **not** free the way
§4A was: at the default cache it reads 6.6% *fewer* bytes, and at a 3.32 GB
cache 8% *more*, because speculative records get evicted before use — it is
a prefetch at small caches and a scheduling change at large ones. And it
moves what §4B and the budget resolver rest on: a record now has to survive
one attention rather than one token, so a cache far below a token's working
set is no longer worth zero ([GATES.md](GATES.md) Gate 7).

The same hook in `moe_chunk` was built and removed: a chunk layer claims
~550 slots against a decode layer's 16, the speculative records are what
LFRU evicts first, and reads went up 6.9% for no wall-clock change. There is
no idle window in the chunk path to fill. [LEARNED.md](LEARNED.md) §34–36
and §39.

### E. Where the bottleneck actually is

This section used to argue from `max(0.85 I/O, 1.03 matmul)` that (A) and
(B) together would leave the engine compute-bound, and that a faster
`vq_rows` was therefore the thing left to build. **The profile says
otherwise, and §5 is what came of believing the arithmetic instead.**

Six decode steps of K3 with read-ahead on:

| stage | s | share |
|---|---|---|
| expert I/O | 9.95 | **54.8%** |
| expert matmul | 4.94 | 27.2% |
| — of which LUT apply | 4.34 | 23.9% |
| kda | 1.69 | 9.3% |
| LUT build | 0.48 | 2.7% |

The reads are still **twice** the arithmetic even after (A) hides them
behind it. The projection had overestimated the matmul and underestimated
the wait, and nothing checked it against a profile before it became a plan.

So the wall is not where this document first put it. Every I/O-side lever
here — bigger cache, batching, speculation, per-activation precision — has
been measured and refused for its own reason, and the arithmetic is not
what is holding the clock either. On this machine the remaining headroom is
hardware: a faster disk, or RAM that holds more than one token's working
set.

> **One I/O-side lever did survive, and it is not on this list (2026-08-01).**
> §4F's correction: the router lookahead. It does not read fewer bytes and it
> does not need a bigger cache — it stops the reads from being *waited on*,
> the same shape as §4A one layer further out. That is the category this
> paragraph missed, not a number it got wrong: every lever enumerated here
> is about the size of the I/O, and both of the two that have ever paid are
> about its timing.

## 5. `vq_rows`, optimized against the wrong premise

Acting on the corrected profile rather than the projection would have said
not to bother. It was done first and is kept, because it is bit-identical
and free, and because what it measured is worth having.

**What shipped.** The inner loop does three table lookups per row, which are
the algorithm, and read the three index bytes one at a time, which are not:
eight rows are 24 consecutive bytes, so six word loads and some shifting
replace 24 byte loads. Per eight rows, 64 memory operations become 46, with
eight independent gather chains instead of four.

Measured back to back against the previous commit's `model.c`, both
harnesses, on one sitting:

| | `waste bench` | `waste run`, 32 tokens | K3, LUT apply at 6 threads |
|---|---|---|---|
| before | 8.45 tok/s | 10.53 tok/s | 3.09 s |
| after | **8.73–8.78** (+3.6%) | **10.88–10.93** (+3.5%) | **3.00 s** (+3%) |

Three harnesses agreeing on 3–3.6% is the result. On K3 that is 3% of a
bucket worth 22% of a step — around 0.7% end to end, which does not clear
the noise. **The loop got faster and the model did not.** That is what
"expert I/O is twice the arithmetic" means in practice.

The first version of this table said +6.6%, from a baseline taken an hour
earlier in the same session rather than back to back. §16 says a row
measured after the machine has been worked is measured on a different
computer, and it is just as true when the drift flatters the change.

**Two things measured and refused on the way:**

- **Accumulators in registers.** Turning the loops inside out for an
  eight-row sub-tile keeps the running sums in registers and deletes the
  `acc` load/store traffic entirely — 30 memory operations per eight rows
  instead of 46, and bit-exact, since each row still sums over `v`
  ascending. It is **17% slower** (8.93 against 11.08 tok/s). Consecutive
  `v` are 192 bytes apart, so the sub-tile re-walks the block's whole index
  span once per sub-tile and touches five cache lines for every one it
  uses. Locality beat operation count, which is §7's lesson arriving from
  the other side.
- **`VQ_SUPER`**, which controls how often the LUT is re-streamed. Swept 1,
  2, 4, 8 on both models: 11.12 → 11.20 tok/s on Kimi-Linear and 4.33–4.35 s
  on K3. Flat. §7 refuted the table-bandwidth theory once already; it is
  still refuted with a third of the index loads gone.

**And one thing found.** The apply saturates at **six threads** — exactly
this machine's performance-core count. 2 → 6 threads is 7.31 → 2.99 s, and
6 → 18 is 2.99 → 2.89. The twelve efficiency cores are worth 3%.

**What is actually left.** The engine is I/O-bound again, 2:1, and every I/O
lever in this document has been measured and refused. The remaining headroom
is not in this file: it is a faster disk, or a machine whose RAM holds more
than a token's working set. On this one, 0.49–0.54 tok/s is close to what
the hardware gives.

> **0.56–0.63 as of 2026-08-01**, from §4F's router lookahead — which this
> paragraph had already refused under the wrong predictor. "Close to what
> the hardware gives" was measured against a schedule, not against the
> hardware.

## 6. Do not rebuild these

Refuted with measurements, in this repo:

- 2-bit experts — 34% error ([LEARNED.md](LEARNED.md) §3, [K3.md](K3.md))
- static per-expert bit allocation — Gate 6 / §20, delta flat 1.01–1.15x
- KBVQ shared low-rank — §3, pending only the activation-weighted rerun
- a 3-bit trunk — §13, the quality wall sits in front of the speed wall
- streaming `lm_head` — §13, a net loss of ~0.8 GB/token

## 7. Order

1. ~~**Asynchronous expert prefetch** (A)~~ — **done**, ~1.6x. §22 of
   [LEARNED.md](LEARNED.md) has what it cost.
2. ~~**Measure the top-16 routing weight distribution**~~ — **done**, and it
   refused (C) for the price of an afternoon rather than a 4.7 h
   reconversion. The tail is a third of the mass, not a tenth.
3. ~~**Prototype the purgeable cache** (B)~~ — **done**, and it is not a
   speedup: volatile memory is memory the kernel takes. Kept off by default
   as an escape hatch for an over-large `--budget`.
4. ~~**Stage-major layout** (C)~~ — cancelled by 2.
5. ~~**Re-measure the cross-layer predictor** (F)~~ — **done 2026-08-01**,
   and it reopened what this document had closed: the refusal in (F) was of
   a co-occurrence table, not of asking the next layer's router. Built,
   shipped, on by default.

**Where that leaves it.** One of the four levers survived contact. (A)
shipped and is worth ~1.6x on K3 and 1.21x on Kimi-Linear; (B), (C) and (D)
are all refuted, each by a measurement that cost hours rather than the days
building them would have. §5 is the one that was built before it was
measured, and §4E is the profile that would have said not to.

**And then a fifth.** (F) was written as a refusal and is now the second
thing that shipped, because the measurement behind it had answered a
narrower question than the heading claimed. The working rule that produced
this document — measure before building — does not protect against that;
only asking what *else* the measurement could have meant does.
