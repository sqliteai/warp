# Feasibility gates

Working rule (Marco, 2026-07-24): before every long/expensive operation,
run a cheap real test that could kill it. Each gate below names the
expensive step it protects, the test, and the recorded verdict.

**Read this as a log, not as a status page.** The gates are dated and kept
as they were written, including the projections that the finished engine
went on to beat or miss. Every gate below has run except Gate 7, which is
open; where one carries a forecast, a note says what actually happened. The measured end-to-end
numbers live in [LEARNED.md](LEARNED.md) §12 and §16 and in the README —
those are the ones to quote.

| gate | question | verdict |
|---|---|---|
| 0 | does the trace→simulate method work? | ✅ passed 2026-07-24 |
| H | is the storage fast enough? | ✅ run 2026-07-27 — internal only |
| 5 | does the real cache behave like the simulation? | ✅ run 2026-07-27 |
| 1 | real K3 dimensions vs our estimates | ✅ answered 2026-07-27, see [K3.md](K3.md) |
| 2 | real batch-1 routing on a Kimi MoE | ✅ run 2026-07-27 on Kimi-Linear |
| 3 | quantization quality at 2–3 bit | ✅ run 2026-07-27, repeated on real K3 experts |
| 4 | engine correctness | ✅ all three steps passed 2026-07-27 |
| 6 | is per-expert bit allocation a real lever? | ❌ run 2026-07-29 — refuted, nothing to allocate |
| 7 | is the budget resolver's quantum still a working set? | ⏳ **open** — opened 2026-08-01 |

## Gate 0 — does the trace→simulate methodology work, and what does real
## batch-1 routing look like? ✅ PASSED (with a sobering data point)

*Protects:* everything downstream — the whole premise that we can measure
routing risk cheaply before building.

*Test (run 2026-07-24):* real per-token routing trace from OLMoE-1B-7B
(64 experts, top-8, 16 layers — already in HF cache), 299 decode tokens at
temp 0.7 on a C-coding prompt, via `tools/trace_hf.py` hooks; then
`tools/routing_stats.py simulate`. Wall time: ~15 min including two script
fixes. Trace fixture: `tests/trace_olmoe_299.jsonl`.

*Results (real, not synthetic):*

- **Next-token expert reuse: 43.5%** — moderate, not the strong locality
  the (refuted) literature claim assumed, and well below the 60.7% our
  synthetic Zipf trace showed. The adversarial verification was right to
  be skeptical.
- Concentration is modest: top 21% of (layer,expert) slots cover 50% of
  activations; top 50% cover 80%; top 73% cover 95%.
- LFRU cache hit-rate at K3-analog cache fractions:
  ~6% of expert bytes cached → **~23%** hit; 12.5% → 35%; 25% → 53%;
  53% → 81%. LRU thrashes to 0% at small caches (each token touches 128
  slots); LFRU degrades gracefully — the frequency-first
  policy is the right one.

*Implication for K3 on 64 GB (40 GB cache ≈ 6% of a ~700 GB expert set):*
if K3 routes like OLMoE, expect ~20-25% hit → ~9.5 GB misses/token at
2.12 bit → **~0.8-1.3 tok/s on a 12 GB/s NVMe**, below the optimistic
2-3 tok/s. The pruning plan B gains weight: half the experts cover 80% of
activations even in this flat-ish model.

*Caveats (why this doesn't decide K3 yet):* different scale and expert
granularity (K3: 896 fine-grained experts + quantile-balanced training —
could route flatter or sharper); 299 tokens is short (LFRU barely warms
up; learned pin sets improve with hours of workload); single
prompt/domain. Gate 2 reruns this exact pipeline on K3's real trace.

## Gate H — is the storage fast enough to stream experts? ✅ RUN 2026-07-27
## VERDICT: external USB disk is 13.6x too slow for inference; internal is fine

*Protects:* 1.5 TB download + conversion onto the wrong device.

*Test:* `tools/diskbench.c` — the engine's real access pattern (12 MB
records, `F_NOCACHE`, `pread`, 1-8 threads), 8 GB working file.

| device | seq write | seq read | random 12 MB (8 thr) | tok/s @12.5 GB/token |
|---|---|---|---|---|
| `/Volumes/WasteDisk` (APFS, USB, ASM246X bridge) | 0.91 GB/s | 0.92 GB/s | **0.94 GB/s** | **0.075** |
| internal SSD (MacBook Pro M5 Pro, 64 GB) | 9.85 GB/s | 9.52 GB/s | **12.78 GB/s** | **1.02** |

The external enclosure saturates at ~0.94 GB/s and does not scale with
threads: an ASMedia ASM246X is a USB 10 Gbps bridge, so the bottleneck is
the *bridge*, not the NVMe inside it. At that rate one cold token takes
~13 s; with the Gate-0-measured 23% LFRU hit rate, ~10 s/token
(**0.1 tok/s** — a 2000-token answer would take 5.5 hours).

The internal SSD hits 12.8 GB/s on random expert-sized reads — exactly the
top of the range every earlier projection assumed, and it has 1.7 TB free.

*Resulting placement decision:*

- **`/Volumes/WasteDisk/k3/` = download + staging** for the ~1.5 TB of raw
  MXFP4 shards (sequential writes at 0.9 GB/s are perfectly adequate, and
  it keeps the raw download off the internal disk permanently).
- **internal SSD = the converted WASTE container** (~700-900 GB, fits in
  1.7 TB free). This is what the engine streams experts from at runtime.
- Conversion reads shards sequentially from external → writes container to
  internal. Neither step is bottlenecked by the USB bridge.
- *Optional upgrade:* the machine has three free Thunderbolt 5 buses
  (120 Gb/s). Moving the same NVMe into a TB5/USB4 enclosure would give
  ~5-6 GB/s and make external-disk inference viable (~0.4-0.5 tok/s at the
  measured hit rate) — worth it only if the internal disk must stay free.

## Gate 5 — does the real expert cache behave like the simulation?
## ✅ RUN 2026-07-27. It does, and better above 12%

*Protects:* the entire feasibility argument. Everything before this point
assumed a cache that did not exist yet — the engine read every expert on
every token, and on a 64 GB machine the kernel's page cache was quietly
holding all 17 GB of Kimi-Linear, so the measured I/O cost was fiction.
K3's ~816 GB gets no such help.

*Test:* [src/ecache.c](../src/ecache.c) — bounded expert cache, LFRU with
sampled eviction, `pread` with `F_NOCACHE` so the page cache is bypassed
and the engine's own cache is the only one. 300 batch-1 decode tokens,
budget swept.

| budget | slots | % of expert set | hit rate | GB read/token | s/token |
|---|---|---|---|---|---|
| 512 MB | 201 | 3.0% | 13.2% | 0.448 | 0.16 |
| 1 GB | 402 | 6.0% | 40.3% | 0.308 | 0.14 |
| 2 GB | 805 | 12.1% | 61.9% | 0.197 | 0.13 |
| 4 GB | 1610 | 24.2% | 84.8% | 0.078 | 0.12 |
| 8 GB | 3221 | 48.4% | 93.9% | 0.031 | 0.11 |
| 16 GB | 6442 | 96.8% | 94.2% | 0.030 | 0.11 |

**Against the Gate 2 simulation** (same model, same policy, simulated):
6% → 40.3% measured vs 40.6% simulated; 12% → 61.9% vs 54.9%; 24% → 84.8%
vs 71.9%. The real cache matches at the K3-relevant fraction and beats the
simulation above it. The 1.5 tok/s projection for K3 on 64 GB stands.

> **What happened instead:** ~0.3 tok/s. The hit-rate model held — the
> floor below is the single most predictive number in the project — but
> the projection assumed a cache several times larger than a 64 GB
> machine can actually give K3 once the 27.28 GB trunk is resident, and
> it cost only the I/O. Measured sweep in [LEARNED.md](LEARNED.md) §12.

**The one place it is worse is the most useful finding.** At 3% the
measured hit rate is 13.2% against a simulated 29.4%, and at 1.5% it is
*exactly zero* — 2604 evictions in 2704 accesses. The reason: one token
touches 208 experts, so a cache of 100 slots keeps nothing alive long
enough to be reused. **The cache floor is one token's working set**, and
useful hit rates start at 2-3x that. For K3 that floor is ~960 experts x
16.5 MB = **~16 GB**, which a 64 GB machine clears with room for 3x — but a
32 GB machine would not, and that is now a measured statement rather than a
guess.

> **Still exactly true, and no longer binding (2026-08-01).** Re-measured on
> K3 in one process at 287 slots — a fifth of a token's 1472 records —
> [LEARNED.md](LEARNED.md) §39 reproduces the zero: **0.0% hit** with the
> router lookahead off. With it on the same 287 slots measure **29.1%**,
> because a prefetched record has to survive from one layer to the next
> rather than from one token to the next. The floor is a property of a
> demand-only cache, which was the only kind that existed when this gate
> ran. Whether the budget resolver's quantum should follow is Gate 7.

Correctness: cache on vs cache off is **bit-identical**, and both match the
oracle at rel 1.50e-06. Placement decides speed, never precision.

*Consequence:* `memplan.py`'s hit curve now comes from this measurement
rather than from simulation.

## Gate 1 — real K3 dimensions vs our estimates. ✅ ANSWERED 2026-07-27,
## the day the weights dropped. The full read is in [K3.md](K3.md)

*Protects:* buying/dedicating a 2 TB NVMe; all format TBDs.
*Test:* `routing_stats.py fetch + math` on the released config/index.
*Kill criterion:* per-token I/O or disk footprint far above estimates
(>20 GB/token @2 bit, or >1 TB at 2.5 bit).

*Verdict:* passed, but only because the MoE turned out to be **latent**.
93 layers rather than the 60 assumed, 92 of them MoE, experts operating on
a 3584-wide projection rather than the full 7168 hidden. At the 3-bit
operating point that is 17.0 GB per cold token and 952 GB of experts on
disk — inside the kill criterion on both axes, and roughly half what the
naive non-latent reading of the config implied. Every format TBD is
settled by the containers built since; see [FORMAT.md](FORMAT.md).

## Gate 2 — real batch-1 routing on a Kimi MoE. ✅ RUN 2026-07-27 on
## Kimi-Linear-48B (no GPU rental needed). Hit rates are BETTER than the
## OLMoE stand-in; K3 itself still to confirm

*Protects:* the 1.5 TB download and months of engine work.

*Test:* `tools/kimi_ref.py --trace` — the pure-PyTorch oracle running off
the 3-bit container, hooking the router each decode step. 300 tokens,
coding prompt, batch 1. Fixture: `tests/trace_kimi_300.jsonl`. This is the
same router family as K3 (sigmoid + grouped top-k, `routed_scaling_factor`)
at 256 experts instead of 896.

*Results:*

- 208 unique (layer, expert) slots per token out of 6656 — **3.12% of the
  expert set touched per token**;
- **next-token reuse 33.6%** (OLMoE gave 43.5%: reuse *falls* as experts
  get finer-grained, which is the direction that matters for K3's 896);
- concentration is much sharper than OLMoE: top 8.7% of slots cover 50% of
  activations, top 28% cover 80%, top 51% cover 95%;
- LFRU hit rate vs cache fraction: 3% → 29.4%, 6% → 40.6%, 12% → 54.9%,
  24% → 71.9%, 48% → 87.4%. **LRU collapses to 5.1% at the smallest cache
  where LFRU still gets 29.4%** — frequency-first is not a nicety.

*Consequence:* `memplan.py`'s hit curve now comes from this measurement
instead of OLMoE. Projected K3 throughput at 3.01 bits (the Gate 3
operating point) on the 12.78 GB/s internal SSD:

| budget | cache | frac of experts | hit | GB/token | tok/s |
|---|---|---|---|---|---|
| 16 GB | 4.9 GB | 0.6% | 6% | 13.7 | 0.93 |
| 32 GB | 20.9 GB | 2.6% | 25% | 10.9 | 1.17 |
| **64 GB** | **52.9 GB** | **6.5%** | **42%** | **8.5** | **1.50** |
| 128 GB | 116.9 GB | 14.3% | 58% | 6.1 | 2.09 |

So ~1.5 tok/s on the target machine at 3 bits — the higher bit-width from
Gate 3 costs less than feared, because the better-measured hit rate pays
part of it back.

> **Superseded by measurement.** The engine does ~0.3 tok/s on K3 — 0.5
> since read-ahead ([EFFICIENCY.md](EFFICIENCY.md)). Two
> reasons, both in this table's assumptions. It gives a 64 GB machine
> 52.9 GB of cache, where the real trunk is 27.28 GB resident and the
> usable cache is 17 GB. And it counts disk time only, where expert I/O
> is 53.5% of a cold decode step — so even instantaneous reads would
> less than double it. Optimistic by about 5x.
> See [LEARNED.md](LEARNED.md) §12 and §16.

*Caveat:* 256 experts, not 896. The trend from OLMoE (64) to Kimi-Linear
(256) is *falling* per-token reuse but *rising* concentration; which
dominates at 896 is exactly what Gate 2b on K3 itself must answer.

## Gate 3 — quantization quality at 2-3 bit. ✅ RUN 2026-07-27 on
## Kimi-Linear experts, then repeated on real K3 experts. 3 bits is the
## operating point

*Protects:* full conversion + engine integration.
*Test:* VQ2R/VQ3R against round-to-nearest at matched bit budgets on real
experts (`tools/quant_lab.py`), weight error vs the source.
*Kill criterion:* reconstruction error ≫ known-good int4 levels →
raise bits (disk grows) or stop.

*Verdict:* VQ beats RTN decisively below 4 bits, and 3 bits clears the
bar: 19.4% weight error against the int4 baseline's 15.2%, where naive
2-bit RTN collapses at 71.8%. 2-bit VQ stays unsafe at 33%. Repeated on
real K3 experts after the drop — 20.3% at 3 bits, same conclusion — with
the full tables in [LEARNED.md](LEARNED.md) §3 and [K3.md](K3.md).

The same gate killed the format's original centrepiece: the KBVQ shared
low-rank basis costs 0.12 bits and buys 0.3 pp, and loses badly at equal
budget. It is specified, not implemented, and the revive-or-delete
criterion is written into [FORMAT.md](FORMAT.md).

## Gate 4 — engine correctness. ✅ ALL THREE STEPS PASSED 2026-07-27

*Protects:* every optimization built on top of the forward pass.

**Steps 1-2** (kernel vs reference, random weights) — see docs/KDA.md:
3.7e-08 / 4.1e-08 max output diff against fla's `naive_recurrent_kda`,
on both the CPU baseline and the NEON path.

**Step 3** (full forward pass vs oracle, real weights). `src/model.c`
loads a WASTE container and runs Kimi-Linear end to end in C — trunk
dequantized at load, experts read one 4 KiB record at a time and
dequantized on demand, KDA through the dispatch table, MLA with a KV
cache, sigmoid + top-k routing. Diffed against `tools/kimi_ref.py` on the
same container and prompt:

| metric | value |
|---|---|
| max abs diff on logits (magnitude ~15) | 4.0e-05 |
| relative error \|\|c-r\|\| / \|\|r\|\| | 1.58e-06 |
| argmax | identical (17374 = " Paris") |
| top-10 tokens | identical, same order |

Sustained over 12 generated tokens the C engine produces:

> The capital of France is Paris, and the capital of Italy is Rome. The
> capital of

— the same continuation the oracle gives.

*Performance (honest, unoptimized):* 2.15 s/token, 208 expert reads per
token as predicted by Gate 2. The matvec is a naive f32 triple loop with
no OpenMP on this build, weights are dequantized to f32 rather than kept
quantized, and nothing is threaded — this measures correctness, not speed.
Optimizing it is the next body of work, and now it has a reference to stay
correct against.

## Gate 6 — is per-expert bit allocation a real lever? ❌ RUN 2026-07-29.
## Refuted: there is nothing to allocate

*Protects:* a reconversion of all 982 GB, plus the engine work that
variable-width records would need — a per-expert index, a cache that can
hold two record sizes, and a stage count threaded through the hot loop.
This is the gate the working rule exists for: the build is days, the
measurement was an afternoon.

*Test:* `tools/bitalloc_lab.py`. Encode real experts at 1, 2 and 3
residual stages against codebooks fitted per (layer, matrix) exactly as
`convert.py` fits them, and look at delta = err2 − err3 per expert. For a
fixed number of demoted experts the optimal set is provably the smallest
deltas, so the only question is whether delta varies at all.

*Kill criterion:* if the greedy allocation does not beat a random one at
matched average bits, the allocator is a coin flip and the mechanism is
not worth building.

*Verdict: killed.* Delta spreads 1.06–1.15x between experts in a layer,
**1.01x between layers** (1, 5, 23, 46, 69, 92) and 1.09–1.30x between
gate, up and down. Greedy beats random by 0.2–1.4% relative — noise.
Confirmed against a 128-expert layer, in case importance hid in a tail
rather than in the variance, and against Kimi-Linear, in case K3's QAT
was what homogenized the experts. Neither. Each residual stage removes
42% of the remaining error in every expert of both models, because
per-channel amax scaling normalizes them all to the same distribution.

Routing frequency is the one importance signal that is not flat, and it
does not rescue the idea: demoting the cold tail buys disk, which is not
scarce, and 0–2% of the reads, which are. The table is in
[LEARNED.md](LEARNED.md) §20, with the activation-weighted measurement
that would revive it.

## Gate 7 — is the budget resolver's quantum still one token's working set?
## ⏳ OPEN, raised 2026-08-01. Not run

*Protects:* redesigning the rule that picks the default memory budget, and
with it what every user of this engine gets when they pass no `--budget`.
Also protects against the opposite mistake — shipping a smaller default
because a four-token measurement liked it, and finding it worse on the
sessions people actually run.

*Why it is open.* [LEARNED.md](LEARNED.md) §4 established that a cache below
one token's working set keeps nothing alive from one token to the next, and
the hit rate is not low but zero. The resolver is built on it: it steps down
in whole multiples of that working set and takes the largest that fits under
seven eighths of RAM.

```c
ws = one token's working set;             /* 16.2 GB on K3 */
for (k = 3; k >= 1; k--)
    if (floor + ws*k <= cap) { budget = floor + ws*k; break; }
```

That number is still exact. §39 reproduced it: 287 slots, lookahead off,
**0.0% hit**. What changed is that the router lookahead needs a record to
survive *one attention* rather than one token, so the same 287 slots measure
**29.1%** with it on, and a 3.32 GB cache lands within 10% of a 17.32 GB one.

The rule's quantum is the thing to question, not its constant. On K3 there
is nothing between `floor` — cache effectively zero — and `floor + 1x`, a
16.2 GB cache. **It cannot express the size that now works.**

*Test:* `tests/sweep.c` with `cache=`, a fine grid between 287 and 1498
slots, over a **long generation — 200 tokens or more**. §39 used four.
Cross-token reuse is what a large cache buys and it accumulates over a
session, so a short run is precisely the condition that flatters a small
one. One process, interleaved arms, as §38.

*Kill criterion:* if at 200+ tokens a 3–4 GB cache is no longer within ~10%
of `floor + 1x`, the rule stands as written and §39 becomes a note about
short sessions. Only a gap that survives the long run justifies touching the
resolver.

*What acting on it would mean, if it survives.* A smaller default trades
peak throughput for machine margin: on this laptop about 10% of tok/s to
leave the OS ~32 GB instead of ~18. §16, §32 and §33 all identify that
margin as what decides whether the engine meets the paging cliff at all, and
§33 found a budget whose throughput spanned 15x across identical runs
because it sat on the edge of it. Whether 10% of peak is worth roughly
doubling the distance from that edge is a product decision and not a
measurement, so this gate stops at the number and does not recommend one.

*Cost of running it:* one process, no code change, and **45 to 50 minutes**
of K3 time — 200 tokens is about 330 s an arm at 0.6 tok/s, and a four-point
grid twice over is eight of them. An earlier draft of this line said half an
hour, which was arithmetic done hopefully.

The collapsed budgets are deliberately not in the grid: at 0.075 tok/s two
hundred tokens is forty-four minutes *each*, and there is nothing left to
learn there — §39 and §33 have both already refused them.
