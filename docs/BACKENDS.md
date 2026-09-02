# Portability and acceleration backends

Scope requirement (Marco, 2026-07-27): the engine runs on **macOS, Windows
and Linux**; acceleration backends (BLAS, CUDA, Metal, NEON, AVX-512, …)
are selected without burdening a build that does not want them, over a
**universal CPU version always available**. The dispatch discipline follows [sqlite-vector](https://github.com/sqliteai/sqlite-vector).

**No dynamic loading.** An earlier draft resolved accelerators with
`dlopen`/`LoadLibrary`; that was complexity without a matching problem, and
sqlite-vector, the model here, uses none. Backends are chosen by
**conditional compilation plus runtime feature detection**: one binary
still adapts to the CPU it runs on, and a build without CUDA simply has no
CUDA code in it.

## The sqlite-vector pattern, and what we take from it

sqlite-vector keeps one global table of function pointers
(`dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX]`,
`src/distance-cpu.c`). At init it `memcpy`s a fully-populated CPU table in,
then the best backend detected at runtime **overwrites the entries it
implements** — `init_distance_functions()` tries AVX-512 → AVX2 → SSE2 on
x86, NEON on ARM, RVV on RISC-V, with a `force_cpu` escape hatch and the
selected backend's name exposed for introspection (`vector_backend()`).
Detection is careful: `cpu_supports_avx512()` checks CPUID *and* XGETBV, so
a CPU whose OS has not enabled ZMM state is correctly rejected.

WARP adopts all of it:

- one dispatch table, filled with a baseline that is **always compiled in**;
- backends **partially override** — an unimplemented kernel keeps the CPU
  version, so a new backend can start with one hot kernel and grow;
- **runtime detection with OS-support checks**, best-first with fallback;
- **force-CPU escape hatch** (`WASTE_BACKEND=cpu`, or
  `waste_backend_init(WASTE_BE_FORCE_CPU)`) — indispensable for bisecting
  numeric differences;
- **name introspection** (`waste_backend_name()`), surfaced by the CLI.

One difference: sqlite-vector's distance functions all share a signature,
so a 2-D array works. WARP's kernels do not, so the table is a **struct of
function pointers** (`waste_kernels` in
[src/waste_backend.h](../src/waste_backend.h)). Same idea, C-idiomatic for
heterogeneous ops.

## How a backend gets in

*Design, not inventory.* Today's inventory, kept in one place so the rest
of this section can stay about the mechanism:

| backend | file | state |
|---|---|---|
| portable C baseline | `kda.c`, `model.c`, `vq.c` | always compiled in |
| NEON | `kda_neon.c` + inline in `model.c`/`vq.c` | default on ARM |
| AVX2 | `simd_avx2.c` | verified on Linux/x86_64 |
| AVX-512 | `simd_avx512.c` | compiled and dispatched, never executed — CI runner is Zen 3, confirmed 2026-07-29 |
| Metal | `metal.m` | correct, off by default, 22% slower |
| CUDA, BLAS, ROCm | — | not implemented; the flag refuses to build. For CUDA that is a measured decision, not an omission — discrete cards 2026-08-04, coherent memory 2026-08-13, both third-party; see below |
| SVE, RVV | — | not implemented |

Details for each are in the dated sections below. The rest of this one
describes where a new backend plugs in.

**SIMD** (NEON, dotprod/i8mm, AVX2, AVX-512, SVE, RVV): one translation
unit per ISA (`kda.c`, `kda_neon.c`, `kda_avx2.c`, …), each guarded by
`#if defined(__ARM_NEON)`-style fences so a build for another architecture
compiles them away, and each built with its own flags. All the ones valid
for the target architecture are compiled in, and `waste_cpu_features()`
picks at runtime — that is what lets a single x86 binary use AVX-512 on a
machine that has it and AVX2 on one that does not.

**Accelerators** (CUDA, Metal, BLAS, ROCm): build-time options, each
needing a source file that registers it. `src/metal.m` exists; `src/cuda.cu`
and `src/blas.c` do not, and setting their flag stops the build with a
message saying so rather than failing at link time on an undefined
`waste_register_*`. Deleting that check is the last step of adding the
backend it guards.

```
make                     # CPU + SIMD only, zero extra dependencies
make WASTE_ENABLE_METAL=1
make WASTE_ENABLE_CUDA=1
```

A build without `WASTE_ENABLE_CUDA` contains no CUDA code and no link
dependency — which is the whole reason dlopen looked tempting, solved more
simply by not linking it. A build *with* it still calls
`waste_register_cuda()` at init, which probes for a usable device and
**returns NULL to decline** if there is none; the engine then keeps the
backend it already had. Declining is normal, not an error.

Metal deserves a note: it is present on every Mac that can run this engine,
so on macOS it is a plain `#ifdef __APPLE__` decision with no runtime
uncertainty beyond device selection.

## Platform abstraction

Beyond kernels, a handful of calls differ per OS. They live in
[src/platform.h](../src/platform.h) rather than sprinkled through the
engine — one Windows implementation and one POSIX line each, so no call
site carries a branch:

| concern | macOS | Linux | Windows |
|---|---|---|---|
| cache-bypass read | `fcntl(F_NOCACHE)` | `O_DIRECT` | `FILE_FLAG_NO_BUFFERING` |
| positional read | `pread` | `pread` | `ReadFile` + `OVERLAPPED` |
| aligned allocation | `posix_memalign` | `posix_memalign` | `_aligned_malloc` |
| CPU count | `sysconf` | `sysconf` | `GetActiveProcessorCount` |
| physical RAM | `sysctlbyname("hw.memsize")` | `sysconf(_SC_PHYS_PAGES)` | `GlobalMemoryStatusEx` |
| CPU features | `sysctlbyname("hw.optional.arm.*")` | `getauxval(AT_HWCAP/2)` | `IsProcessorFeaturePresent` / CPUID |
| threads | pthreads | pthreads | pthreads (winpthreads) |

Nothing is mapped: the engine reads by offset and never calls `mmap`,
which is what makes the streaming path one shim rather than two.

The expert-streaming path is the one that really cares: Gate H showed
throughput is set by random 12 MB reads with the page cache bypassed, and
that call differs on all three platforms.

## Status

Implemented and verified today:

- `waste_cpu_features()` — x86 CPUID + XGETBV (checks OS-enabled AVX/AVX-512
  state, as sqlite-vector does), aarch64 with per-OS dot-product/i8mm
  detection (macOS `sysctlbyname`, Linux `getauxval`, Windows
  `IsProcessorFeaturePresent`);
- `waste_backend_init()` — CPU baseline, then the best SIMD backend for the
  machine, then any accelerator compiled into this build;
- first kernel family (KDA) wired through the table:
  [src/kda.c](../src/kda.c) is the universal baseline,
  [src/kda_neon.c](../src/kda_neon.c) the NEON specialization.

Verified on this machine (MacBook Pro M5 Pro): `tools/kda_ref.py` passes
against the official reference on *both* paths —

| path | output max\|diff\| | state max\|diff\| |
|---|---|---|
| `WASTE_BACKEND=cpu` (baseline) | 4.10e-08 | 1.79e-07 |
| auto (NEON) | 4.47e-08 | 1.19e-07 |

`waste_cpu_features()` also detects dotprod and i8mm on this CPU, and the
build used to name itself `NEON+i8mm` on the strength of that. Nothing in
the engine emits SDOT or SMMLA, so the name has been cut back to `NEON`:
the backend string reports what the binary *uses*, not what the silicon
offers. Put the suffix back in the commit that adds the kernel.

That equivalence is the contract every future backend must meet: **same
results, only faster.** Windows is built and run in CI as of 2026-07-29;
what it cost, and what is still not claimed, is at the end.

## Machine-specific optimization: measured, not guessed (2026-07-27)

Optimizing started from a profile of the C forward pass on Kimi-Linear
(`WASTE_PROFILE=1`), not from intuition. The order the profile dictated:

| step | s/token | what the profile then said |
|---|---|---|
| first correct version | 2.15 | MoE 71%, of which dequant 37% + matmul 30% |
| NEON f32 dot + thread pool | 0.93 | expert **dequantization 87.5%** |
| fused VQ matvec (no dequant) | 0.22 | expert matmul 67%, KDA 12%, I/O 12% |
| hoist gate/up tables out of the expert loop | **0.18** | expert matmul 56%, I/O 17%, KDA 15% |

**11.9x, and the logits still match the oracle** (max abs diff 4.6e-05,
relative 1.5e-06, argmax and top-10 identical) — the same check is rerun
after every step, because an optimization that changes results is not an
optimization.

The two that mattered:

1. **Never dequantize an expert.** The first version expanded VQ indices
   into f32 weights and then multiplied — 87% of the time. Instead, note
   that `sum_s C_s[i] . x_v` depends only on (stage, code, vector
   position), never on the output row: tabulate it once per matrix and
   every row becomes 3 table lookups per 8 weights. This is
   sqlite-vector's turbo-LUT idea applied to a weight matrix rather than a
   distance. Dequantization dropped from 87.5% to nothing, and the
   remaining 16% under "expert deq" is now purely file I/O.
2. **Hoist what does not vary.** Every routed expert in a layer sees the
   same input and the same per-layer codebooks for its gate and up
   matrices, so those two tables are built once per token instead of once
   per expert — 8x less table-building on two of the three matrices.

**Thread scaling** on this M5 Pro (18 logical cores, 6 performance):

| threads | 1 | 4 | 8 | 12 | 18 |
|---|---|---|---|---|---|
| s/token | 0.45 | 0.20 | 0.20 | 0.18 | 0.18 |

2.5x, flattening after ~4 — consistent with the performance-core count and
with a workload that is becoming memory-bound. The pool splits by row, so
results are bit-identical at any thread count; `WASTE_THREADS` overrides.

**What that table was hiding.** The flattening is real, but part of it was
not the work — it was the dispatch. Measured directly with an empty kernel
on the same machine:

| pool | empty dispatch |
|---|---:|
| 18 threads (6 performance + 12 efficiency, parked) | **54.1 us** |
| 6 threads, all performance, spinning | **1.5 us** |

A decode step dispatches on the order of 150 times, so 54 us is not a
detail. Two things cause it: the pool's workers park on a condvar, and the
barrier at the end of a job waits for the slowest of them — which on this
machine is an efficiency-core thread the scheduler takes ~50 us to run.

So the pool now spins briefly before parking (`WASTE_SPIN`, iterations;
0 restores the pure condvar), and a dispatch only reaches for the slow
group when the job is big enough to hide waking it — 4 MB of weights
touched by default, `WASTE_WIDE_MIN`. Below that it goes to the fast group,
which is already awake. On a machine with one kind of core `n_fast` is 0,
`waste_pool_fast()` is the whole pool, and the threshold changes nothing.

Kimi-Linear, `waste bench`, 150 tokens: **14.41 -> 16.74 tok/s**, and the
inversion §47 recorded — six threads beating eighteen — is gone with it:
16.97 at eighteen against 15.35 at six, where before it was 14.46 against
14.49. The E-cores were never the problem; waking them 150 times a token
was. LEARNED.md §67.

### int8 and SDOT: where the instruction actually fits

The expert matmul's inner loop is a **gather** (`acc += lut[block + s*256 +
code]`), with no multiply — SDOT cannot vectorize it, and ARM has no gather
instruction. Cache-blocking it (`VQ_TILE`, swept: 64 and 128 tie, larger is
worse) bought 0.18 -> 0.15 s/token; the loop is latency-bound on dependent
loads, not bandwidth-bound.

Where SDOT *does* fit is the trunk: those are dense dots, and the trunk is
already stored Q8G (int8 + one fp16 scale per 128 inputs). Keeping it int8
instead of expanding to f32 at load gives three modes:

| mode | s/token | RSS | logits vs oracle | top-10 |
|---|---|---|---|---|
| f32 weights (expand at load) | 0.15 | 9.5 GB | rel 1.6e-06 | identical |
| **int8 stored, f32 math** | **0.13** | **3.9 GB** | **rel 1.5e-06** | **identical** |
| int8 stored, int8 acts + SDOT | 0.13 | 3.9 GB | rel 1.2e-02 | **reordered** |

SDOT does what it promises on its own slice — the trunk phases drop from
0.30 s to 0.16 s, ~1.9x — but the trunk is only ~16% of a token, so Amdahl
caps the end-to-end gain at ~13%, which the f32-math path matches without
quantizing activations. Quantizing them costs four orders of magnitude of
accuracy and reorders the top-10; argmax survived here, but that is luck,
not a guarantee.

**Default: int8 storage with f32 arithmetic.** It keeps the container's
precision exactly, and the memory saving is the part that matters — 5.6 GB
freed is 5.6 GB more expert cache, and Gate 2 says cache is what buys
tokens/sec. `WASTE_SDOT=1` enables the activation-quantized path for anyone
who wants to measure the trade on their own workload.

Still on the table: i8mm/SMMLA for batched prefill (where activations are a
matrix and the accuracy trade is amortized over more work), a NEON pass
over the LUT accumulation, and Metal for the prefill GEMMs.

## Not yet done

*(This list is what was outstanding on 2026-07-27. AVX2, AVX-512, Metal,
the platform I/O wrapper and the CI matrix all landed the following day
and have their own sections below; what remains of it is CUDA, BLAS and
Windows.)*

The CUDA and BLAS backends. Windows was on this list until 2026-07-29;
"the branches are written" had turned out to mean one branch, in
dot-product detection. The measurement, and the port that followed it,
are at the end of this document.

## i8mm/SMMLA for the batched matmul: 2x on its own work, 1.2% overall

SMMLA multiplies a 2x8 int8 tile by an 8x2 tile into a 2x2 int32
accumulator — 32 MACs per instruction against 4 for an fp32 FMA. The
natural target is `mmq_rows`, the batched trunk matmul the chunked
prefill uses for the latent projections, the shared experts and the
dense FFN. `src/model.c` now has `mmq_rows_i8mm`, tiling two weight rows
by two tokens and accumulating per quantization group so each group's
pair of scales applies once.

It works, and it is off by default. Measured on a 19-token chunked
prefill of K3:

| | batched mm | total prefill |
|---|---|---|
| f32 path | 2.38 s (6.0%) | 46.20 s |
| SMMLA | 1.15 s (3.0%) | 45.66 s |

**2.07x on the kernel, 1.2% end to end** — because the batched matmul is
only 6% of prefill to begin with. The time is in the VQ path: LUT apply
37.0% plus LUT build 17.7%, then expert I/O 37.6%. That is where the
next SIMD work belongs, and it is gather-shaped rather than GEMM-shaped,
so SMMLA does not reach it.

It also is not free numerically. SMMLA needs int8 on both sides, so the
activations get quantized per group, which the f32 path deliberately
avoids. Logits move by 6.8e-02 relative — argmax and top-5 held on the
prompt tested, but that is a real change, not fp noise.

Hence two switches, both off: build with `make WASTE_NATIVE=1` (the
default build targets baseline ARM, where `__ARM_FEATURE_MATMUL_INT8` is
not defined and the kernel compiles away) and run with `WASTE_I8MM=1`.
Turning it on by default would trade measurable accuracy for 1.2%. If
the VQ path ever gets fast enough that the batched matmul's share grows,
revisit — and at that point move the kernel into its own translation
unit so runtime dispatch can pick it, instead of requiring a native
build.

## First Linux runs (2026-07-28)

The engine had never been built on Linux. Docker, both architectures,
`tests/run.sh` against the Kimi-Linear container:

| | build | suite | backend | generation |
|---|---|---|---|---|
| Linux arm64 | ok | 12 pass, 0 fail, 4 skip | `NEON` | correct |
| Linux x86_64 | ok | 12 pass, 0 fail, 4 skip | `CPU` | correct |
| macOS arm64 | ok | 17 pass | `NEON` | correct |

*(2026-07-29: re-run on ubuntu:24.04 / gcc 13.3, model-free, after the CI
defects in [LEARNED.md](LEARNED.md) §17 — both Linux targets 17 pass,
0 fail, 8 skip, plus the sanitizer suite and 400 fuzz cases. x86_64 now
names itself `AVX2` rather than `CPU`.)*

Both Linux targets produce "The capital of France is Paris, and the
capital of Italy is Rome" — the same continuation as macOS — and both
pass *engine matches the PyTorch oracle*, so the numerics carry across
platforms and architectures. The four skips need `uv` or the source
weights, neither of which is in the image.

x86_64 names itself `CPU`, not AVX2, even though `waste_cpu_features()
`detects AVX2 there. That is the version string doing its job: detection
is not a kernel, and there is no x86 SIMD in this engine.

Three defects turned up, none in the engine and all invisible from macOS:

- The Makefile added `kda_neon.c` when `uname -m` contained "arm". Linux
  on aarch64 reports **aarch64**, which does not, so the translation unit
  was dropped while `backend.c` — which tests `__aarch64__` — still
  emitted the call. Undefined `waste_kda_register_neon` at link.
- `check_budget.sh` measured peak RSS with `/usr/bin/time -l`: a BSD-only
  flag, and the tool is not in a plain debian image at all. It reported
  the budget as exceeded when it had simply measured nothing. Now
  `getrusage(RUSAGE_CHILDREN)` through python3.
- The first attempt bind-mounted the working tree, so the Linux build
  overwrote the macOS objects and `./waste` became "Exec format error"
  on the host. `Dockerfile.test` copies instead.

O_DIRECT works on Docker's volumes: `waste_stats.direct_io` stays 1 and
no warning prints, while `WASTE_DIRECT=0` produces the fallback and the
"hit rate is partly the kernel's" note. Both directions of that path are
now exercised, which is more than the macOS-only build could do.

CUDA remains untested and untestable here on two counts: there is no
NVIDIA GPU on this machine, and there is still no CUDA source to compile
— `WASTE_ENABLE_CUDA=1` stops the build with a message saying so. *(Both
are still true, and the second is now the reason rather than an accident:
see "Coherent memory: the shape claim does not hold". Since 2026-08-04 the
question itself has been answered on other people's hardware — first on a
discrete card, then on a coherent-memory part that reversed one half of the
answer.)*

## AVX2 and AVX-512 (2026-07-28)

x86 ran pure scalar C until now. Two translation units, `src/simd_avx2.c`
and `src/simd_avx512.c`, each built with its own `-mavx*` flags and
selected by `waste_backend_init` from CPUID — so one binary adapts, which
is why they are separate files rather than `#ifdef`s inside model.c.

They implement the two range kernels that carry the arithmetic:
`mvq_rows_f32` (every trunk projection) and `lutb_range` (the VQ table).
Those moved behind the dispatch table for this, with their argument
structs and the two shared inlines in a new `src/simd.h`. The third hot
path, the VQ gather, gets nothing — no x86 SIMD helps it either, for the
same reason NEON does not.

> **Qualified on 2026-08-13; the last sentence is about VQ3R only.** It was
> written when 3x256 was the only expert format, and for that shape it is
> exactly right: §41 measured why, and the arithmetic is not close. A
> 256-entry stage table is 256 bytes — sixteen vector registers of the
> thirty-two that exist — so it fits no byte-permute primitive on any ISA,
> and blocking cannot make it fit.
>
> **VQ4P broke the premise, not the conclusion's wording.** 4x64 spends the
> same 24 bits per 8-weight vector, and a 64-entry stage table is 64 bytes,
> four registers — which is precisely what `vqtbl4q_s8` addresses. So the
> apply *did* get a NEON kernel (`src/model.c`: four `vqtbl4q_s8` per 16
> rows over a `vld3q_u8` deinterleave, inside a 64-row `VQ_TILE`), and an
> x86 build takes the `else` branch — one row at a time, four scalar table
> loads per row. That is not a narrower kernel, it is the absence of one, on
> the same build where VQ3R's table gets `simd_avx512.c`.
>
> **How much this costs depends on who meets it, and that is narrower than
> issue #32 states.** #32 says VQ4P is what `convert.py` produces by
> default; it is not. `--index-bits` defaults to 8 and `--stages` to 3, so a
> default conversion is VQ3R, and VQ4P has to be asked for explicitly with
> `--index-bits 6 --stages 4 --entries 64`. An x86 build therefore meets the
> scalar apply only when someone requests that shape. The catch is who that
> is: the shape is exactly the one §50 left open as the 64-entry crossover,
> so the people most likely to hit it are the people trying to answer the
> open question — and they would get an ISA difference reported as a
> table-size result.
>
> The candidate is AVX-512 **VBMI's `vpermi2b`**, which indexes 128 bytes
> across two zmm registers — the same primitive with room to spare for 64
> entries. Zen 4 and later, Ice Lake and later. `src/simd_avx512.c` already
> exists and dispatch resolves once at init, so it is a slot to fill and not
> an `#ifdef` in `model.c`.
>
> **Unmeasured in both directions, and worth saying before anyone spends a
> weekend on it:** the path may be bound by the deinterleave rather than by
> the lookup, in which case the table primitive buys much less than it looks
> like it should; and cross-lane byte permutes are not uniformly cheap
> across x86 microarchitectures. This is a hypothesis with an obvious first
> experiment, not a plan — issue #32 tracks it.
>
> One consequence beyond the kernel: **`docs/LEARNED.md` §47's tuning table
> is an ARM result.** `WASTE_XPAR`, `WASTE_XPAR_BATCH`, `WASTE_P6_CHUNK` and
> the thread counts were all measured against the vectorized apply, and on
> x86 the kernel underneath them is a different kernel, so the inversion
> between Kimi-Linear and K3 has never been retested there. `CLAUDE.md`
> carries those settings as engine-wide guidance; read them as ARM guidance
> until someone re-runs them.
>
> If someone writes it, the bar is bit-identity against the portable path
> and not "close" — `-DWASTE_P6_SCALAR` already makes the ARM path
> self-checking, and §43 says why an int8 table raises the bar that far. The
> check that decides it is *SIMD backend matches the CPU baseline*, and it
> has to run on an `--index-bits 6` container: the suite's default
> `WASTE_REF_MODEL` is a VQ3R conversion, and running the suite on the wrong
> container shape is exactly how a load path once stayed broken through
> green runs.

**AVX2 is verified.** On Linux/x86_64 the engine reports `backend AVX2`,
the suite is 12 passed / 0 failed, and *SIMD backend matches the CPU
baseline* passes — that check runs `WASTE_BACKEND=cpu` against the
dispatched path and compares logits, so it is exactly the claim that
matters. It is now "within fp noise" rather than bit-identical, because
AVX2 accumulates in a different order.

**AVX-512 is compiled and dispatched, never executed.** No machine here
has it: the container's CPU reports `AVX512F=0` in CPUID leaf 7, and so
does `qemu-x86_64 -cpu max` under Rosetta, whose XCR0 leaves the opmask
and ZMM bits clear. The detection is doing the right thing by declining —
that much is confirmed — but the kernels themselves have never run an
instruction. Treat the first AVX-512 machine as the test, and expect the
same "matches the CPU baseline" check to be the thing that decides it.

No performance numbers from any of this: x86 here is emulated, so timings
would measure Rosetta.

### Emulation cannot close this gap (2026-07-29)

Asked directly whether Docker could execute the AVX-512 path, and the
answer is no, for a reason worth writing down rather than re-deriving.
A CPUID probe inside `--platform linux/amd64` reports:

```
cpu        : VirtualApple @ 2.50GHz
AVX2       : 1
AVX512F    : 0
XCR0       : 0x7  (SSE 1 YMM 1 opmask 0 ZMM_hi 0 hi16 0)
```

That brand string is **Rosetta 2**, not QEMU — Docker Desktop on Apple
silicon translates x86 with Rosetta by default, which is also why
`QEMU_CPU=max`, `Skylake-Server-v4` and `Sapphire-Rapids` all change
nothing: the variable means nothing to a translator that is not QEMU.
Rosetta implements AVX2 and not AVX-512, and `XCR0 = 0x7` says the
opmask and ZMM state is not enabled, so an AVX-512 instruction could not
retire even if it decoded. Switching Docker to QEMU would not obviously
help either: TCG has implemented AVX/AVX2 for some releases and AVX-512
is not among them.

So the gap needs hardware. The cheapest that might already exist is the
CI runner, and the workflow now prints the CPU model and its avx512
flags before it builds — if a hosted runner has them, `waste version`
on the next push says `AVX-512` instead of `AVX2` and the *SIMD backend
matches the CPU baseline* check becomes the confirmation, at no cost.
If it does not, this stays open until someone runs the suite on an Ice
Lake, a Sapphire Rapids or a Zen 4. It did not; see below.

### The runner answered, and the answer is no (2026-07-29)

The workflow asked, and the hosted x86 runner closed the cheap option:

```
model name	: AMD EPYC 7763 64-Core Processor
  avx2: yes
  avx512f: no   avx512bw: no   avx512dq: no   avx512vl: no
WASTE 0.6.0 (container v0, backend AVX2, crc32 slice8, x86_64)
```

An EPYC 7763 is Milan, which is Zen 3 — AMD added AVX-512 in Zen 4, so
this is the generation before. The Windows x86_64 job reports `backend
AVX2` as well. The AVX-512 kernels are therefore still compiled, linked,
dispatched past and never entered, on every target this project builds
for.

One consequence is worth stating, because the check invites exactly the
wrong reading: *SIMD backend matches the CPU baseline* **passes** on
linux-x86_64 and says nothing whatsoever about AVX-512. It compares the
dispatched path against `WASTE_BACKEND=cpu`, and the dispatched path
there is AVX2. A green CI is not evidence about the AVX-512 kernels;
only the flags line above says which backend was under test.

So this stays open, and the hardware it wants is now specific: Zen 4 or
later on AMD, Ice Lake or Sapphire Rapids or later on Intel. A larger
hosted runner class would also reach it, at cost.

## Metal: it works, and it loses (2026-07-28)

`src/metal.m` implements `mvq_rows_f32` — the quantized matvec every trunk
projection and the output head go through. Built with
`make WASTE_ENABLE_METAL=1`; the shader is compiled from source at first
use, because the offline Metal compiler ships with Xcode and not the
Command Line Tools.

The design point that makes it worth trying at all: **no copies**. The
engine moves ~17 GB of weights per token, so a backend that staged host
memory to the device would lose before it started. Apple Silicon has
unified memory, so trunk tensors are allocated page-aligned — the
`waste_dio_alloc` that O_DIRECT already needed, widened from 4 KiB to the
16 KiB page — and wrapped with `newBufferWithBytesNoCopy`. The GPU reads
the same physical pages the CPU does.

It is correct: logits match the CPU baseline to 8.1e-06, argmax and top-5
unchanged, on both Kimi-Linear and K3.

It is also slower, on K3, 5 decode steps:

| | total | kda | mla | lm_head |
|---|---|---|---|---|
| CPU/NEON | 15.93 s | 1.78 | 0.39 | 0.03 |
| Metal | 19.43 s | 3.56 | 0.67 | 0.11 |

**22% slower.** The first kernel put one thread on a whole row, which is
the obvious mapping and the wrong one — adjacent threads read addresses
`rowbytes` apart, so every load is its own cache line. Rewriting it as one
threadgroup per row with strided, coalesced loads and a threadgroup
reduction took the total from 21.28 s to 19.43 s. Still behind.

`lm_head` is the clearest case, because it is one dispatch per token over
1.17 GB of int8 weights: CPU 6 ms, GPU 22 ms. That is 195 GB/s against
53 GB/s — the CPU path is already running at the machine's memory
bandwidth, and this is a bandwidth-bound matvec. There is no headroom for
the GPU to take, and it pays a synchronous round-trip per call on top.

The conclusion is about shape, not about Metal. This engine issues several
hundred small dependent matvecs per token and waits for each; that is the
worst possible fit for an accelerator. Making the GPU pay would mean
moving the whole forward pass on-device so there is one dispatch per layer
instead of six, with the residual stream never returning to the host — a
different engine, not a backend. The code stays, off by default, because
it is correct and because that argument should be re-run if the CPU path
ever stops being bandwidth-bound.

**That re-run has since happened, twice, on NVIDIA hardware, and what is
left of this paragraph is smaller than it looks.** The reasoning under it —
no headroom, plus a round-trip per call — does not transfer to a discrete
card unchanged (next section). And the conclusion itself, "a different
engine, not a backend", was a *shape* stated from a *mechanism*, and does
not survive a vehicle where the mechanism is absent (2026-08-13 section
below). Read all three together; the last one is the current position.

## CUDA: the gates, answered off-repo (2026-08-04)

Everything here is **third-party and not reproduced on this machine**,
which is why issue #11 was written as a set of gates for someone else to
run rather than as a plan. `ssarthak15` ran gate 1 on a dual EPYC 7J13
bare-metal node with K3; `fab2s` ran gates 1, 2 and 3 on a Ryzen 9 9900X
(Zen 5, AVX-512) with an RTX 5060 Ti, on Kimi-Linear-48B. Method, sample
spreads and the byte-accounting checks that make them trustworthy are in
`docs/LEARNED.md` §48 and §50; what follows is only what it means for this
document.

**The round-trip clause does not transfer.** On the RTX 5060 Ti (sm_120,
15.5 GB usable, PCIe Gen5 x8), against that machine's own CPU time for the
same tensor:

| | CPU (AVX-512) | GPU, full round trip |
|---|---|---|
| `lm_head`, 383 MB Q8G | 6.600 ms, 58.1 GB/s | **1.056 ms** |

6.25x, with the kernel itself at 85-90% of theoretical VRAM bandwidth and
an empty-kernel dispatch floor of 4.39 us — so 27 dispatches per token cost
0.1 ms. Dispatch overhead is not what would stop a restructured engine. It
is what stops *this* one: at the current shape, 624 dispatches per token is
7.13 ms of pure overhead.

**The bandwidth clause holds for the kernel and not for the step.** Both
contributors reproduce "the CPU matvec runs at the machine's bandwidth" on
x86 — 98.7% of a matched host scan on EPYC/AVX2, 73-76% of ceiling on
Zen 5/AVX-512. But `lm_head` is 0.2% of a K3 decode step and 4.9-7.3% of a
Kimi-Linear one. Varying core clock and DRAM speed independently puts ~84%
of the step on the core clock and ~23% on DRAM: the step is bound by the
LUT path's dependent gather chains, which is *latency*, not bandwidth and
not vector width.

**What actually decides it is a term Metal never had: the expert stream.**

| | |
|---|---|
| H2D, 544 MiB = one token's routed experts | **19.8 ms at 28.8 GB/s** |
| the same bytes, already in host RAM | 63-80 GB/s |
| expert bank vs usable VRAM | 16.5 GiB vs 15.5 GiB — does not fit |

28.8 GB/s is 90% of Gen5 x8, so the link is healthy; it is simply **slower
than the host's own memory**. A discrete card is structurally on a worse
path to the bytes this engine spends its time moving. GPUDirect Storage
does not rescue it either — `nvidia_fs` was unavailable on that host, and
at a 98.1% cache hit rate it could touch ~2% of reads regardless.

With the VQ expert matmul implemented and measured on-device rather than
substituted from an int8 kernel (13.9 ms/token, not the 3.7 ms a `matvec_q8g`
stand-in implies), a fully restructured on-device engine projects to
**~37.5 ms/token against a measured 62.7 ms — about 1.7x**, before KDA and
MLA, which are a further 24% of CPU time and would need kernels of their
own.

**And the configuration that would remove the transfer term fails the
quality gate.** A 2-stage container's expert bank is 11.04 GiB and fits
VRAM with 3.25 GiB to spare, but its reconstruction error is 33% against
VQ3R's 19.5% — `verify_container.py` FAILs it at the 0.30 threshold, which
independently reproduces `docs/GATES.md` Gate 3 on other hardware. The only
shape that fits does not pass; the shape that passes misses VRAM by
2.27 GiB.

**So nothing changes here.** `src/cuda.cu` still does not exist,
`WASTE_ENABLE_CUDA=1` still stops the build, and filling the
`waste_backend` slots with CUDA kernels would still reproduce the Metal
result on different silicon. What has changed is that this is now a
measured position rather than an argument by analogy from Apple silicon.

**What would move it**, stated so it can be tested rather than argued: a
card with enough VRAM to hold the experts resident — VQ3R's 17.77 GiB total
fits a 24 GB part comfortably — which turns 19.8 ms of per-token H2D into a
one-time load and deletes the deciding row. That is still one dispatch per
layer with the residual never returning to the host, i.e. still a different
engine and not a backend. The GPU VQ-decode throughput measured above
applies to it unchanged.

> **Superseded in part on 2026-08-13.** The condition this paragraph names
> was met from the other side, and the sentence about filling the
> `waste_backend` slots is false on that class of vehicle. The discrete-card
> answer above is unaffected. See the next section.

## Coherent memory: the shape claim does not hold (2026-08-13)

**Third-party, third contributor, not reproduced here** — the same standing
caution as the two sections above. `mccoyspace` ran an incremental CUDA
offload on an NVIDIA **GB10**, a coherent unified-memory part rather than a
discrete card, on K3 and Kimi-Linear. Numbers, contracts and the negative
results are in `docs/LEARNED.md` §61; what follows is only what it means for
this document.

**Two sentences in this file were wrong, and they were wrong in the same
way.** Metal concluded that making a GPU pay "would mean moving the whole
forward pass on-device ... a different engine, not a backend", and the
section above concluded that filling the `waste_backend` slots "would still
reproduce the Metal result on different silicon." Both took a *mechanism*
that had been measured — synchronous launch and a round-trip per call, over
several hundred small dependent matvecs — and restated it as a *shape*.
Coherent memory removes the round trip without removing the dependency, so
the mechanism is absent and the shape claim does not follow.

What was actually done there was incremental and kernel-class — KDA, then
the dense projections, then the VQ gather — with **attention state, routing
and final expert accumulation left on the CPU**. That is much closer to
filling slots than to a rewritten engine, and it paid: 0.34-0.35 → 0.902
tok/s on K3 under a byte-identical-logits capture, 0.637 tok/s at 121.35 W
on a held-out set.

**Read the held-out figure against this machine, not against that host's
CPU.** The README publishes 0.45-0.62 tok/s for K3 on an M5 Pro laptop, so
0.637 clears the top of that band by about 3% — the same order as the band's
own width. The 2.6x is real and it is the right ratio for asking whether the
GPU paid *on that host*; it is the wrong one for asking whether the hardware
buys anything over a laptop. Both readings belong here.

**What does not change:**

- **The discrete-card answer** of the previous section. PCIe is still
  slower than host RAM and a 16 GB card still does not hold a 16.5 GiB
  expert bank. GB10 is not evidence about a 5060 Ti.
- **The build guard.** `src/cuda.cu` still does not exist and
  `WASTE_ENABLE_CUDA=1` still stops the build. Nothing here is a decision
  to accept a CUDA backend.
- **The reason it is not one.** This project cannot execute a line of that
  code — no NVIDIA hardware, no way to regress-check a numerical contract
  per release. Adding a surface no machine here can reach is a maintenance
  question, not a kernel question, and it is sharpened rather than softened
  by issue #36: the suite's non-synthetic path does not run in CI on any
  platform either. That is what has to be answered before code, and
  answering it afterwards is how a project acquires a backend nobody can
  fix.

The design detail worth carrying into any such discussion: CUDA emitted one
partial per selected expert and the **CPU reduced them in original router
order**, which is what made byte-identical logits reachable and route
invariance an interpretable gate rather than a tolerance.

## CI

`.github/workflows/ci.yml` builds on linux-x86_64, linux-arm64 and
macos-arm64, plus two jobs the matrix does not cover: a Metal build (off
by default, so otherwise never compiled anywhere) and a guards job that
checks the unimplemented accelerator flags still refuse with a message
and that every source file carries its SPDX header.

It exists because of what the first Linux run found. Both defects — the
Makefile missing `aarch64`, and a peak-RSS check using a BSD-only flag of
a tool debian does not ship — were invisible from macOS and would have
been caught by a single push.

**What it cannot do.** The end-to-end engine checks need a container, and
the smallest is 19 GB, so CI runs only what is independent of one: the
three builds, and the kernels against their PyTorch references — enough
to catch a numerical regression in SiTU, the decay gates, AttnRes or KDA
on a platform nobody develops on. Verified in a Linux arm64 container:
with `uv` present those two checks run and pass; `tests/run.sh` skips the
rest and exits 0.

**The gap is now closed**, by `tools/make_test_container.py`: a valid
1 MB container of deterministic noise, stdlib-only so it needs neither
torch nor the converter's C extension. `tests/run.sh` builds one whenever
no real container is given, which took the model-less run from 4 checks
to 13 at the time — chunked prefill against token-at-a-time, int8 storage
against f32, the SIMD backend against the CPU baseline, the expert cache
against no cache, session-state round-trip, the RAM plan and the
format-version guard all now run on every platform and in CI.

Counts drift as checks are added, so take them from a run rather than
from here. As of 2026-08-02 the suite is **42 checks**; without a container it is
**29 pass / 0 fail / 11 skip**.

Those skips are what still needs real weights: the oracle diff (those
logits belong to actual Kimi-Linear weights), the container round-trip
against the source shards, and anything that drives the CLI with text —
the synthetic container deliberately carries no tokenizer.

## What Windows actually costs (2026-07-29)

"The Windows branches are written" was generous: there is one, in
dot-product detection. So the question was asked properly — cross-compile
with MinGW-w64 and count what breaks:

| translation unit | for Windows |
|---|---|
| `backend.c` `ecache.c` `image.c` `kda.c` `kda_neon.c` | compiles |
| `tokenizer.c` `version.c` `vision.c` `waste.c` `cli/main.c` | compiles |
| `simd_avx2.c` `simd_avx512.c` | compiles with their ISA flags |
| `model.c` `vq.c` | **stop on one undeclared constant** |

Ten of thirteen build unchanged, and the gap is three POSIX calls:

| call | where | Windows equivalent |
|---|---|---|
| `sysconf(_SC_NPROCESSORS_ONLN)` | `threads.h` | `GetSystemInfo().dwNumberOfProcessors` |
| `posix_memalign` | `ecache.c` | `_aligned_malloc` — and `_aligned_free` on the way out, which `free()` cannot do |
| `pread` | `model.c` | `ReadFile` with `OVERLAPPED`, or the cache-bypass open below |

Two of those are only *warnings* under gcc 13, which is the trap: an
implicit declaration compiles and fails at link, so "it builds" would
have been the wrong question. MinGW supplies pthreads through
winpthreads, so the thread pool needs nothing.

That is a small port — three shims plus `FILE_FLAG_NO_BUFFERING` for the
page-cache bypass, which is where the real work is, since the whole
expert-streaming argument rests on it.

## The port (2026-07-29)

The estimate above was right about the three calls and wrong about the
size, because it counted what fails to compile. Two things that compile
cleanly were the actual work:

**`long` is 32 bits on Windows.** LLP64 keeps `int` and `long` at 32 and
widens only pointers, so every file offset in the engine — `pread_all`'s
argument, `waste_tensor.file_off`, the manifest's `off` and `scale_off`
through `js_int`, a bank's `bytes` before it is divided into records —
was a silent 2 GB truncation on a format whose small container is 17 GB.
None of it is a compile error and none of it is visible on a machine
where `long` is 64 bits. They are `int64_t` now.

**The archiver is part of the target.** `ar rcs` on macOS accepts PE
objects and writes a 96-byte archive without a word of complaint; the
link then fails with a page of undefined references naming every public
symbol, which reads like a source problem and is not one. `AR` follows
`CC`.

What the port is: [src/platform.h](../src/platform.h), which holds the
six calls that are not POSIX — positional read, aligned allocation, CPU
count, file size, cache-bypass open — with a Windows implementation and a
one-line POSIX one, so no call site branches. `pread` becomes `ReadFile`
with an `OVERLAPPED` offset, which does not move the shared file pointer
and so keeps the property the expert cache needs. `posix_memalign`
becomes `_aligned_malloc`, whose pointer must not be passed to `free()` —
that is heap corruption rather than a leak, which is why the allocation
and its release are a pair and neither is called directly. MinGW supplies
pthreads through winpthreads, so the thread pool needed nothing.

**CI builds it and then runs it**, in two jobs: cross-compile with
MinGW-w64 on a Linux runner, then execute the artifacts on
`windows-latest` against a synthetic container — the expert records
through the C structs, `waste info` and `waste plan`, and the forward
pass sequential against chunked. The old `windows-portability` tripwire
counted unresolved POSIX calls without linking; the build now subsumes
it, since a fourth dependency fails it outright.

What is still not claimed: MSVC (the sources are GNU C), ARM64 Windows,
and the bypass under load. CI reports whether Windows granted
`FILE_FLAG_NO_BUFFERING` on the runner's filesystem, which is a different
claim from a hit rate measured against a container that does not fit in
RAM.

---

# AMD ROCm / HIP Backend (2026-09-02)

## Architectural Motivation & Overview

On modern hybrid MoE inference architectures (e.g. Kimi-Linear, GLM-5, K3), the computational pipeline divides into two distinct operational profiles:
1. **Trunk Projections & Output Layer (`lm_head`)**: Dense, wide matrix-vector multiplications ($[151936 \times 4096]$ in `lm_head`, moving ~593.5 MB of weights per token). On CPU DDR5 (50–65 GB/s), this single layer consumes ~11.8 ms.
2. **Sparse MoE Experts**: Routed on-demand via NVMe Direct I/O and CPU memory cache (`ecache`).

The **AMD ROCm/HIP backend** (`src/hip.hip`, `src/waste_hip.h`) introduces zero-overhead GPU acceleration for quantized trunk matvecs (`mvq8` Q8G and `mvq4` Q4G) while strictly preserving WARP's lightweight C architecture and CPU fallback.

## Key Design Principles

1. **Persistent VRAM Residency (Zero Per-Token PCIe Copies)**:
   Static weights are uploaded to VRAM once on initial access and cached in a thread-safe 8192-slot linear probing table (`hip_get_or_upload_vram`). Only the transient activation vector ($x \in \mathbb{R}^K$, 16 KB) and output logits (600 KB) traverse PCIe during token decoding.
2. **Dynamic Wavefront Reductions (`__shfl_down`)**:
   Kernels use in-register warp shuffles that automatically adapt to AMD's dynamic wave size (`warpSize = 32` on RDNA / `64` on CDNA), eliminating shared memory bank conflicts and barrier latency.
3. **Clean Memory Lifecycle**:
   When a model is unloaded or reloaded, `waste_hip_release_host_buffers()` is called from `waste_backend_release_host_buffers()`, freeing all resident GPU allocations with zero memory leaks.
4. **Deterministic CPU Fallback**:
   Setting `WASTE_BACKEND=cpu` completely bypasses HIP at runtime without re-compilation.

## Empirical Performance (AMD Radeon RX 9060 XT - 16 GB VRAM)

Measured via standalone micro-harness `tools/hipbw` and `tools/bench_rocm.py`:

| Operation | Shape & Quantization | PCIe H2D Transfer | Resident VRAM Execution | Speedup vs PCIe Transfer |
|---|---|---|---|---|
| **`lm_head.weight`** | $[151936 \times 4096]$ (Q8G, 593.5 MB) | 30.58 ms (18.95 GB/s) | **2.06 ms (280.95 GB/s)** | **14.8x** |
| **Attention Projection** | $[4096 \times 4096]$ (Q8G, 16.0 MB) | 0.75 ms (20.83 GB/s) | **0.035 ms (446.82 GB/s)** | **21.4x** |
| **Attention Projection (4-bit)** | $[4096 \times 2048]$ (Q4G, 8.0 MB) | 0.45 ms (17.38 GB/s) | **0.014 ms (548.72 GB/s)** | **32.1x** |
| **Kernel Launch Floor** | Empty Dispatch | N/A | **2.96 µs / dispatch** | — |

## Numerical Parity & Correctness

Verified via automated 50-step recurrent parity test (`tests/test_hip_parity.c`):
- **`mvq8_hip` (8-bit Q8G)**: Max relative drift $\le 4.36 \times 10^{-4}$ (within FP32 floating-point accumulation noise).
- **`mvq4_hip` (4-bit Q4G)**: Max absolute difference $\le 7.15 \times 10^{-7}$ (bit-exact precision).

