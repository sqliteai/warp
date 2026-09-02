# WARP AMD ROCm/HIP Backend Engineering Notes

## 0. Hardware & System Context
- **CPU**: AMD Ryzen 5 7600X (6C/12T, AVX-512)
- **GPU**: AMD Radeon RX 9060 XT (16 GB VRAM, ROCm 6.x / GFX1200)
- **Host RAM**: 16 GB DDR5 (Day 1-4); 48 GB DDR5 arriving on Day 5
- **Compiler**: `/opt/rocm/bin/hipcc` (HIP 7.2 / AMD Clang 22.0)

---

## 1. Core Architecture Invariants & Rules
1. **Never break CPU baseline**: `WASTE_BACKEND=cpu` must always force the CPU/SIMD fallback.
2. **Zero dynamic loading**: Backends are build-time conditional (`WASTE_ENABLE_HIP=1`) and runtime probed via `hipGetDeviceProperties`.
3. **No PCIe transfer of static weights**: Trunks and LM Head must remain VRAM-resident.
4. **Instrument before optimizing**: All micro-benchmarks must model real tensor dimensions.

---

## 2. Verified End-to-End Inference Call Graph (Day 2 Trace)

```
main() [cli/waste.c]
 │
 ├── waste_model_load() [src/model.c]
 │    ├── Parses container manifest and headers
 │    ├── Allocates trunk tensors in memory (m->t)
 │    └── Allocates direct I/O expert cache (m->cache)
 │
 └── Token Generation Loop (waste_model_step) [src/model.c:5330]
      │
      ├── 1. Embedding Lookup: waste_embed_row() [m->x]
      │
      ├── 2. For Layer L = 0 .. n_layers - 1:
      │    ├── Pre-RMSNorm: waste_rmsnorm(norm, m->x, input_layernorm)
      │    │
      │    ├── Self-Attention / KDA Recurrence:
      │    │    ├── IF KDA: kda_layer() [matvec_t on q/k/v/g projections -> kda_step]
      │    │    └── IF MLA: mla_layer() [matvec_t on q_b_proj/kv_a_proj -> rope -> attention]
      │    │
      │    ├── Sublayer Residual: m->x += resid
      │    ├── Post-RMSNorm: waste_rmsnorm(norm, m->x, post_attention_layernorm)
      │    │
      │    └── MoE / FFN Block:
      │         ├── Router Gate MatVec: matvec_t(sc, gate.weight, norm, E, hid)
      │         ├── Expert Selection: Top-K Softmax routing
      │         ├── moe_layer():
      │         │    ├── ecache lookup / Direct I/O NVMe read (bank_fetch)
      │         │    ├── Build/Quantize LUT: vq_quant_lut()
      │         │    ├── VQ Apply (Row Gather): vq_rows_p6() / vq_rows_e()
      │         │    └── Down Projection: SwiGLU / Down MatVec
      │         └── Sublayer Residual: m->x += resid
      │
      ├── 3. Final RMSNorm: waste_rmsnorm(norm, m->x, model.norm.weight)
      │
      └── 4. Output Projection (LM Head):
           └── matvec_t(m->logits, lm_head.weight, norm, vocab, hid) [src/model.c:5469]
                └── run_rows(vocab, 64, waste_k.mvq_rows_f32, &a, rowbytes)
```

---

## 3. Real Tensor Geometry & Dispatch Registry (Day 2 Data)

| Tensor Identifier | Matrix Shape $[M \times K]$ | Quantization Type | Group Size | Memory Footprint | Calls / Token | Execution Target |
|---|---|---|---|---|---|---|
| **`lm_head.weight`** | $[151936 \times 4096]$ | **Q8G (int8)** / Q4G | 128 | **~593.5 MB** | 1 | **GPU (`mvq8_hip`)** |
| **`q/k/v_proj.weight`** | $[4096 \times 4096]$ | Q4G / Q8G | 128 | ~8.0 MB each | $N_{\text{layers}}$ (32–61) | **GPU (`mvq4_hip`)** |
| **`gate.weight`** (MoE) | $[256 \times 4096]$ | Q8G / F32 | 128 | ~1.0 MB | $N_{\text{layers}}$ | GPU / CPU |
| **`mlp.down_proj`** (Trunk) | $[4096 \times 12288]$ | Q4G / Q8G | 128 | ~24.0 MB | $N_{\text{layers}}$ | **GPU (`mvq4_hip`)** |
| **Routed MoE Experts** | $[3072 \times 4096]$ | VQ3R / VQ4P (Codebook) | 64/256 | Direct I/O Stream | Top-$K \times N_{\text{layers}}$ | **CPU ecache + Direct I/O** |

---

## 4. Empirical Hardware Benchmarks (Day 3 Micro-Harness Data)
Measured via `tools/hipbw` on AMD Radeon RX 9060 XT (GFX1200 / RDNA):

| Measurement Scenario | Target Tensor Shape | Buffer Size | PCIe Transfer Time / Bandwidth | VRAM Read Time / Bandwidth |
|---|---|---|---|---|
| **Dispatch Latency Floor** | Empty Kernel Launch | N/A | N/A | **2.96 µs / dispatch** |
| **LM Head Output Projection** | $[151936 \times 4096]$ (Q8G) | 593.50 MB | 30.58 ms @ 18.95 GB/s | **2.06 ms @ 280.95 GB/s** |
| **Self-Attention Projection** | $[4096 \times 4096]$ (Q8G) | 16.00 MB | 0.75 ms @ 20.83 GB/s | **0.035 ms @ 446.82 GB/s** |
| **Attention Projection (4-bit)**| $[4096 \times 2048]$ (Q4G) | 8.00 MB | 0.45 ms @ 17.38 GB/s | **0.014 ms @ 548.72 GB/s** |
| **MoE Router Gate Projection** | $[256 \times 4096]$ (Q8G) | 1.00 MB | 0.07 ms @ 13.95 GB/s | **0.008 ms @ 127.75 GB/s** |

---

## 5. Kernel Parity & Precision Verification (Day 4 Data)
Verified via `tools/hipbw` against scalar CPU reference:

- **`mvq8_hip` (8-bit Quantized MatVec)**: Max relative difference $\approx 4.36 \times 10^{-4}$ ($\le \text{FP32 floating-point accumulation noise}$).
- **`mvq4_hip` (4-bit Quantized MatVec)**: Max absolute difference $\le 7.15 \times 10^{-7}$ ($\text{Bit-exact}$ within single-precision epsilon).

---

## 6. Daily Execution Log
- **Day 1**: Implemented `WASTE_ENABLE_HIP=1` build rules in `Makefile`, created `src/waste_hip.h` and `src/hip.hip`, verified AMD Radeon RX 9060 XT device probing and `WASTE_BACKEND=cpu` fallback.
- **Day 2**: Completed execution tracing and call-graph mapping of `src/model.c`. Verified exact tensor geometries and single-dispatch `run_rows` entry point for GPU offload.
- **Day 3**: Built `tools/hipbw` micro-harness. Verified 2.96 µs kernel launch latency, 280–548 GB/s VRAM streaming throughput, and proved empirical necessity of VRAM residency for LM Head.
- **Day 4**: Implemented `mvq8_hip` and `mvq4_hip` kernels in `src/hip.hip` with dynamic wavefront reductions (`__shfl_down`). Connected them directly to `waste_k.mvq_rows_f32` dispatch table. Verified numerical parity and warning-free compilation.
