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
| **`lm_head.weight`** | $[151936 \times 4096]$ | **Q8G (int8)** / Q4G | 128 | **~593.5 MB** | 1 | **GPU (Day 4: `mvq8_hip`)** |
| **`q/k/v_proj.weight`** | $[4096 \times 4096]$ | Q4G / Q8G | 128 | ~8.0 MB each | $N_{\text{layers}}$ (32–61) | GPU (Day 4: `mvq8_hip` / Post-RAM Q4) |
| **`gate.weight`** (MoE) | $[256 \times 4096]$ | Q8G / F32 | 128 | ~1.0 MB | $N_{\text{layers}}$ | GPU / CPU |
| **`mlp.down_proj`** (Trunk) | $[4096 \times 12288]$ | Q4G / Q8G | 128 | ~24.0 MB | $N_{\text{layers}}$ | GPU (Post-RAM) |
| **Routed MoE Experts** | $[3072 \times 4096]$ | VQ3R / VQ4P (Codebook) | 64/256 | Direct I/O Stream | Top-$K \times N_{\text{layers}}$ | **CPU ecache + Direct I/O** |

---

## 4. Architectural Dispatch Findings (Day 2 Code Review)
1. **The Exact Hook for GPU Acceleration**: 
   - All trunk projections and the LM Head pass through `matvec_t()` $\rightarrow$ `run_rows()` $\rightarrow$ `waste_k.mvq_rows_f32(0, n, arg)`.
   - When `waste_k.on_device == 1`, `run_rows()` dispatches the entire row range $[0, M)$ directly to the device without splitting across the CPU threadpool!
2. **Bandwidth Justification**:
   - `lm_head` is a single dense $[151936 \times 4096]$ multiply moving **593 MB** of weights. On CPU at 50 GB/s DDR5, this costs $\approx 11.8\text{ ms}$. On the RX 9060 XT VRAM ($>400\text{ GB/s}$), this takes **$< 1.5\text{ ms}$**.
3. **Synthetic Harness Design for Day 3**:
   - `tools/hipbw` will be parameterized with these exact shapes: Shape A ($151936 \times 4096$, LM Head) and Shape B ($4096 \times 4096$, Attention).

---

## 5. Empirical Hardware Benchmarks (Day 3 Micro-Harness Data)
Measured via `tools/hipbw` on AMD Radeon RX 9060 XT (GFX1200 / RDNA):

| Measurement Scenario | Target Tensor Shape | Buffer Size | PCIe Transfer Time / Bandwidth | VRAM Read Time / Bandwidth |
|---|---|---|---|---|
| **Dispatch Latency Floor** | Empty Kernel Launch | N/A | N/A | **2.96 µs / dispatch** |
| **LM Head Output Projection** | $[151936 \times 4096]$ (Q8G) | 593.50 MB | 30.58 ms @ 18.95 GB/s | **2.06 ms @ 280.95 GB/s** |
| **Self-Attention Projection** | $[4096 \times 4096]$ (Q8G) | 16.00 MB | 0.75 ms @ 20.83 GB/s | **0.035 ms @ 446.82 GB/s** |
| **Attention Projection (4-bit)**| $[4096 \times 2048]$ (Q4G) | 8.00 MB | 0.45 ms @ 17.38 GB/s | **0.014 ms @ 548.72 GB/s** |
| **MoE Router Gate Projection** | $[256 \times 4096]$ (Q8G) | 1.00 MB | 0.07 ms @ 13.95 GB/s | **0.008 ms @ 127.75 GB/s** |

### Critical Architectural Takeaway from Day 3:
- **Zero-Copy / VRAM Residency is Mandatory**: Synchronously transferring 593.5 MB of `lm_head` across PCIe costs **30.58 ms**, destroying all performance. However, executing directly out of resident VRAM takes only **2.06 ms** (a massive **14.8x speedup** over PCIe streaming and **5.7x speedup** over CPU DDR5!).
- **Low Dispatch Overhead**: The 2.96 µs launch latency confirms that per-layer GPU dispatches (even 60 layers) add less than 0.18 ms total overhead.

---

## 6. Daily Execution Log
- **Day 1**: Implemented `WASTE_ENABLE_HIP=1` build rules in `Makefile`, created `src/waste_hip.h` and `src/hip.hip`, verified AMD Radeon RX 9060 XT device probing and `WASTE_BACKEND=cpu` fallback.
- **Day 2**: Completed execution tracing and call-graph mapping of `src/model.c`. Verified exact tensor geometries and single-dispatch `run_rows` entry point for GPU offload.
- **Day 3**: Built `tools/hipbw` micro-harness. Verified 2.96 µs kernel launch latency, 280–548 GB/s VRAM streaming throughput, and proved empirical necessity of VRAM residency for LM Head.

