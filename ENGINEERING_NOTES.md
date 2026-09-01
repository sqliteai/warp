# WARP AMD ROCm/HIP Backend Engineering Notes

## 0. Hardware & System Context
- **CPU**: AMD Ryzen 5 7600X (6C/12T, AVX-512)
- **GPU**: AMD Radeon RX 9600 XT (16 GB VRAM, ROCm 6.x)
- **Current Host RAM**: 16 GB DDR5 (Upgrade to 48 GB arriving on Day 5)
- **Target OS**: Linux x86_64 / Omarchy (Wayland)
- **ROCm Path**: `/opt/rocm/` (Compiler: `/opt/rocm/bin/hipcc`)

---

## 1. Core Architecture Invariants & Rules
1. **Never break CPU baseline**: `WASTE_BACKEND=cpu` must always work and remain bit-exact.
2. **Zero dynamic loading**: Backends are build-time conditional (`WASTE_ENABLE_HIP=1`) and runtime feature-probed.
3. **No PCIe transfer of static weights**: Trunks and LM Head must remain VRAM-resident.
4. **Instrument before optimizing**: Record real shapes, strides, and group counts from WARP's model execution before tuning.

---

## 2. Kernel & Tensor Shape Registry (Populated during Day 2)
*(To be filled during Day 2 execution tracing)*

| Tensor Name | Shape $[M \times K]$ | Quantization Type | Group Size | Calling Function / Layer | Dispatches / Token |
|---|---|---|---|---|---|
| `lm_head` | TBD | Q8G / Int8 | 128 | `waste_model_step` | 1 |
| `q_proj` / `k_proj` / `v_proj` | TBD | Q4G / Q8G | 128 | `attention_layer` | $N_{\text{layers}}$ |
| `dense_ffn` / `moe_gate` | TBD | Q4G / Q8G | 128 | `moe_layer` | $N_{\text{layers}}$ |

---

## 3. Backend Dispatch Call Graph
```
main() [cli/waste.c]
 └── waste_backend_init() [src/backend.c]
      ├── 1. waste_kda_register_cpu(&waste_k) [src/kda.c] (Base table)
      ├── 2. waste_cpu_features() -> AVX2/AVX512 overrides [src/simd_*.c]
      └── 3. waste_register_hip(&waste_k) [src/hip.hip] (Device override)
```

---

## 4. Daily Execution Log
*(Append findings and decisions daily)*
