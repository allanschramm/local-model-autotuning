# Session 2026-06-29: BeeLlama, TCQ, CopySpec, DFlash & IQ3 experiments

## Goal
Push Ornith 9B and 35B TPS/score beyond current bests (9B: 0.580/52.2 TPS, 35B: 0.555/31.5 TPS) using BeeLlama features, hyperparam tuning, and IQ3 quantization.

## Hardware
- GPU: RTX 4060 8 GB VRAM
- CPU: AMD Ryzen 7 5800X
- RAM: 24 GB
- OS: WSL2 Ubuntu 24.04

## Results

### 9B Ornith
| Config | TPS | Score | VRAM | Verdict |
|--------|-----|-------|------|---------|
| Q4_K_M baseline (stock fork) | **52.2** | **0.580** | 7.4 GB | ✅ Best |
| BeeLlama q4_0 baseline | 41.7 | 0.375 | 8.0 GB | ❌ -20% slower |
| BeeLlama turbo3 | 31.4 | 0.375 | 7.4 GB | ❌ TPS -40% |
| BeeLlama TCQ turbo3_tcq | 0.0 | N/A | — | ❌ HTTP 500 at 131k ctx |
| BeeLlama CopySpec | 45.1 | 0.250 | 7.8 GB | ⚠️ Marginal |
| TEMP=0.7 | 51.9 | 0.325 | 7.8 GB | ❌ Worse |
| TOP_P=0.9 | 51.3 | 0.425 | 7.7 GB | ⚠️ LCB improved, overall worse |
| REPEAT_PENALTY=1.05 | 48.2 | 0.490 | 7.9 GB | ❌ Full 10-task: 0.49 < 0.58 |

**9B hyperparam tuning conclusion:** No hyperparam (temp, top_p, top_k, repeat_penalty) improves score beyond 0.58. The model's capability ceiling is reached.

### 35B Ornith
| Config | TPS | Score | VRAM | Verdict |
|--------|-----|-------|------|---------|
| Q4_K_M n-cpu-moe 32 (stock fork) | **31.5** | **0.555** | 7.7 GB | ✅ Best |
| Q4_K_M n-cpu-moe 34 | 29.2 | 0.250 | 7.1 GB | ❌ More CPU = slower |
| BeeLlama q4_0 baseline | 26.2 | 0.250 | 7.9 GB | ❌ BeeLlama overhead |
| BeeLlama CopySpec | 22.0 | 0.250 | 7.9 GB | ❌ Worse |
| BeeLlama DFlash (GPU) | 3.3 | — | — | ❌ Draft competes for GPU |
| BeeLlama DFlash (CPU offload) | OOM | — | — | ❌ 16 GB RAM overflow |
| IQ3_M n-cpu-moe 32 | 25.4 | 0.375 | — | ❌ Slower decode kernel |
| IQ3_M n-cpu-moe 30 | 26.7 | 0.375 | — | ❌ Slower |
| IQ3_M n-cpu-moe 28 | 27.3 | 0.250 | — | ❌ Slower |

### BeeLlama Observations
- **BeeLlama baseline is 20% slower than stock fork** on both 9B and 35B for pure inference (no DFlash/CopySpec).
- **TCQ turbo3_tcq crashes** at large context (HTTP 500). Not compatible with hybrid SSM+Attention architectures (Qwen35).
- **BeeLlama server crashes** between benchmark rounds on LCB/BigCode (connection refused). Server stability issue.
- **CopySpec adds overhead** that outweighs gains on MoE models.
- **DFlash draft** (Qwen3.5-35B-A3B) is architecture-compatible with Ornith (same qwen35 arch, 40 layers) but:
  - On GPU: draft competes for compute, TPS drops to 3.3
  - On CPU: cross-attention buffer at 8192 ctx overflows 16 GB RAM

### IQ3_M 35B
- File: 15.7 GB (vs Q4_K_M 19.7 GB)
- **Consistently slower TPS than Q4_K_M** at every n-cpu-moe tested (32, 30, 28)
- Root cause: IQ3 CUDA decode kernel less optimized than Q4_K_M on RTX 4060
- Quality degraded: MBPP+ dropped from 0.9 to 0.0 in validation

### n-cpu-moe Sweet Spot
- n-cpu-moe 32 (8/40 layers' experts on GPU): **31.5 TPS** best
- n-cpu-moe 34 (6/40 on GPU): **29.2 TPS** — more CPU, slower
- n-cpu-moe 36 (4/40 on GPU): **27.9 TPS** — even slower
- n-cpu-moe 30 (10/40 on GPU): likely OOM/crash

## Key Takeaways
1. **9B at 0.580 / 52.2 TPS is the ceiling for RTX 4060 8 GB.** No tuning fork, feature, or hyperparam improves it.
2. **Stock fork (BoFan MTP+TurboQuant) is faster and more stable than BeeLlama** for this hardware.
3. **DFlash and CopySpec don't help on 8 GB VRAM.** The overhead of speculative decoding exceeds gains when GPU compute is already saturated.
4. **IQ3 quantization is not a free lunch on small GPUs.** Smaller file doesn't always mean faster — CUDA kernel optimization matters more.
5. **The 35B is hardware-limited.** 8 GB VRAM can't feed a 35B MoE well. CPU bottleneck for 32/40 expert layers is unavoidable.
6. **For daily driver, use 9B.** It's the right model for this hardware.


