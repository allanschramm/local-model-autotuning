# Ornith-1.0-35B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/Ornith-1.0-35B-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen35 (model uses Qwen 3.5 MoE architecture)
**License:** Apache-2.0
**Local file:** `models/Ornith-1.0-35B-UD-Q4_K_XL.gguf` (22.32 GB) (previously `models/deepreinforce-ai_Ornith-1.0-35B-IQ3_M.gguf`)
**Family:** Ornith (based on Qwen 3.5 MoE architecture)
**Quantization:** `UD-Q4_K_XL` (Unsloth Dynamic Q4_K_XL)

## Architecture (from GGUF metadata, verified via llama-server log)
- Causal LM (hybrid Attention + SSM + MoE)
- **`block_count` = 40 layers**
- **35B total / 3B activated** (MoE, `expert_count=256, expert_used_count=8` + 1 shared expert)
- Hidden **2048**, vocab 248320, ctx **262144**
- **Hybrid Attention + SSM (Mamba-2 style) + MoE layers**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - Contains `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`, `ssm.inner_size=4096`
  - 10 layers of full attention (head count: 16 Q, 2 KV, key/value length 256)
  - 30 layers of SSM / linear path
- `rope.freq_base = 10,000,000`
- **`general.name` = `Ornith 1.0 35B`**, file_type=15 (Q4_K_M), quantization_version=2
- 733 tensors total

## Hardware requirements (per community and size)
- Quantization file size: **19.7 GiB**
- Total target memory (RAM + VRAM): **~24 GB**

**Our target:** 8 GB VRAM (RTX 4060) + 24 GB RAM. We are at the 4-bit limit. Offloading routed experts to CPU/RAM is mandatory to fit in VRAM.

## VITRIOL / Split strategy (MoE expert offloading)
With MoE, we place **attention + shared expert + routing** on the GPU, and keep the **256 routed experts in CPU/RAM**.
- `--n-gpu-layers 99` — load active paths into VRAM.
- `--n-cpu-moe 32` — offloads MoE experts of the first 32 layers to the CPU, keeping the last 8 layers of experts on the GPU.

## Recommended Settings (based on Qwen 3.5)
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)

## MTP (Multi-Token Prediction)
- **NO MTP tensors in this GGUF.** No spec_type configured.

## Our config baseline (Verified 2026-06-27)
- `MODEL = 'ornith-1.0-35b-Q4_K_M.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `N_CPU_MOE = 32`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`

### Benchmark Scores (10 tasks baseline)
- **Coding Score:** `0.5550`
  - **LiveCodeBench:** `0.4000`
  - **HumanEval+:** `0.8000`
  - **MBPP+:** `0.8000`
  - **BigCodeBench Hard:** `0.1000`
- **Peak VRAM:** `7.7 GB`
- **TPS:** `31.5`

## Sources / Verification
- HuggingFace Model Card (`deepreinforce-ai/Ornith-1.0-35B-GGUF`)
- Verification validation run completed successfully on 2026-06-27.

## Variant: IQ3_M (2026-07-02 GGUF verified)

**Local file:** `models/deepreinforce-ai_Ornith-1.0-35B-IQ3_M.gguf` (15.74 GB)
**Quantization:** `IQ3_M` (file_type=27, importance-quantized 3-bit)

GGUF metadata (verified via `gguf.GGUFReader` 2026-07-02):
- `general.name` = `Ornith 1.0 35B` — same model, different quant
- `general.architecture` = `qwen35moe` — same 40-layer hybrid arch
- `general.license` = `mit` (vs Apache-2.0 for Q4_K_M — upstream license differs by release)
- `general.sampling.temp` = 1.0, `top_k` = 20, `top_p` = 0.95
- Same expert config: `expert_count=256`, `expert_used_count=8`, `full_attention_interval=4`
- 733 tensors total, no MTP tensors

See [IQ3_M variant card](ornith-1.0-35b-iq3_m.md) for detailed testing results.

## Tuning History (2026-06-29)

### BeeLlama tested (no gains)
- BeeLlama baseline: 27 TPS (stock fork is faster)
- BeeLlama + CopySpec: 22 TPS (worse)
- BeeLlama + DFlash (GPU): 3.3 TPS (draft competes for VRAM)
- BeeLlama + DFlash (CPU): OOM (cross-attention overflows 16 GB RAM)

### IQ3_M tested (slower)
- File: 15.7 GB vs Q4_K_M 19.7 GB
- Slower at every n-cpu-moe setting (kernel less optimized)
- Not recommended

### n-cpu-moe sweep results
| n-cpu-moe | GPU layers | TPS | VRAM | Notes |
|-----------|-----------|-----|------|-------|
| 36 | 4/40 | 27.9 | 6.3 GB | Too much on CPU |
| 34 | 6/40 | 29.2 | 7.1 GB | |
| **32** | **8/40** | **31.5** | **7.7 GB** | **✅ Best** |
| 30 + turbo3 | 10/40 | 19.6 | — | Turbo3 kills TPS |

### Verdict
- **n-cpu-moe 32 is the sweet spot** for RTX 4060 8 GB
- No fork, quant, or speculative decoding improves TPS or score
- 35B is hardware-limited by 8 GB VRAM

### 2026-07-19 Update (Unsloth Dynamic 4-bit XL Quant)
- Upgraded local model to the newly released `Ornith-1.0-35B-UD-Q4_K_XL.gguf` (22.32 GB) from Unsloth.
- Alias: `ornith-35b` (INDEX name; old `o35` retired 2026-07-23). Prefer UD-Q4_K_XL — do not keep a parallel Q3_K_XL alias.
- Performance/TPS results pending benchmark.

## Open questions
- None (baseline verified).
