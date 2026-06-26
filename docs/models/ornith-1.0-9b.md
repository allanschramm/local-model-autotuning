# Ornith-1.0-9B — Model Card (Local)

**Source repo:** https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen35 (model uses Qwen 3.5 architecture)
**License:** Apache-2.0
**Local file:** `models/ornith-1.0-9b-Q4_K_M.gguf` (5.63 GB)
**Family:** Ornith (based on Qwen 3.5 architecture)
**Quantization:** `Q4_K_M`

## Architecture (from GGUF metadata, verified via gguf lib)
- Causal LM (hybrid Attention + SSM)
- **`block_count` = 32 layers**
- Hidden **4096**, vocab 248320, ctx **262144**
- **Hybrid Attention + SSM (Mamba-2 style) layers**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - Contains `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`, `ssm.inner_size=4096`
  - 8 layers of full attention (head count: 16 Q, 4 KV, key/value length 256)
  - 24 layers of SSM / linear path
- `rope.freq_base = 10,000,000`
- **`general.name` = `Ornith 1.0 9B`**, file_type=15 (Q4_K_M), quantization_version=2
- 427 tensors total

## Hardware requirements (per community and size)
| Quant | Total RAM / VRAM |
|---|---|
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** (VRAM target is ~5.6 GB + KV cache overhead) |
| Q8_0 | ~9.5 GB VRAM |

**Our target:** 8 GB VRAM (RTX 4060). The 4-bit model size is ~5.6 GB, meaning it fits entirely in GPU VRAM (NGL = 999). However, active KV cache overhead for large contexts can push VRAM usage above 8 GB. Setting safe context size limits is important.

## Recommended Settings (based on Qwen 3.5)
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)

## MTP (Multi-Token Prediction)
- **NO MTP tensors in this GGUF.** No spec_type configured.

## VITRIOL / Split strategy
Since the model is ~5.6 GB and we have 8 GB of VRAM, we can run with maximum GPU offload (`--n-gpu-layers 999`), loading the model completely into GPU VRAM.

## Our config baseline (TBD)
- `MODEL = 'ornith-1.0-9b-Q4_K_M.gguf'`
- `CTX_SIZE = 32768` (safe initial limit for 8GB VRAM)
- `KV_CACHE = 'q4_0'`
- `NGL = 999`
- `THREADS = 8`

## Sources / Verification
- HuggingFace Model Card (`deepreinforce-ai/Ornith-1.0-9B-GGUF`)
- Checked with local scratch tool parsing `models/ornith-1.0-9b-Q4_K_M.gguf` metadata via `GGUFReader` on 2026-06-26.

## Open questions
- None (baseline specs verified).
