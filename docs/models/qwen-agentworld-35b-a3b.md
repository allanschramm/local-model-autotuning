# Qwen-AgentWorld-35B-A3B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.6 (same architecture family)
**License:** Apache-2.0
**Local file:** `models/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf` (16.56 GB)
**Family:** Qwen-AgentWorld (Qwen 3.5 MoE architecture)
**Quantization:** Unsloth Dynamic 2.0 — `UD-IQ4_XS` (importance-quantized 4-bit extra-small)

## Architecture (from GGUF metadata, verified via gguf.GGUFReader 2026-07-02)
- Causal LM (hybrid Attention + SSM + MoE)
- **`block_count` = 40 layers**
- **35B total / 3B activated** (MoE, `expert_count=256, expert_used_count=8` + 1 shared expert)
- Hidden **2048**, vocab 248320, ctx **262144**
- **Hybrid Attention + SSM (Mamba-2 style) + MoE layers**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - Contains `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`, `ssm.inner_size=4096`
  - 10 layers of full attention (head count: 16 Q, 2 KV, key/value length 256)
  - 30 layers of SSM / linear path
- `rope.freq_base = 10,000,000`, `rope.dimension_count = 64`
- Expert FFN: `expert_feed_forward_length=512`, `expert_shared_feed_forward_length=512`
- **`general.name` = `Qwen-Agentworld-35B-A3B`**, file_type=30 (IQ4_XS), quantization_version=2
- 733 tensors total
- Imatrix calibrated: `unsloth_calibration_Qwen-AgentWorld-35B-A3B.txt`

## Hardware requirements
| Quant | Total RAM (RAM + VRAM) |
|---|---|
| **IQ4_XS (our pick)** | **~17 GB** |
| Q4_K_M (reference) | ~23 GB |

**Our target:** 8 GB VRAM (RTX 4060) + 16-24 GB RAM. IQ4_XS saves ~6 GB vs Q4_K_M, making VITRIOL split more feasible on 16 GB RAM.

**Note:** IQ4_XS is importance-quantized — uses imatrix calibration data for better quality at lower size. Expect quality close to Q4_K_M with ~25% less memory.

## Recommended Settings (based on Qwen 3.5 MoE family)
- **Temperature:** 0.6
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)

## MTP (Multi-Token Prediction)
- **NO MTP tensors in this GGUF.** No spec_type configured.

## VITRIOL / Split strategy (MoE expert offloading)
Same as Qwen3.6-35B-A3B: attention + shared expert + routing on GPU, 256 routed experts in CPU/RAM.
- `--n-gpu-layers 99` — load active paths into VRAM.
- `--n-cpu-moe 40` — all 40 layers' MoE experts on CPU.

Source: https://www.youtube.com/watch?v=ZwNCsUTNWOA (Codacus technique).

## Our config baseline (TBD — not yet run)
- `MODEL = 'Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `N_CPU_MOE = 40`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`
- `TEMP = 0.6`
- `TOP_P = 0.95`
- `TOP_K = 20`

## Sources / Verification
- HuggingFace: `unsloth/Qwen-AgentWorld-35B-A3B-GGUF`
- GGUF metadata verified via `gguf.GGUFReader` on 2026-07-02
- Architecture identical to Qwen3.6-35B-A3B (same qwen35moe arch, same layer/expert counts)

## Open questions
- **TBD (2026-07-02):** First validation run needed — baseline score and TPS on RTX 4060.
- **TBD:** Compare IQ4_XS vs Q4_K_M quality at same settings (imatrix calibration may help or hurt).
- **TBD:** Relationship to Qwen3.6-35B-A3B — same base model? Different fine-tune? HF card claims "AgentWorld" variant.
