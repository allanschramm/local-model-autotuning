# Qwen-AgentWorld-35B-A3B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/qwen-agentworld-35b-a3b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.6 (same architecture family)
**License:** Apache-2.0
**Local file:** `models/Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf` (16.56 GB)
**Family:** Qwen-AgentWorld — native language world model (CPT → SFT → RL/GSPO); base `Qwen/Qwen3.5-35B-A3B-Base`, **not** a Qwen3.6 fine-tune (official card, 2026-08-02; arXiv 2606.24597)
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
| --- | --- |
| **IQ4_XS (our pick)** | **~17 GB** |
| Q4_K_M (reference) | ~23 GB |

**Our target:** 8 GB VRAM (8 GB-class discrete NVIDIA) + 16-24 GB RAM. IQ4_XS saves ~6 GB vs Q4_K_M, making VITRIOL split more feasible on 16 GB RAM.

**Note:** IQ4_XS is importance-quantized — uses imatrix calibration data for better quality at lower size. Expect quality close to Q4_K_M with ~25% less memory.

## Recommended Settings (official Best Practices, verified 2026-08-02)
- **Temperature:** 0.6
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)
- Thinking mode on by default; recommended output length 32,768 tokens.
- Source: https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B (2026-08-02). Qwen3.6 sampler profiles do not apply (different family, no MTP). Reasoning control: qwen35-family uses --reasoning on/off (Baseline REASONING) and --reasoning-budget N (REASONING_BUDGET, server-side template-independent); REASONING_PRESERVE only where /props reports supports_preserve_reasoning — unverified for this model.

## MTP (Multi-Token Prediction)
- **NO MTP tensors in this GGUF.** No spec_type configured.
- **No MTP claim on the official card** (absence-of-evidence, 2026-08-02): the base is the Qwen3.5-era checkpoint (MTP is a Qwen3.6-family feature) — treat MTP as unsupported until a local header check proves otherwise. Do not assume MTP.

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
- Official Qwen card + README: https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B (2026-08-02) — base model, Best Practices sampling, tech report arXiv 2606.24597
- GGUF metadata verified via `gguf.GGUFReader` on 2026-08-29 (header: qwen35moe, 40 blk, 256 exp, 262144 ctx, 733 tensors; MTP tensors not present — no `nextn_predict_layers` / `blk.40.nextn.*`).
- Architecture skeleton is the Qwen3.5 MoE family (qwen35moe arch, 40 layers, 256 experts); the AgentWorld fine-tune is a separate world-model family — do not equate it with Qwen3.6.
- HF API `unsloth/Qwen-AgentWorld-35B-A3B-GGUF`: extracted 2026-08-29 (tags: qwen, gguf, unsloth, world-model, agent, dataset:Qwen/AgentWorldBench; arch qwen35moe, ctx 262144, 34660610688 bytes BF16 reference; siblings include Qwen-AgentWorld-35B-A3B-UD-IQ4_XS.gguf, UD-IQ2/3/4/5/6/8, MXFP4_MOE, Q2_K_XL, Q3_K_M; no `mtp-*` files; imatrix_unsloth.gguf_file sidecar 192 MB).
- HF API `Qwen/Qwen-AgentWorld-35B-A3B`: extracted 2026-08-29 (tags: qwen3_5_moe, dataset:AgentWorldBench, base_model Qwen3.5-35B-A3B-Base; 34.66 GB BF16; context 262144; chat_template with thinking/reasoning_content + multi_step_tool logic; dataset AgentWorldBench).


- **TBD:** Compare IQ4_XS vs Q4_K_M quality at same settings (imatrix calibration may help or hurt).
- **TBD (2026-08-29):** Verify `/props` `supports_preserve_reasoning` on this GGUF before seeding REASONING_PRESERVE (qwen35 family — Ornith-1.5 confirmed true; unverified for Qwen-AgentWorld).
- **TBD (2026-08-29):** Local GGUF re-verification of full tensor name list (only 733-tensor header summary re-confirmed 2026-08-29 via API; full tensor audit on local file is pending).
- **TBD (2026-08-29):** No local rows match this basename in results.db; no measured TPS or Objective Vector yet.
- **TBD (2026-08-29):** Confirm no duplication with `docs/models/qwen3.6-35b-a3b.md`: Qwen-AgentWorld is separate world-model family (CPT→SFT→RL/GSPO from Qwen3.5-35B-A3B-Base); qwen3.6-35b-a3b is Qwen3.6 base with MTP.
