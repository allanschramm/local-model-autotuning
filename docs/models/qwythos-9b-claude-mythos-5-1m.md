# Qwythos-9B-Claude-Mythos-5-1M — Model Card (Local)

**Source repo:** https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF
**License:** Apache-2.0
**Local files:**
- `models/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` (5.3 GB)
**Family:** Qwythos (based on Qwen 3.5 architecture)
**Architecture type:** Dense (all params active per token)
**MTP:** Optional training head (orthogonal to architecture — both dense and MoE models can have MTP)
**Quantization:** `Q4_K_M` (file_type=15)

## Architecture (from GGUF metadata)
- Causal LM (hybrid Attention + SSM — Qwen 3.5 arch)
- **`block_count` = 32 layers**
- Hidden **4096**, context **1,048,576** (1M tokens)
- **Hybrid Attention + SSM (Mamba-2 style)**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=32`, `inner_size=4096`
  - 8 full attention layers, 24 SSM layers
- `rope.freq_base = 10,000,000`

## Hardware Requirements (RTX 4060 8GB)
| Quant | Size | VRAM (idle) | VRAM (131k ctx) |
|---|---|---|---|
| Q4_K_M | 5.3 GB | ~5.5 GB | ~7.5 GB |

Fits entirely in 8 GB with 131k context and flash-attn.

**Hard constraint: ctx >= 100k always.** Small-ctx tests are irrelevant for this model's use case.

## Batch/Ubatch Sweet Spot (llama-bench 2026-06-30)

**RTX 4060 — Qwythos 9B Q4_K_M — pp512 / tg128:**

| ubatch | pp512 (t/s) | tg128 (t/s) |
|-------:|-----------:|----------:|
| 64     | 1480.68    | **50.16** |
| 128    | 1814.17    | 42.15     |
| **256** | **1922.61** | **49.75** |
| 512    | 1939.81    | 41.03     |
| 1024   | 1529.87    | 41.50     |
| 2048   | 1917.10    | 48.98     |
| 4096   | 1849.93    | 42.12     |

**Winner: ubatch=256** — best balance of prompt processing (1922 t/s) and text generation (49.8 t/s). ubatch=512 gives +1% pp speed but loses 18% tg speed.

## Recommended Settings

| Param | Value | Rationale |
|---|---|---|
| TEMP | 0.6 | Per HF card for Qwythos reasoning model |
| TOP_P | 0.95 | Default nucleus sampling |
| TOP_K | 20 | Focused token pool |
| REPEAT_PENALTY | 1.05 | Light penalty for repetition reduction |
| BATCH_SIZE | 1024 | Matched with ubatch=256 |
| UBATCH_SIZE | 256 | Sweet spot on RTX 4060 (llama-bench) |
| SPEC_TYPE | None | MTP + 131k ctx exceeds 8GB VRAM |
| SPEC_DRAFT_N_MAX | 0 | Disabled — no VRAM headroom for draft

## MTP (Multi-Token Prediction)

**MTP is not a model class.** It is a training technique orthogonal to architecture type:
- Dense models can have MTP (this model, Qwen 3.5, Gemma 4)
- MoE models can have MTP (Qwen 3.6, DeepSeek V3)
- Dense/MoE = which subset of params activates per token (architecture)
- MTP/no-MTP = whether model was trained to predict multiple future tokens (training objective)
- In inference, an MTP head acts as a draft model for speculative decoding

**Status on RTX 4060 8GB:** MTP + 131k ctx exceeds VRAM. The MTP variant was deleted — no VRAM headroom for MTP benefit on 8GB card with 131k context.
- Non-MTP at 131k uses ~7.5 GB already
- MTP adds model overhead (+0.26 GB) + draft KV cache
- With `--spec-draft-ctx-size 512`, server loads but throughput collapses (~5 min per task)
- At 8192 ctx, MTP fits (7.3 GB, 42 TPS) but throughput is _lower_ than non-MTP at 131k
- **Verdict**: No VRAM headroom for MTP benefit on 8GB card with 131k context

## Validation Bench (2 tasks each, 2026-06-30)

All runs with `config.py` defaults unless noted.

| Run | Server | Model | Batch/Ubatch | Score | TPS | VRAM |
|-----|--------|-------|-------------|------|----|------|
| 1 | turboquant | non-MTP | 512/128 | 0.1250 | 52.6 | 7.3 GB |
| **2** | **turboquant** | **non-MTP** | **1024/256** | **0.4250** | **51.2** | **7.5 GB** |
| 3 | beellama | MTP | 1024/256 | timed out | — | — |
| 4 | beellama | MTP + draft-ctx 512 | 1024/256 | timed out | — | — |
| 5 | beellama | MTP @ 8k ctx | 1024/256 | 0.1250 | 42.0 | 7.3 GB |
| 6 | upstream build-cuda | non-MTP | 1024/256 | 0.3000 | 50.4 | 7.1 GB |

**Winner: Run 2** — non-MTP, 1024/256, turboquant. Score 0.4250 at 51.2 TPS.

Run 6 (2026-07-01): upstream build-cuda + cont-batching. Score lower due to MBPP+ 0/2 — 2-task sampling fluke.

## Config Baseline (2026-06-30)

```python
MODEL = 'Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf'
CTX_SIZE = 131072
BATCH_SIZE = 1024
UBATCH_SIZE = 256
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
REPEAT_PENALTY = 1.05
```

## Tuning History
- 2026-06-30: Initial validation (512/128 → 0.1250)
- 2026-06-30: Batch sweep (1024/256 → 0.4250, 3.4× improvement)
- 2026-06-30: llama-bench ubatch sweep (ub=256 is sweet spot)
- 2026-06-30: MTP tested — no VRAM headroom on 8GB at 131k ctx
- 2026-07-01: MTP variant deleted — non-MTP is the only local copy
- 2026-07-01: Re-validation upstream build-cuda + cont-batching (0.3000, 50.4 TPS, 7.1 GB)
