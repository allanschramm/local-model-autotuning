# Qwen3.5-9B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.5
**License:** Apache-2.0
**Local file:** `models/Qwen3.5-9B-UD-Q4_K_XL.gguf` (6.14 GB) (previously `models/Qwen3.5-9B-Q4_K_M.gguf`)
**Family:** Qwen 3.5 (Alibaba)
**Quantization:** Unsloth Dynamic Q4_K_XL with MTP — `UD-Q4_K_XL` (QAT-lossless 4-bit)

## Architecture (from GGUF metadata)
- Causal LM (hybrid Attention + SSM — Qwen 3.5 architecture)
- **`block_count` = 32 layers**
- Hidden **4096**, context **131072**
- **Hybrid Attention + SSM (Mamba-2 style)**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=32`, `inner_size=4096`
  - 8 full attention layers, 24 SSM layers

## Hardware Requirements (RTX 4060 8GB)
- Fits entirely in GPU VRAM (NGL = 99).
- Model size ~6.14 GB, leaving adequate headroom for context cache.

## MTP (Multi-Token Prediction)
- **MTP tensors are integrated into this GGUF.**
- Configured with `spec_type = "draft-mtp"`.
- Uses `--spec-draft-n-max 2` for speculative decoding, achieving up to 1.5x–2.0x faster generation speeds.

## Recommended Settings
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.05
- **Chat Template:** Jinja (requires `--jinja` flag)

## Config Baseline (2026-07-19)
- `MODEL = 'Qwen3.5-9B-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_N_MAX = 2`

### Status
- **UNTESTED.** Upgraded to Unsloth Dynamic Q4_K_XL with built-in MTP.
