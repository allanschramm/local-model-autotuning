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
- **MTP tensors are embedded in this GGUF** (`qwen35.nextn_predict_layers`, `blk.32.nextn.*` verified 2026-07-20).
- Enable: `SPEC_TYPE = "draft-mtp"`, `SPEC_DRAFT_N_MAX = 4` — **no** `--spec-draft-model`.
- Fair matrix (2026-07-20, `llama-cli` `-n 512`): base **38.7 t/s** → MTP **57.3 t/s** (**+48%**). Evidence: [session](../sessions/2026-07-20-small-model-tps-matrix.md).

## Recommended Settings
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.05
- **Chat Template:** Jinja (requires `--jinja` flag)

## Config Baseline (2026-07-20 TPS matrix knobs)
- `MODEL = 'Qwen3.5-9B-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `KV_CACHE_K = 'q4_0'`
- `KV_CACHE_V = 'q4_0'`
- `NGL = 99`
- `THREADS = 6`
- `THREADS_BATCH = 8`
- `BATCH_SIZE = 256`
- `UBATCH_SIZE = 128`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_N_MAX = 4`
- `NO_MMAP = True`

### Status
- **Measured (2026-07-20):** embedded MTP works on upstream CUDA; +48% vs base under fair knobs. Slower absolute than Gemma+draft MTP (122 t/s).
