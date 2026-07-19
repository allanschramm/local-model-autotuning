# Gemma-4-E4B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
**License:** Apache-2.0 (Gemma 4 license)
**Local file:** `models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (4.22 GB)
**Family:** Gemma 4 (Google DeepMind)
**Quantization:** Unsloth Dynamic QAT — `UD-Q4_K_XL` (QAT-lossless 4-bit)

## Hardware requirements (RTX 4060 8GB)
- Fits entirely in GPU VRAM (NGL = 99).
- Model size ~4.22 GB, leaving plenty of headroom for active KV cache (even up to 131k ctx).

## Recommended Settings (Gemma 4)
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)
- **Chat Template:** Gemma 4 (requires `--jinja` flag)

## Config Baseline (2026-07-19)
- `MODEL = 'gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072` (Maximum context)
- `KV_CACHE = 'q4_0'`
- `NGL = 99` (GPU Offloaded)
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`

### Status
- **UNTESTED.** Upgraded to Unsloth QAT Dynamic Q4_K_XL.
