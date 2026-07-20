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

## Config Baseline (2026-07-20 - Optimized)
- `MODEL = 'gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 65536`
- `KV_CACHE_K = 'q8_0'`
- `KV_CACHE_V = 'q8_0'`
- `NGL = 99`
- `THREADS = 8`
- `THREADS_BATCH = None`
- `BATCH_SIZE = 512`
- `UBATCH_SIZE = 256`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = None`
- `SPEC_DRAFT_N_MAX = 0`
- `CONT_BATCHING = False`

### Status
- **Tuned (2026-07-20):** Optimized via `autoloop.py` under TPS mode with perplexity ceiling. Increased throughput from **72.6 t/s** to **75.7 t/s** (PPL: 16.3360, up slightly from 16.2608 but well within the 1% ceiling).
- **MTP/Speculation Note:** While manual single-prompt `llama-cli` runs with `mtp-gemma-4-E4B-it.gguf` showed speedups, the autoloop discovered that *disabling* speculation (`SPEC_DRAFT_N_MAX = 0`) yields the highest throughput (75.7 t/s) for batch/evaluation configurations, as the draft model overhead exceeds target inference benefits on this small 4.2B model.
