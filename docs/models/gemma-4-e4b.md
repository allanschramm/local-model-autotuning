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
- `KV_CACHE_K = 'q4_0'`
- `KV_CACHE_V = 'q4_0'`
- `NGL = 99`
- `THREADS = 6`
- `THREADS_BATCH = 8`
- `BATCH_SIZE = 256`
- `UBATCH_SIZE = 128`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_MODEL = 'models/draft/mtp-gemma-4-E4B-it.gguf'`
- `SPEC_DRAFT_N_MAX = 4`
- `CONT_BATCHING = False`
- `NO_MMAP = True`

### Status
- **Tuned (2026-07-20):** Optimized via `autoloop.py` under TPS mode with perplexity ceiling. Over the course of 760 search rounds, increased throughput from **72.6 t/s** to **76.67 t/s** with absolutely **zero perplexity degradation** (PPL remains at baseline 16.2608).
- **MTP/Speculation Note:** Under batch/evaluation configurations, the autoloop discovered that speculative decoding with the draft model can be optimized to beat the no-spec baseline, reaching **76.67 t/s** when KV cache types are set to `q4_0` (Key) / `q4_0` (Value) and speculative draft max tokens is set to `4` (with `threads=6`, `threads_batch=8`).
- **Direct Verification (llama-cli):** Single-prompt generation speeds verified directly using `llama-cli`:
  - **Shorter Generation (-n 128 - Prime Numbers):**
    - **Without MTP:** 69.9 t/s
    - **With MTP (`--spec-draft-n-max 4`):** **136.6 t/s** (+95.4% speedup).
  - **Shorter Generation (-n 128 - Quantum Mechanics):**
    - **With MTP:** 128.8 t/s.
  - **Sustained Generation (-n 512 - Quantum Mechanics Tutorial):**
    - **With MTP:** **113.4 t/s**.
