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

## MTP (Multi-Token Prediction)

- **Main GGUF has NO `nextn` tensors.** Do not treat the UD file as “MTP-inside.”
- MTP lives in the **external assistant draft:** `models/draft/mtp-gemma-4-E4B-it.gguf` (`gemma4-assistant.*` metadata).
- Flags: `SPEC_TYPE=draft-mtp`, `SPEC_DRAFT_MODEL=draft/mtp-gemma-4-E4B-it.gguf` (path relative to `models/`), `SPEC_DRAFT_N_MAX=4`.
- Operator guide: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md).

## Config Baseline (2026-07-20 — speed winner)
- `MODEL = 'gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072` (do not change without user permission; cli TPS gate does not pass `-c`)
- `KV_CACHE_K = 'q4_0'`
- `KV_CACHE_V = 'q4_0'`
- `NGL = 99`
- `THREADS = 6`
- `THREADS_BATCH = 8`
- `BATCH_SIZE = 256`
- `UBATCH_SIZE = 128`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_MODEL = 'draft/mtp-gemma-4-E4B-it.gguf'`
- `SPEC_DRAFT_N_MAX = 4`
- `CONT_BATCHING = False`
- `NO_MMAP = True`

### Status
- **Default speed Baseline (2026-07-20):** winner of fair small-model TPS matrix — see [session](../sessions/2026-07-20-small-model-tps-matrix.md).
- **Fair matrix (`llama-cli` `-n 512`, shared knobs):** base **67.6 t/s** → MTP **122.0 t/s** (**+80%**). Fastest absolute TPS among tested small models.
- **Earlier spot checks:** `-n 128` MTP **136.6 t/s** (+95.4% vs 69.9 base); sustained `-n 512` MTP **113.4 t/s** (pre-matrix knobs).
- **Autoloop note:** server-path TPS with PPL ceiling previously peaked **76.67 t/s** at same draft/`q4_0`/n_max=4 — lower than raw cli-bench (different workload). Prefer cli matrix for apples-to-apples MTP compares.
- **KV:** use `q4_0` only on upstream builds. `turbo*` KV types are **not** in upstream `llama.cpp`.
