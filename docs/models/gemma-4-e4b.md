# Gemma-4-E4B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/gemma-4-e4b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
**HF model:** https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF
**License:** Apache-2.0 (Gemma 4 license)
**Local file:** `models/lmstudio-community/gemma-4-e4b-it-gguf/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (also under `models/unsloth/gemma-4-E4B-it-qat-GGUF/`)
**Family:** Gemma 4 (Google DeepMind)
**Quantization:** Unsloth Dynamic QAT — `UD-Q4_K_XL` (QAT-lossless 4-bit)

## 2. Architecture (GGUF-verified values)
- **Context length:** 131072 tokens (matches publisher config; `--ctx` 131k accepted)
- **Architecture:** gemma4
- **Quantization:** UD-Q4_K_XL — QAT-lossless 4-bit; K q4_0, V q4_0
- **Tokenizer chat template:** `enable_thinking`-conditional Jinja (default: thinking off)
- **Reasoning control:** template reads `enable_thinking` only → `--reasoning on/off` maps to it; `--reasoning-effort` is a silent NO-OP; `--reasoning-budget N` forces end-of-thinking tag (server-side, template-independent)
- **Embedding count:** 4.5B effective (PLE); 8B with embeddings — `E4B` = "effective parameters" via Per-Layer Embeddings

## 3. Hardware requirements (discrete 8 GB-class NVIDIA)
- Fits entirely in GPU VRAM (NGL = 99).
- Model size ~4.22 GB, leaving plenty of headroom for active KV cache (even up to 131k ctx).
- **Publisher table:** E4B effective-params story — 4.5B effective (PLE) / 8B with embeddings; ctx 128K tokens; MMLU Pro 69.4%; AIME 2026 no tools 42.5%.

## 4. Recommended Settings (Gemma 4)
- **Temperature:** 0.4 (our speed winner) / publisher std: **1.0**
- **Top P:** 0.95 (shared across publisher std)
- **Top K:** 20 (our config) / publisher std: **64**
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)
- **Chat Template:** Gemma 4 (requires `--jinja` flag)
- **Reasoning control:** `enable_thinking` template var → enable via `--reasoning on` or system-prompt `<|think|>` token; `--reasoning-effort` is a silent no-op on this template; `--reasoning-budget N` is server-side and effective regardless

## 5. MTP (Multi-Token Prediction)
- **Main GGUF has NO `nextn` tensors.** Do not treat the UD file as "MTP-inside."
- MTP lives in the **external assistant draft:** `models/draft/mtp-gemma-4-E4B-it.gguf` (`gemma4-assistant.*` metadata).
- Flags: `SPEC_TYPE=draft-mtp`, `SPEC_DRAFT_MODEL=draft/mtp-gemma-4-E4B-it.gguf` (path relative to `models/`), `SPEC_DRAFT_N_MAX=4`.
- Operator guide: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md).

## MoE split (VITRIOL)

N/A — dense MatFormer E4B (no expert tensors); `--n-cpu-moe` not applicable.

## 6. Config Baseline (2026-07-20 — speed winner)
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
- **Default speed Baseline (2026-07-20):** winner of fair small-model TPS matrix — see [session](../performance/2026-07-20-small-model-tps-matrix.md).
- **Fair matrix (`llama-cli` `-n 512`, shared knobs):** base **67.6 t/s** → MTP **122.0 t/s** (**+80%**).
- **Earlier spot checks:** `-n 128` MTP **136.6 t/s** (+95.4% vs 69.9 base); sustained `-n 512` MTP **113.4 t/s** (pre-matrix knobs).
- **Autoloop note:** server-path TPS with PPL ceiling previously peaked **76.67 t/s** at same draft/`q4_0`/n_max=4 — lower than raw cli-bench (different workload). Prefer cli matrix for apples-to-apples MTP compares.
- **KV:** use `q4_0` only on upstream builds. `turbo*` KV types are **not** in upstream `llama.cpp`.

## Historical runtime evidence (7. Measured (claw-full))

Alias path: ctx **65k**, draft-mtp n=4, KV q4_0, cont-batching on.

| Stage | note |
| --- | --- |
| 2026-07-24 | Pre harness fix; bench_tg 113.3; peak 5.8 GB |
| 2026-08-08 remasure | Post `reasoning_content` / max_tokens fix; TPS 126.7; peak **5.1 GB** |

Evidence: `results.tsv` remasure row; [thinking-models-claw-harness.md](../performance/thinking-models-claw-harness.md).

## 8. Sources/Verification (dated 2026-08-29)
- HF model card: https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF (JSON API `api/models/unsloth/gemma-4-E4B-it-qat-GGUF`, raw README)
- Publisher benchmarks from README table: MMLU Pro 69.4%, AIME 2026 no tools 42.5%, LiveCodeBench v6 52.0%, Codeforces ELO 940, GPQA Diamond 58.6%, Tau2 42.2%, MMMLU 76.6%
- Context: 128K tokens (publisher) / 131072 tokens (GGUF store, verified via `api/models/` `context_length` field)
- License: Apache-2.0 (consistent across HF API cardData and publisher)
- Quantization: UD-Q4_K_XL — Unsloth Dynamic QAT, QAT-lossless 4-bit
- Reasoning template: verified `enable_thinking` via chat_template Jinja extraction; `--reasoning-effort` no-op; `--reasoning on/off` mapping confirmed
- GGUF store metadata: ctx_length=131072, architecture=gemma4, totalFileSize ~7.46 GB

## 9. Open questions
- **Row volume explained (2026-08-29):** 3404 of the rows are genuine `engine-tps` candidates written by the AutoLoop R1 hill-climb on **2026-07-19/20** (3334 in a single `7b41639-dirty` commit state; 2702 distinct configs; measurements vary per candidate — real search, not repeated writes). These are the only `engine-tps` rows in the store because this GGUF was the sole autoloop target; `status=incomplete` is the store-wide norm for engine-tps rows (recompute grades capability rows only). Upper-case `REASONING_*` keys = legacy writer era. Trial_ids all distinct (3416/3416); no store corruption.
- Gemma 4 MoE variant (26B A4B) not in scope for this 8 GB card; future work if needed.
- `--reasoning-budget` interaction with `enable_thinking`-only templates not yet benchmarked at scale.
