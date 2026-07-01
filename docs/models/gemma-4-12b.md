# Gemma-4-12B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
**MTP guide:** https://unsloth.ai/docs/models/mtp
**License:** Apache-2.0 (Gemma 4 license)
**Local file:** `models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (6.3 GB)
**Family:** Gemma 4 (Google DeepMind)
**Quantization:** Unsloth Dynamic 2.0 — `UD-Q4_K_XL` (QAT-lossless)

## Architecture
- Dense 12B model (not MoE)
- 42 layers, GQA with KV cache sharing across layers
- Architecture key: `gemma4`
- Context length: 1M+ tokens (YaRN)

## MTP (Multi-Token Prediction)
Uses a **separate draft head** (not embedded in main GGUF like Qwythos).
- `MTP/gemma-4-12B-it-Q4_0-MTP.gguf` (242 MB, 4-layer `gemma4-assistant` model)
- Requires `--spec-draft-model mtp-gemma-4-12B-it.gguf` flag
- Draft can run on CPU with `--spec-draft-ngl 0` to save VRAM
- Recommended `--spec-draft-n-max 2` per Unsloth docs
- **Status**: Draft head failed to load on upstream llama.cpp (arch name mismatch). Not yet validated.

## Hardware Requirements (RTX 4060 8GB)
| Quant | Size | VRAM (131k ctx) |
|---|---|---|
| UD-Q4_K_XL (our pick) | 6.3 GB | 8.0 GB (maxed) |

**VRAM ceiling reached** at 131k ctx. Fits but pegs 8 GB completely.

## Recommended Settings

| Param | Value | Rationale |
|---|---|---|
| TEMP | 0.4 | Optimal for coding benchmark suite |
| TOP_P | 0.95 | Default nucleus sampling |
| TOP_K | 20 | Focused token pool |
| REPEAT_PENALTY | 1.05 | Light repetition penalty |
| BATCH_SIZE | 1024 | From llama-bench sweep |
| UBATCH_SIZE | 256 | Sweet spot on RTX 4060 |
| SPEC_TYPE | None | MTP draft not yet validated |
| N_CPU_MOE | N/A | Dense model, no MoE |

## Validation Bench (2 tasks each, 2026-06-30)

| Bench | Score | TPS |
|---|---|---|
| HumanEval+ | **1.0000** | 34.0 |
| MBPP+ | 0.5000 | 34.1 |
| LCB | 0.5000 | 37.4 |
| BigCode Hard | 0.0000 | 32.8 |
| **Overall** | **0.5500** | **34.6** |
| **VRAM** | **8.0 GB** | |

## vs Qwythos 9B (best run)

| Metric | Gemma 4 12B | Qwythos 9B | Delta |
|---|---|---|---|
| Score | **0.5500** | 0.4250 | **+0.1250** |
| TPS | 34.6 | 51.2 | -32% |
| VRAM | 8.0 GB | 7.5 GB | +0.5 GB |

Gemma 4 scores higher (+29%) but runs slower (-32%). VRAM fully saturated at 131k ctx.

## Config Baseline (2026-06-30)

```python
MODEL = 'gemma-4-12B-it-qat-UD-Q4_K_XL.gguf'
CTX_SIZE = 131072
BATCH_SIZE = 1024
UBATCH_SIZE = 256
SPEC_TYPE = None
SPEC_DRAFT_N_MAX = 0
TEMP = 0.4
TOP_P = 0.95
TOP_K = 20
REPEAT_PENALTY = 1.05
```

## Tuning History
- 2026-06-30: Initial validation (base), score 0.5500, VRAM 8.0 GB

---

# Gemma4-12B v2 (agentic-fable5) — Variant

**Source repo:** https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF
**Local file:** `models/gemma4-v2-Q3_K_M.gguf` (5.8 GB)
**Base model:** `deepreinforce-ai/gemma-4-12B-it` finetuned for agentic coding

An agentic coding fine-tune on Gemma 4 base. ~3.5× improvement on tau2-bench telecom over base (15% → 55%). Purchased at 3-bit.

## Architecture
- Same `gemma4` arch as base (dense 12B, 42 layers)
- Post-trained on top of Gemma 4 for coding & tool-use
- Native context: 262k tokens
- Chat template: Gemma 4 (requires `--jinja`)

## Validation (2-task, 2026-07-01, 131k ctx)

| Bench | Score | TPS |
|---|---|---|
| HumanEval+ | **1.0000** | 44.8 |
| MBPP+ | **0.5000** | 37.9 |
| LCB | **0.5000** | 36.4 |
| BigCode Hard | 0.0000 | 40.0 |
| **Overall** | **0.5500** | **38.4** |
| **VRAM** | **7.9 GB** (131k ctx) |

## vs Gemma 4 base (UD-Q4_K_XL)

| Metric | v2 Q3_K_M | Base UD-Q4_K_XL | Delta |
|---|---|---|---|
| Score | **0.5500** | 0.5500 | **0%** |
| TPS | 38.4 | 34.6 | **+11%** |
| VRAM | 7.9 GB | 8.0 GB | -0.1 GB |

Same coding score as base at Q4_K_XL, measured at 131k ctx. v2's agentic fine-tune compensates for Q3 quantization loss. BigCode Hard still zero — library-call tasks suffer at 3-bit.

## Limitations
- **BigCode zero** = library-call tasks hit quality cliff at 3-bit
- Q3 decode kernel ~25% slower than Q4 on RTX 4060
- VRAM at 7.9 GB leaves only 0.1 GB headroom — any overhead risks OOM

## Tuning History
- 2026-07-01: Downloaded Q3_K_M, validated at 131k (0.5500, 38.4 TPS, 7.9 GB) — no OOM
