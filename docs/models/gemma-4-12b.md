# Gemma-4-12B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/gemma-4-12b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/google/gemma-4-12B-it; https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/gemma-4
**MTP guide:** https://unsloth.ai/docs/models/mtp
**License:** Apache-2.0 (Gemma 4 license)
**Local file:** N/A (GGUF purged — never capability-evaluated locally)
**Family:** Gemma 4 (Google DeepMind)
**Quant:** N/A (last quant: UD-Q4_K_XL, purged)

## Architecture
- Dense 12B model (not MoE)
- 48 layers (Google `gemma4_unified`); 42 layers (Unsloth GGUF `gemma4`)
- GQA with KV cache sharing across layers
- Architecture key: `gemma4` / `gemma4_unified`
- Context length: 256K tokens (Google: 256K explicit; Unsloth GGUF: 262144 = 256K)
- **Reasoning control**: Google Gemma 4 documents `enable_thinking` flag in chat template. Per publisher/harness notes: thinking modes are configurable via `enable_thinking=True/False`. Since the GGUF was purged and never dumped, local template verification is not claimed. Do not claim local template verification for this model (GGUF purged — never dumped).

## Hardware Requirements (discrete 8 GB-class NVIDIA)
| Quant | Size | VRAM (131k ctx) |
| --- | --- | --- |
| UD-Q4_K_XL (last local quant, purged) | 6.3 GB | 8.0 GB (maxed) |

**VRAM ceiling reached** at 131k ctx. Fits but pegs 8 GB completely.
**Note**: No local benchmark runs; VRAM numbers from publisher documentation only. No local capability evaluation performed.

## Recommended Settings
| Param | Value | Rationale |
| --- | --- | --- |
| TEMP | 0.4 | Optimal for coding benchmark suite (per Unsloth docs) |
| TOP_P | 0.95 | Default nucleus sampling |
| TOP_K | 20 | Focused token pool |
| REPEAT_PENALTY | 1.05 | Light repetition penalty |
| **enable_thinking** | True/False | Per Google docs: set `enable_thinking=True` to activate reasoning mode; `False` disables. Since GGUF purged, local verification not claimed. |
| BATCH_SIZE | 1024 | From llama-bench sweep |
| UBATCH_SIZE | 256 | Sweet spot on 8 GB-class discrete NVIDIA |
| SPEC_TYPE | None | MTP draft not yet validated locally |
| N_CPU_MOE | N/A | Dense model, no MoE |

## MTP (Multi-Token Prediction)
Uses a **separate draft head** (not embedded in main GGUF like Qwythos).
- `MTP/gemma-4-12B-it-Q4_0-MTP.gguf` (242 MB, 4-layer `gemma4-assistant` model) — from Unsloth repo
- Requires `--spec-draft-model mtp-gemma-4-12B-it.gguf` flag
- Draft can run on CPU with `--spec-draft-ngl 0` to save VRAM
- Recommended `--spec-draft-n-max 2` per Unsloth docs
- **Status**: Draft head failed to load on upstream llama.cpp (arch name mismatch). Not yet validated locally.
- **Note**: MTP tensors exist in Unsloth GGUF release; local validation never performed.

## MoE split (“VITRIOL split”)
- Dense model, no MoE
- `N_CPU_MOE=None`

## Our config baseline (TBD)
*No local validation yet — baseline to be established after GGUF recovery or new run.*

## Sources / Verification
- Google Gemma-4-12B-it: https://huggingface.co/api/models/google/gemma-4-12b-it (extracted 2026-08-29)
- Unsloth gemma-4-12B-it-qat-GGUF: https://huggingface.co/api/models/unsloth/gemma-4-12B-it-qat-GGUF (extracted 2026-08-29)
- Gemma 4 license: https://ai.google.dev/gemma/docs/gemma_4_license
- Gemma 4 architectural details: https://arxiv.org/abs/2607.02770
- Unsloth Gemma 4 docs: https://unsloth.ai/docs/models/gemma-4
- **Note**: No local GGUF file exists; all publisher-declared values. No capability evaluation performed locally.

## Open questions
- [2026-08-29] Can local GGUF be recovered for capability evaluation?
- [2026-08-29] What are the exact reasoning/think template parameters for Gemma 4 GGUF?
- [2026-08-29] MTP validation status on local hardware?
- [2026-08-29] What quant was last locally tested and why was it purged?
- [2026-08-29] Verify `enable_thinking` flag behavior on actual Gemma 4 GGUF (template not available locally)

---
Gemma4-12B v2 (agentic-fable5) — Variant

**Source repo:** https://huggingface.co/yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF
**Local file:** `models/gemma-4-12B-fable5-Q3_K_M.gguf` (5.8 GB)
**Base model:** `deepreinforce-ai/gemma-4-12B-it` finetuned for agentic coding

An agentic coding fine-tune on Gemma 4 base. ~3.5× improvement on tau2-bench telecom over base (15% → 55%). Purchased at 3-bit.

## Architecture
- Same `gemma4` arch as base (dense 12B, 42 layers)
- Post-trained on top of Gemma 4 for coding & tool-use
- Native context: 262k tokens
- Chat template: Gemma 4 (requires `--jinja`)

## Validation (2-task, 2026-07-01, 131k ctx)

Config: `b=1024`, `ub=256`, `t=8`, `fa=on`, `cont-batching`, `ngl=99`

| Measurement | Recorded runtime |
| --- | --- |
| HumanEval+ throughput | 45.6 t/s |
| LCB throughput | 53.9 t/s |
| BigCode Hard throughput | 52.5 t/s |
| **Overall** throughput | **43.0** t/s |
| **VRAM** | **7.4 GB** (131k ctx) |

## vs Gemma 4 base (UD-Q4_K_XL)

| Metric | v2 Q3_K_M | Base UD-Q4_K_XL | Delta |
| --- | --- | --- | --- |
| TPS | **43.0** | 36.0 | **+19%** |
| VRAM | **7.4 GB** | 8.0 GB | -0.6 GB |

Agentic fine-tune compensates for Q3 loss. Also uses 0.6 GB less VRAM.

## Limitations
- **BigCode zero** = library-call tasks hit quality cliff at 3-bit
- Q3 decode kernel ~30% slower than Q4 on 8 GB-class discrete NVIDIA (bench tg 31.2 vs 33.4)

## Tuning History
- 2026-07-01: Renamed to `gemma-4-12B-fable5-Q3_K_M.gguf`, validated at 131k with b1024/ub256 (0.5500, 43.0 TPS, 7.4 GB)
- 2026-07-01: Initial validation as `gemma4-v2-Q3_K_M.gguf` (0.5500, 38.4 TPS, 7.9 GB)

## Historical runtime provenance

Historical CLI bench tg: **31.2 t/s**. The archived source identifies the validation conditions; this is not a new full-context tuning measurement.
