# Model Cards

Local model cards for GGUF models we run on this rig.
Pattern: 1 card per model, with hardware reqs, sampling, MTP status, and local config baseline.

## Architecture vs Training Technique

**Dense vs MoE** = architecture class (which params activate per token).
**MTP (Multi-Token Prediction)** = training technique (orthogonal to architecture).
Dense and MoE models can both have MTP support.

**MTP model files come in two forms:**
- **Full model + MTP head** (~5+ GB): contains the full base model weights plus embedded `nextn` heads. Run with `--spec-type draft-mtp --spec-draft-n-max N` (no separate draft). Examples: `Qwen3.5-9B-UD-Q4_K_XL.gguf`, `Ornith-1.0-9B-MTP-Q4_K_M.gguf`.
- **Draft-only MTP head** (<500 MB): `gemma4-assistant` layers only. NOT standalone — pair with base via `--spec-draft-model`. Example: `draft/mtp-gemma-4-E4B-it.gguf` (main Gemma UD has **no** `nextn`).

**Detect:** scan GGUF metadata for `nextn` or `gemma4-assistant`. Inventory + TPS matrix: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md) · [session 2026-07-20](../sessions/2026-07-20-small-model-tps-matrix.md).

Cards:
- [Gemma-4-12B](gemma-4-12b.md)
- [Gemma-4-E4B](gemma-4-e4b.md) — **default speed Baseline** (122 t/s with draft MTP)
- [Qwythos-9B-Claude-Mythos-5-1M](qwythos-9b-claude-mythos-5-1m.md) — also notes Qwythos-9B-v2
- [Qwen3.5-9B](qwen3.5-9b.md)
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md)
- [Qwen-AgentWorld-35B-A3B](qwen-agentworld-35b-a3b.md)
- [Gemma-4-26B-A4B](gemma-4-26b-a4b.md)
- [Ornith-1.0-9B](ornith-1.0-9b.md) — UD base + Hub MTP GGUF
- [Ornith-1.0-35B](ornith-1.0-35b.md) — Q4_K_M (primary) + IQ3_M variant
- [Ornith-1.0-35B IQ3_M](ornith-1.0-35b-iq3_m.md) — IQ3_M variant (rejected: slower than Q4_K_M)
- [VITRIOL technique](vitriol-technique.md) — the Codacus MoE-split strategy
- [Bonsai-27B](bonsai-27b.md) — Q1_0 max TPS ~41 t/s @ 131k q4_0 no-spec (upstream CUDA; DSpark/Ternary slower)

## Open extraction tasks
Unsloth's web docs return ~5k chars per `web_extract` call, truncating longer pages. The following sections are still missing from our cards and should be re-extracted via `browser_navigate` or a longer timeout when needed:
- Qwen3.6 "🦙 Llama.cpp Guide" (canonical command for the model)
- Qwen3.6 "💡 Thinking: Enable/Disable + Preserve Thinking" details
- Gemma 4 "🦙 llama.cpp Guide" and "Recommended Settings" sampling params
