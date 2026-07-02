# Model Cards

Local model cards for GGUF models we run on this rig.
Pattern: 1 card per model, with hardware reqs, sampling, MTP status, and local config baseline.

## Architecture vs Training Technique

**Dense vs MoE** = architecture class (which params activate per token).
**MTP (Multi-Token Prediction)** = training technique (orthogonal to architecture).
Dense and MoE models can both have MTP support.

**MTP model files come in two forms:**
- **Full model + MTP head** (~5+ GB): contains the full base model weights plus an extra prediction head. Can run standalone OR as a draft model for speculative decoding.
- **Draft-only MTP head** (<500 MB): contains only the MTP prediction layers (typically 3-4 extra blocks). NOT a standalone model — must be loaded as draft alongside its base model via `--spec-type draft-mtp --model-draft <head>`. Examples: `DRAFT/gemma-4-12B-it-*-MTP.gguf` (49 tensors, 4 layers each), `mtp-gemma-4-12B-it.gguf`.

Cards:
- [Gemma-4-12B](gemma-4-12b.md)
- [Qwythos-9B-Claude-Mythos-5-1M](qwythos-9b-claude-mythos-5-1m.md)
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md)
- [Qwen-AgentWorld-35B-A3B](qwen-agentworld-35b-a3b.md)
- [Gemma-4-26B-A4B](gemma-4-26b-a4b.md)
- [Ornith-1.0-9B](ornith-1.0-9b.md)
- [Ornith-1.0-35B](ornith-1.0-35b.md) — Q4_K_M (primary) + IQ3_M variant
- [Ornith-1.0-35B IQ3_M](ornith-1.0-35b-iq3_m.md) — IQ3_M variant (rejected: slower than Q4_K_M)
- [VITRIOL technique](vitriol-technique.md) — the Codacus MoE-split strategy

## Open extraction tasks
Unsloth's web docs return ~5k chars per `web_extract` call, truncating longer pages. The following sections are still missing from our cards and should be re-extracted via `browser_navigate` or a longer timeout when needed:
- Qwen3.6 "🦙 Llama.cpp Guide" (canonical command for the model)
- Qwen3.6 "💡 Thinking: Enable/Disable + Preserve Thinking" details
- Gemma 4 "🦙 llama.cpp Guide" and "Recommended Settings" sampling params
