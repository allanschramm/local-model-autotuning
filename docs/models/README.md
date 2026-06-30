# Model Cards

Local model cards for GGUF models we run on this rig.
Pattern: 1 card per model, with hardware reqs, sampling, MTP status, and local config baseline.

Cards:
- [Qwythos-9B-Claude-Mythos-5-1M](qwythos-9b-claude-mythos-5-1m.md)
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md)
- [Gemma-4-26B-A4B](gemma-4-26b-a4b.md)
- [VITRIOL technique](vitriol-technique.md) — the Codacus MoE-split strategy

## Open extraction tasks
Unsloth's web docs return ~5k chars per `web_extract` call, truncating longer pages. The following sections are still missing from our cards and should be re-extracted via `browser_navigate` or a longer timeout when needed:
- Qwen3.6 "🦙 Llama.cpp Guide" (canonical command for the model)
- Qwen3.6 "💡 Thinking: Enable/Disable + Preserve Thinking" details
- Gemma 4 "🦙 llama.cpp Guide" and "Recommended Settings" sampling params
