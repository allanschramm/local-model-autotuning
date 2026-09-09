# Model Cards

Local model cards for GGUF models we run on the operator host.
Pattern: 1 card per model, with hardware reqs, sampling, MTP status, and local config baseline.

Scores live in `results.tsv` + cards/leaderboards.

## Architecture vs Training Technique

**Dense vs MoE** = architecture class (which params activate per token).
**MTP (Multi-Token Prediction)** = training technique (orthogonal to architecture).
Dense and MoE models can both have MTP support.

**MTP model files come in two forms:**
- **Full model + MTP head** (~5+ GB): contains the full base model weights plus embedded `nextn` heads. Run with `--spec-type draft-mtp --spec-draft-n-max N` (no separate draft). Examples: `Qwen3.5-9B-UD-Q4_K_XL.gguf`, `Ornith-1.0-9B-MTP-Q4_K_M.gguf`.
- **Draft-only MTP head** (<500 MB): `gemma4-assistant` layers only. NOT standalone — pair with base via `--spec-draft-model`. Example: `draft/mtp-gemma-4-E4B-it.gguf` (main Gemma UD has **no** `nextn`).

**Detect:** scan GGUF metadata for `nextn` or `gemma4-assistant`. Inventory + TPS matrix: [docs/discovery/small-model-mtp-tps.md](../discovery/small-model-mtp-tps.md) · [session 2026-07-20](../sessions/2026-07-20-small-model-tps-matrix.md).

Cards:
- [Gemma-4-12B](gemma-4-12b.md) — coding **0.5650**; claw-full missing for this basename
- [Gemma-4-E4B](gemma-4-e4b.md) — claw **0.3333** / coding **0.555**
- [Nanbeige4.2-3B](nanbeige4.2-3b.md) — claw **0.2667** / coding **0.2800**; needs a compatible Nanbeige prebuilt release
- [Qwythos-9B-Claude-Mythos-5-1M](qwythos-9b-claude-mythos-5-1m.md) — non-MTP claw **0.3333** / coding **0.640**; MTP complete too
- [Qwen3.5-9B](qwen3.5-9b.md) — UD claw **0.1333** / coding **rejected**; MTP claw **0.2000** / coding **0.495**
- Qwen3.5-4B-MTP (`Qwen3.5-4B-MTP-Q4_K_M.gguf`) — claw **0.2667** / coding **0.385**; no dedicated card yet
- [Qwen3.6-35B-A3B](qwen3.6-35b-a3b.md) — claw **0.5333** / coding **0.4300** (UD-Q4_K_XL @ 100k no-spec; Q3_K_XL vector incomplete)
- [Qwen3.8-27B](qwen3.8-27b.md) — UD-IQ1_S dense (text-only GGUF of the multimodal publisher); bench 28.4 t/s @65k; reasoning-effort ladder xhigh/medium/low (the only verified one)
- [Nemotron-3-Nano-4B](nemotron3-nano-4b.md) — dense `nemotron_h` hybrid; agentic 0.7333 / coding 0.5100 @131k; enable_thinking-only; daily budget 2048 unmeasured
- [Qwen-AgentWorld-35B-A3B](qwen-agentworld-35b-a3b.md)
- [Gemma-4-26B-A4B](gemma-4-26b-a4b.md) — claw **0.1333** / coding **0.590**
- [Ornith-1.0-9B](ornith-1.0-9b.md) — UD / MTP / deepreinforce `ornith-1.0-9b-Q4_K_M` are separate Trials
- [Ornith-1.5-9B](ornith-1.5-9b.md) — claw **0.9333** + coding **0.615**; on_front; **best Ornith for 8 GB-class**; family comparison table inside; required 4096-token agentic floor + 420 s Claw timeout + keepout env fixes
- [Ornith-1.5-35B-A3B](ornith-1.5-35b.md) — claw **0.8667** (4096 floor rerun) + coding **0.630**; on_front; MoE `n-cpu-moe 41`; embedded MTP untested
- [Ornith-1.0-35B](ornith-1.0-35b.md) — UD-Q4 / Q3 / deepreinforce Q4_K_M are separate Trials
- [Ornith-1.0-35B IQ3_M](ornith-1.0-35b-iq3_m.md) — IQ3_M variant
- [KAT-Coder-V2.5-Dev](kat-coder-v2.5-dev.md) — IQ4_XS; claw **0.6000** + coding **0.640**
- [Laguna-XS-2.1](laguna-xs-2.1.md) — claw-full **0.6667**; coding **0.195**
- [Bonsai-27B](bonsai-27b.md) — claw **0.4667** / coding **0.455**
- [Ternary-Bonsai-27B](ternary-bonsai-27b.md) — rejected (below TPS floor / PrismML)
- [LFM2.5-1.2B](lfm2.5-1.2b.md) — claw **0.6000** / coding **0.350**
- [LFM2.5-2.6B](lfm2.5-2.6b.md) — claw **0.8667** / coding **0.5050** (post-harness remasure; was false 0.3333)
- [LFM2.5-8B-A1B](lfm2.5-8b-a1b.md) — claw ≤0.20 / coding **0.365**
- [POCKET-35B](pocket-35b.md) — claw **0.6667** + coding **0.615**; **Night** pick
- [POCKET-26B](pocket-26b.md) — claw **0.2000** + coding **0.490** (historical)
- [Spark-X2.5-4B](spark-x2.5-4b.md) — dense `spark2_5` (requires llama.cpp $\ge$ `b10828`); bench **73.1 t/s**, quick smoke **1.0000** (5/5), peak VRAM 6.8 GB @ 131k
- [VITRIOL technique](vitriol-technique.md) — stock MoE `--n-cpu-moe` + study notes on Randozart DMA fork (not Trial engine)

Pareto frontier: [pareto-leaderboard.md](../discovery/pareto-leaderboard.md).  
Claw-Eval ranks: [claw-eval-leaderboard.md](../discovery/claw-eval-leaderboard.md).  
Coding-10 ranks: [coding-leaderboard.md](../discovery/coding-leaderboard.md).

## Open extraction tasks
All three previously-truncated Unsloth sections were re-extracted on **2026-08-02** (primary sources) and applied to the cards:
- Qwen3.6 "🦙 Llama.cpp Guide" → canonical command (q8_0 KV) in [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) MTP section.
- Qwen3.6 "💡 Thinking: Enable/Disable + Preserve Thinking" → `enable_thinking` / `preserve_thinking` kwargs; harness seed is Baseline `REASONING_PRESERVE` in [qwen3.6-35b-a3b.md](qwen3.6-35b-a3b.md) Recommended settings.
- Gemma 4 "🦙 llama.cpp Guide" + "Recommended Settings" → `TEMP=1.0 / TOP_P=0.95 / TOP_K=64` + guide command in [gemma-4-26b-a4b.md](gemma-4-26b-a4b.md).
Evidence: [docs/sessions/2026-08-02-research-gap-closure.md](../sessions/2026-08-02-research-gap-closure.md).
