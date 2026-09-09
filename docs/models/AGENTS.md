# `docs/models/` — Model Card Contract

<!-- Scope: any agent adding or editing a model card under /docs/models. -->

## Purpose
One markdown file per GGUF model we run on the operator host. Cards are the canonical local reference: where the file lives, what quant it is, what the model architecture actually is (verified from the GGUF, not the HF card), recommended inference settings, MTP/QAT status, and what local config baseline to start from.

## Ownership
- `/AGENTS.md` (repo root) owns the MTP/QAT/TurboQuant rule.
- `/docs/AGENTS.md` owns the documentation contract.
- This doc owns the **schema** each card must follow.
- Each model card is a leaf doc — no children.

## Local Contracts
- **Filename:** lowercase, dashes, matches the model id minus the `-GGUF` suffix and the quant suffix. Example: `qwen3.6-35b-a3b.md`, not `Qwen3.6-35B-A3B-UD-Q4_K_M.md`.
- **One model per file.** Technique notes (e.g. `vitriol-technique.md`) are separate from model cards; reference them via link, don't merge.
- **Verify from the GGUF, not the HF card.** Use `gguf.GGUFReader` to read the local file's header — that gives the truth (architecture name, block_count, expert_count, tensor prefixes). The HF card is marketing; the GGUF is what's running.
- **Architecture class (MoE vs dense):** state it explicitly in the Architecture section (`expert_count > 1` ⇒ MoE). Harness `is_moe_model` / VITRIOL / `N_CPU_MOE` gates read **GGUF metadata only** — never filename tokens (`A3B`, `ORNITH`, `LAGUNA`, …). Cards must match the GGUF; do not invent a parallel name filter.
- **MoE config baseline:** start with `N_CPU_MOE=None` (harness → GGUF `block_count`) unless the quant fits physical VRAM — then `N_CPU_MOE=0`. Record the resolved N and measured TPS/VRAM after validation.
- **Sampler seed before Trials:** § Recommended settings is the source of truth for `SAMPLER_DEFAULTS`. Match the profile to the upcoming job (agentic/general vs coding). Only change sampler after an intentional quality experiment — never as the first Search mutation.
- **Universal sampler fallback:** when a card has **no** § Recommended settings section, seed `SAMPLER_DEFAULTS` from `UNIVERSAL_FALLBACK_SAMPLER` (defined in `autoresearch/core/config.py` and `config.py.example` — llama.cpp server defaults, neutral start for agentic/general). Never run Trials on the old arbitrary template (`TEMP=0.4`). A card's Recommended settings always win when present.
- **Mark TBDs explicitly.** Anything we couldn't verify (extraction truncated, doc missing) gets a `**TBD:**` marker and a row in the "Open questions" section. Never invent values.

## Work Guidance

### Required sections (in this order)
1. **Header block** — Source repo, Unsloth/docs URL, MTP-specific repo (if exists), license, local absolute file path, symlink path, family, quantization.
2. **Architecture** — verified from GGUF metadata: `block_count`, hidden dim, ctx, expert counts, attention pattern (full / sliding / DeltaNet / hybrid), shared expert, tensor types for embd/output.
3. **Hardware requirements** — Unsloth's published table for the quant row we chose + warnings (CUDA version, offload, OOM risk).
4. **Recommended settings** — sampling params by job when the publisher splits them (e.g. thinking/general vs precise coding vs instruct). Include TEMP / TOP_P / TOP_K / MIN_P / presence / repeat. Agents **must seed** `SAMPLER_DEFAULTS` from this section before the first Trial — do not leave the template defaults. Cite Unsloth/HF. Optional engine: `REASONING_PRESERVE` (`None` / `True` / `False`) when the publisher documents preserved thinking **and** local `/props` `supports_preserve_reasoning` is true — seed for agentic, not coding, not Search.
5. **MTP section** — does THIS GGUF contain MTP tensors? If not, where do we get MTP from? Which flags? Plus a `verified from our common/arg.cpp` note so we don't re-introduce the `draft-mtp` bug.
6. **MoE split (“VITRIOL split”)** — stock 2-knob: `--n-gpu-layers` + `--n-cpu-moe N` (experts on **CPU**). Section title kept for card continuity. Not the Randozart DMA fork — see [`vitriol-technique.md`](./vitriol-technique.md).
7. **Our config baseline (TBD)** — concrete flag values to start from. Mark TBD until we run.
8. **Sources / Verification** — URL + extraction date for every external claim. Note if truncation occurred.
9. **Open questions** — bulleted list of TBDs with what we need to resolve them.

### Reading order for new agents
1. `/AGENTS.md` (repo root) — global rules.
2. `/docs/AGENTS.md` — doc contract.
3. THIS file — model card schema.
4. The model card itself.

### Updating an existing card
- After any new benchmark or run that affects the card, update only the relevant section. Don't rewrite the whole file.
- After a new TBD is resolved, remove the TBD marker AND the row from "Open questions" — don't leave dangling markers.
- If the GGUF file changes (re-download, new quant), re-run `gguf.GGUFReader` and update the Architecture section. The model name and layer count can change between releases.

## Verification
- Every card's Architecture section must have values that match the local file (spot-check `block_count`, `expert_count`, `head_count_kv` against `GGUFReader` output).
- MTP section must say whether the local GGUF has MTP tensors (call `GGUFReader` and grep for `mtp` / `speculative` / `draft` in field keys).
- "Sources / Verification" must list URLs with extraction dates.
- "Open questions" must be empty for production-use models; new TBDs are allowed but must be dated and actionable.

## Child DOX Index
- [`README.md`](./README.md) — index and summary of available GGUF model cards.
- [`bonsai-27b.md`](./bonsai-27b.md) — Bonsai 27B model card.
- [`laguna-xs-2.1.md`](./laguna-xs-2.1.md) — Laguna-XS-2.1 MoE; best claw-full (0.6667); weak coding (0.195).
- [`ternary-bonsai-27b.md`](./ternary-bonsai-27b.md) — Ternary Q2_0 (PrismML; ~10.6 t/s on 8 GB-class).
- [`lfm2.5-1.2b.md`](./lfm2.5-1.2b.md) — LFM2.5-1.2B; claw 0.6000 / coding 0.350.
- [`lfm2.5-2.6b.md`](./lfm2.5-2.6b.md) — LFM2.5-2.6B Q8_0 dense `lfm2`; agentic 0.8667 / coding 0.5200 @65k; no thinking template vars; HF-refreshed 2026-08-29.
- [`lfm2.5-8b-a1b.md`](./lfm2.5-8b-a1b.md) — LFM2.5-8B-A1B MoE hybrid (`lfm2moe`); full VRAM `n-cpu-moe 0`.
- [`gemma-4-12b.md`](./gemma-4-12b.md) — Gemma 4 12B model card.
- [`gemma-4-26b-a4b.md`](./gemma-4-26b-a4b.md) — Gemma 4 26B A4B; claw 0.1333 / coding 0.590 @ 65k.
- [`gemma-4-e4b.md`](./gemma-4-e4b.md) — Gemma 4 E4B model card.
- [`nemotron3-nano-4b.md`](./nemotron3-nano-4b.md) — NVIDIA Nemotron-3-Nano-4B Q4_K_M dense `nemotron_h` hybrid; agentic 0.7333 / coding 0.5100 @131k; enable_thinking-only template (created 2026-08-29).
- [`ornith-1.0-9b.md`](./ornith-1.0-9b.md) — UD / MTP / deepreinforce Q4_K_M are separate Trials (coding 0.580 on deepreinforce basename; claw pending for that id).
- [`qwen3.8-4b-distill.md`](./qwen3.8-4b-distill.md) — Qwen3.8-4B-Distill Q4_K_M; **on_front claw 0.8667 / coding 0.6400 @131k, bench 74.9**; dense Gated DeltaNet hybrid; current Baseline pick (2026-08-23).
- [`ling-3.0-tiny.md`](./ling-3.0-tiny.md) — Ling-3.0-Tiny Q4_K_M MoE; dominated coding 0.39 but **agentic 0.8667 @ only 2.5 GB peak** — VRAM-efficient fallback (2026-08-23).
- [`qwen3.8-27b.md`](./qwen3.8-27b.md) — Qwen3.8-27B UD-IQ1_S dense `qwen35`; bench tps 28.4 @65k; template was the only verified `reasoning_effort` ladder (xhigh/medium/low, verified 2026-08-29).
- [`ornith-1.5-9b.md`](./ornith-1.5-9b.md) — Ornith-1.5-9B official Q4_K_M; on_front claw **0.9333** / coding **0.6150**; dense `qwen35` **no MTP** (0/427, 2026-08-22 `GGUFReader`); needed 4096-token agentic floor + 420 s Claw timeout + keepout env fixes.
- [`ornith-1.5-35b.md`](./ornith-1.5-35b.md) — Ornith-1.5-35B-A3B official Q4_K_M; on_front claw **0.8667** / coding **0.6300**; MoE `n-cpu-moe 41` auto; **embedded MTP** `qwen35moe.nextn_predict_layers=1` 4/753 `blk.40.nextn.*` (2026-08-22).
- [`ornith-1.0-35b.md`](./ornith-1.0-35b.md) — Ornith 1.0 35B; Q4 claw 0.6000 / coding 0.580; Q3 claw 0.4667 / coding 0.555.
- [`ornith-1.0-35b-iq3_m.md`](./ornith-1.0-35b-iq3_m.md) — Ornith 1.0 35B IQ3_M quant model card.
- [`qwen-agentworld-35b-a3b.md`](./qwen-agentworld-35b-a3b.md) — Qwen AgentWorld 35B A3B model card.
- [`qwen3.5-9b.md`](./qwen3.5-9b.md) — Qwen 3.5 9B; claw 0.1333; coding-10 **rejected** (VRAM) on 8 GB.
- [`qwen3.6-35b-a3b.md`](./qwen3.6-35b-a3b.md) — Qwen 3.6 35B A3B model card.
- [`qwythos-9b-claude-mythos-5-1m.md`](./qwythos-9b-claude-mythos-5-1m.md) — Qwythos 9B Claude Mythos model card.
- [`nanbeige4.2-3b.md`](./nanbeige4.2-3b.md) — Nanbeige4.2-3B looped dense (arch fork required).
- [`kat-coder-v2.5-dev.md`](./kat-coder-v2.5-dev.md) — KAT IQ4_XS; claw **0.6000** + coding **0.640**.
- [`pocket-35b.md`](./pocket-35b.md) — POCKET-35B Q3_K_M; claw **0.6667** + coding **0.615**; Night pick.
- [`pocket-26b.md`](./pocket-26b.md) — POCKET-26B; historical claw **0.2000** / coding **0.490**.
- [`spark-x2.5-4b.md`](./spark-x2.5-4b.md) — Spark-X2.5-4B dense `spark2_5` (requires llama.cpp $\ge$ `b10828`); bench **73.1 t/s**, quick smoke **1.0000** (5/5), peak VRAM 6.8 GB @ 131k.
- [`vitriol-technique.md`](./vitriol-technique.md) — stock `--n-cpu-moe` path + absorbed notes from Randozart/VITRIOL (fork = study only).
