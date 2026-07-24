# `docs/models/` — Model Card Contract

<!-- Scope: any agent adding or editing a model card under /docs/models. -->

## Purpose
One markdown file per GGUF model we run on this rig. Cards are the canonical local reference: where the file lives, what quant it is, what the model architecture actually is (verified from the GGUF, not the HF card), recommended inference settings, MTP/QAT status, and what local config baseline to start from.

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
- **Mark TBDs explicitly.** Anything we couldn't verify (extraction truncated, doc missing) gets a `**TBD:**` marker and a row in the "Open questions" section. Never invent values.

## Work Guidance

### Required sections (in this order)
1. **Header block** — Source repo, Unsloth/docs URL, MTP-specific repo (if exists), license, local absolute file path, symlink path, family, quantization.
2. **Architecture** — verified from GGUF metadata: `block_count`, hidden dim, ctx, expert counts, attention pattern (full / sliding / DeltaNet / hybrid), shared expert, tensor types for embd/output.
3. **Hardware requirements** — Unsloth's published table for the quant row we chose + warnings (CUDA version, offload, OOM risk).
4. **Recommended settings** — sampling params (thinking vs instruct), max context, output length. Cite the Unsloth doc.
5. **MTP section** — does THIS GGUF contain MTP tensors? If not, where do we get MTP from? Which flags? Plus a `verified from our common/arg.cpp` note so we don't re-introduce the `draft-mtp` bug.
6. **VITRIOL split** — the Codacus 2-knob MoE strategy: `--n-gpu-layers` + `--n-cpu-moe N`. Cite the YouTube source.
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
- [`laguna-xs-2.1.md`](./laguna-xs-2.1.md) — Laguna-XS-2.1 MoE; best claw-full Val Score (0.6667).
- [`ternary-bonsai-27b.md`](./ternary-bonsai-27b.md) — Ternary Q2_0 (rejected/deleted; PrismML; ~10.6 t/s).
- [`lfm2.5-1.2b.md`](./lfm2.5-1.2b.md) — LFM2.5-1.2B dense; preferred 65k f16; 128k KV matrix.
- [`lfm2.5-8b-a1b.md`](./lfm2.5-8b-a1b.md) — LFM2.5-8B-A1B MoE hybrid (`lfm2moe`); full VRAM `n-cpu-moe 0`.
- [`gemma-4-12b.md`](./gemma-4-12b.md) — Gemma 4 12B model card.
- [`gemma-4-26b-a4b.md`](./gemma-4-26b-a4b.md) — Gemma 4 26B A4B MoE model card.
- [`gemma-4-e4b.md`](./gemma-4-e4b.md) — Gemma 4 E4B model card.
- [`ornith-1.0-9b.md`](./ornith-1.0-9b.md) — Ornith 1.0 9B model card.
- [`ornith-1.0-35b.md`](./ornith-1.0-35b.md) — Ornith 1.0 35B model card.
- [`ornith-1.0-35b-iq3_m.md`](./ornith-1.0-35b-iq3_m.md) — Ornith 1.0 35B IQ3_M quant model card.
- [`qwen-agentworld-35b-a3b.md`](./qwen-agentworld-35b-a3b.md) — Qwen AgentWorld 35B A3B model card.
- [`qwen3.5-9b.md`](./qwen3.5-9b.md) — Qwen 3.5 9B model card.
- [`qwen3.6-35b-a3b.md`](./qwen3.6-35b-a3b.md) — Qwen 3.6 35B A3B model card.
- [`qwythos-9b-claude-mythos-5-1m.md`](./qwythos-9b-claude-mythos-5-1m.md) — Qwythos 9B Claude Mythos model card.
- [`nanbeige4.2-3b.md`](./nanbeige4.2-3b.md) — Nanbeige4.2-3B looped dense (arch fork required).
- [`vitriol-technique.md`](./vitriol-technique.md) — Codacus VITRIOL technique note (2-knob MoE offload strategy).
