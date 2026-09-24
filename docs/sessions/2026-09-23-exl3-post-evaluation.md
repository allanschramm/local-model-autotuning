# 2026-09-23 — EXL3 social-post evaluation: hype-grade claims mapped to project reality

## Goal

Operator surfaced a long-form X/social-media thread titled *"EXL3 is the biggest paradigm shift in local LLMs right now"* — a tier-by-tier recipe map claiming EXL3 is "the cheapest upgrade in local AI right now," with per-VRAM-tier tok/s numbers, HF pack links, and a follow-up *"models, how to run, sources"* reply. Asked: *"what do you think? It's viable to explore more about exl3?"*

This log captures the analytical pass: which technology claims are factual (the post's real substrate), which performance numbers are vendor- or relay-only with no bench protocol, and how the post's "map" maps onto (i) what this rig already measured, (ii) the tiers this project doesn't operate in. **No commands run, no harness touched, no pack downloaded.** Pure analytical pass over the operator-supplied text.

## Hardware

- `discrete_gpu`, Baseline `VRAM_LIMIT_MB` 7676-class (±200 MB WDDM variance), Windows, 32 GB host RAM.
- The post's tiered map covers VRAM classes this rig does not have (8–12 GB, 16 GB, 24 GB, 32 GB, 48 GB, 96 GB, 128 GB DGX Spark × 1/2/4). Project sits at the bottom rung only.

## Setup

- Source: 2-message thread by anonymous author, published same week. Message 1 = per-tier recipe map with embedded tok/s and BPW claims. Message 2 = model pack links + install instructions + author attribution list.
- Comparator: project measured data from the results store + [`../discovery/fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md) §3/§3.1 (EXL3 3.00bpw probe, 2026-08-31).
- Vendor primary sources cited by name in the post: turboderp (`turboderp/Qwen3.8-27B-exl3`, etc.), Mia-AiLab (`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` + drafter pack), mratsim (`mratsim/GLM-4.6-EXL3`), malaiwah (`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`), tpurtell (`tpurtell/deepseek-v4-flash-0731-exl3-k2-spark`), anoane (`anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw`), Cruz (`vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe` and an `exllamav3` fork at `523ecd3`), PonyExl3 (`beamivalice/PonyExl3`, Metal port), Pollard, BotLab-21.

## Commands

None — analytical only. No GPU/CPU cycles, no `llama-server`, no harness edits, no Baseline restore needed.

## Findings

### 1. Technology claims that hold up (the post's real substrate)

| Claim | Verification | Project locale |
| --- | --- | --- |
| EXL3 is built on QTIP trellis coding (Cornell RelaxML) + Hadamard transform, not weight-by-weight rounding | turboderp README + QTIP papers (Cornell 2024) | [`fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md) §3 |
| One-step quantization via fused Viterbi kernel (minutes for small models on RTX 4090) | `convert.py` v1.4.5 source | same |
| Coherent Llama-3.1-70B at 1.6 bpw with EXL3 | turboderp README + turboderp-published evals | same |
| MTP heads ship inside some Qwen3.6+ packs (`draft_mode: mtp`) | turboderp README arch notes | same §3.1 |
| ExLlamaV3 v1.4.5 ships prebuilt Windows wheels `win_amd64` for `cp310–cp313` × `cu128` × `torch2.10.0` | GitHub releases API (verified 2026-08-31) | same §3 |
| NVFP4 KV-cache quantization available on Ada/Blackwell via ExLlamaV3; Hadamard-4 fallback on Ampere (3090) | ExLlamaV3 release notes | new this session |
| Native 262k–1M context with quantized KV cache (`CACHE_QUANT`, multiple of 256) | ExLlamaV3 release notes + post 2026-09 | new this session |
| The "turboderp dropped MTP head on EXL3 3.00bpw Qwen3.5-9B" finding from project is reproducible ecosystem-wide | HF pack inspection + 2026-08-31 probe audit | §3.1 of the discovery doc |

The technology is real, the math is interesting, and the project's measured data does not contradict the substrate — only the speed claims.

### 2. Performance / recipe claims that don't survive a "where's the bench log" check

Pattern: the post's tier deck is unusually tidy — every VRAM tier has a winner, every winner has a tok/s number, none has a failure case. Examples:

- **1× DGX Spark, Cruz's Flash-Next 176B at 3.05 bpw, 102.6 t/s code decode @200k** — a 320B-class model. The same post claims a *different* 320B-class recipe on the same Spark at **64 t/s @2.05 bpw**. Both are author's own benches; no protocol attached, no permalink, no control for temperature / batch / ctx / prompt distribution. The project's existing coverage of Qwen3.8-Flash-Next measures **19.0–24.4 tok/s** on a 12 GB GPU + 64 GB RAM + NVMe ([`frontier-class-local-ai-guide.md`](../discovery/frontier-class-local-ai-guide.md)) — same model class, different optimization surface, an order of magnitude lower.
- **96 GB RTX PRO 6000 + DeepSeek-V4-Flash K2 + dSpark drafter at 133.6 t/s** — single-card 320B-class decode at 133 tok/s with a drafter accept rate implied large enough to nearly double raw throughput. No primary bench log cited.
- **"+33 % end-to-end with a quantized drafter"** + **"int8 mixers inside a 3-bit pack bought 40 % more decode in four days"** — both attributed to the same anonymous author, both phrased as one-shot before/after with no prompt-set, no temperature axis, no ctx axis.
- **Qwen3.8-27B EXL3 2.20bpw @ 25 t/s on a 12 GB 3060 with Q4 KV + vision on** — cited as *"real user receipt (SC_2.20bpw_H3_V3 pack)"*. No permalink, no bench. Plausible in principle given the trellis thesis, unverified.
- **64 GB Apple Silicon M5 Max running Flash-Next 125B EXL3 hybrid @ 70+ t/s** — claimed as "verified direct, not a relay." No bench log on HF or repo. PonyExl3 (the cited Metal fork) is a real project but their published README numbers don't match the post's range at this scale.
- **"the kit defaults to cu130 torch and that can fight an older system CUDA, pin your torch wheel to match"** — matches project reality (the 2026-08-31 probe pinned `torch 2.10.0+cu128`) but is presented as the *post's* discovered gotcha rather than upstream's documented environment contract.

The post's structural tells are all marketing tells: every tier has a one-line answer, no failure mode, vague attributions ("real user receipt," "verified direct, not a relay"), adverbs that *describe* veracity without proving it.

### 3. Project-measured reality (the bit the post erases)

The post authors its speed claims from tiers the project doesn't run. What we *do* measure on this rig's hardware class:

- **EXL3 3.00bpw on Qwen3.5-9B = 49.6 t/s** (peak VRAM **5437 MB**), base only — turboderp's pack carries **zero MTP tensors** (1363-tensor audit). Verbatim: `RESULT {"mtp": false, "new_tokens": 511, "time_generate": 10.30, "tps": 49.61, "peak_vram_mb": 5437, "stop_reason": null}`.
- **llama.cpp Q4_K_M + MTP** on the same class = **57.3 t/s** (bench_tg 67.5 @32k). llama.cpp Q4_K_M base = 38.7 t/s.
- → **EXL3 base is +28 % vs llama.cpp base but −13 % vs llama.cpp-with-MTP.** llama.cpp stays the daily config.
- **The fair-MTP leg is hardware-blocked on this rig, twice over** (per [`2026-08-31-qwen35-9b-exl3-probe.md`](./2026-08-31-qwen35-9b-exl3-probe.md)):
  - **Wall 1 — head quantization.** `convert.py` v1.4.5 default head 6 bpw (mul1 Viterbi) on a 248K-vocab Qwen3.5 commits **55 GB** of process memory and stays frozen at 0 % progress for 45 min. 8 bpw retry: free RAM 24 → 0.8 GB in ~2 min, pagefile 5.8 GB, circuit-breaker protocol kills. State scales with vocab size; 248K ≈ 2× Llama-class.
  - **Wall 2 — unquantized head doesn't place.** `-hb 16` pack is **7.18 GiB** (bf16 head 2.03 + bf16 embed 2.03 + 3 bpw layers ≈ 3.2). `Model.load` autosplit rejects with `Insufficient VRAM in split for model and cache`. **No escape hatches in v1.4.5**: `Model.load` exposes no embedding-offload (only MoE CPU-offload); `convert.py` has no embedding-bits option.
  - Turboderp's own pack fits only because its vision tower is skipped at text-gen and its head is 4-bpw-quantized — *exactly the head quant Wall 1 forbids on 32 GB RAM*.
- **Ecosystem sweep** (HF `models ls`, 2026-08-31): the only MTP-bearing EXL3 pack for the Qwen3.5 family is `komeijishiki/DeepSeek-V4-Pro-Qwen3.5-9B-EXL3-6.50bpw-H8-V8-MTP8` (published 2026-09-01T01:08Z, unsloth finetune base) — 9.6 GB total, **~3 GB over this rig's placement ceiling**. No ≤4 bpw base-model pack with a head exists. The "head quant" recipe works on bigger-RAM hosts.

The post's `8 to 12 GB` row cites a recipe (Qwen3.8-27B EXL3 2.20bpw @ 25 t/s) that occupies the same hardware class this rig sits in. If the receipt were reproducible by us with current turboderp packs, it would matter. It is not reproducible by us: there is no Qwen3.8-27B turboderp 2.20bpw pack measured on this rig, no `Model.load` debug log, no protocol, no permalink.

### 4. Cross-mapping the post's tier card to project docs

| Post tier | Post's card | Project doc that already covers it (status) |
| --- | --- | --- |
| 8–12 GB RTX 3060/4060 | Qwen3.8-27B EXL3 2.2 bpw @ 25 t/s, 64k ctx, Q4 KV, vision on | measured (2026-08-31 EXL3 probe): rig sits here, but on a different arch and at a different bpw; 49.6 t/s, fair-MTP blocked |
| 16 GB RTX 4080/A5000/5060 Ti | Qwen3.8-27B EXL3 3.0 bpw + MTP @ 55 t/s code decode @110k | not measured; out of rig class |
| 24 GB RTX 3090/4090 | Qwen3.8-27B EXL3 3.5 bpw + DFlash2 5.0 bpw drafter @ 130–150 t/s ceiling, 94–105 sustained, 22 GB VRAM | not measured; out of rig class |
| 32 GB RTX 5090 | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated` @ 155–212 t/s decode, native 262k | not measured |
| 48 GB | `mratsim/GLM-4.6-EXL3` tuned-K @ 3.84 bpw @202k ctx across four 48 GB cards | not measured |
| 96 GB RTX PRO 6000 | DeepSeek-V4-Flash + dSpark (133.6 t/s fastest) / `anoane/DeepSeek-V4-Flash-0731-exl3-2.32bpw` (114.5 t/s @1M ctx reach) | not measured |
| 128 GB DGX Spark × 1 / × 2 / × 4 | Cruz (1×102.6 t/s), Mia-AiLab (2×45.5–51.5 t/s per stream), Pollard + BotLab-21 (4×54.7 t/s × 178.7 at 6 streams, 1M ctx, KV pool 2.56 M tokens) | partial — project covers Flash-Next arch with mmap/NVMe tactics in [`frontier-class-local-ai-guide.md`](../discovery/frontier-class-local-ai-guide.md); measures 19.0–24.4 tok/s @12 GB |
| Apple Silicon (M5 Max / unified memory) | `beamivalice/PonyExl3`, Flash-Next 125B @ 49 GB residence @ 70+ t/s; Qwen3.6-35B-A3B 4.0 bpw @ 68.5 t/s on M5 Max vs 52 on 4090 (same weights) | not measured; rig is not Apple Silicon |

**Coverage gap**: only the bottom rung has on-rig measurement. Everything above 8 GB is unverified on this project, including most of the post's headline numbers. The post's tier map is *exactly* the region's the project doesn't reach, and that's the entire reason it's not falsifiable here.

### 5. Where this *would* matter for the project (hypothetical, decision-trigger only)

- **Rig promotion above 16 GB VRAM**: re-run the §3.1 probe on the new tier. The post's 16 GB row cites Qwen3.8-27B EXL3 3.0 bpw + MTP at 55 t/s. On a host with ≥48 GB RAM, the head-quant wall (Wall 1) lifts and the fair EXL3-MTP comparison becomes feasible.
- **MTP-bearing EXL3 pack at ≤4 bpw base + head ≤6 bpw**: today's only such pack is `komeijishiki/DeepSeek-V4-Pro-Qwen3.5-9B-EXL3-6.50bpw-H8-V8-MTP8` (9.6 GB, over ceiling). When one ships at ≤7 GB total, the daily-config question flips.
- **Upstream ExLlamaV3 adds embedding-offload or vocab-aware head quant**: 2026-08-31 hardware walls dissolve; the 8 GB fair-MTP comparison becomes feasible (open question already logged in `2026-08-31-qwen35-9b-exl3-probe.md`).
- **Independent re-measurement network**: if a peer with ≥16 GB publishes a primary bench log for one of the post's tier cards with reproducible protocol and a per-row config, that single receipt — even just one — converts the post from marketing to measured.

### 6. Verdict (today)

- **Technology substrate** — adopt and credit. EXL3 trellis + Hadamard is the right place to invest attention. QTIP's error-distribution idea is sound and the project's harness doesn't reject EXL3 on principle (only on measured output).
- **Performance claims** — do not adopt as project guidance. Numbers relay as upper bounds from anonymous benches, not measurements. The 8 GB-class figure the post's headline 8–12 GB row implies (~25 t/s on a 27B at 2.2 bpw) is not reproducible on this rig with current packs, and no primary bench log makes it falsifiable here.
- **Format lockout** — still hold. Project's GGUF store + model cards + results-store lineage are GGUF. Adopting EXL3 wholesale is a re-quantization + new harness path, gated by "keep the harness stable" contract.
- **Daily config** — unchanged. llama.cpp Q4_K_M + MTP on the MTP-bearing Qwen3.5-9B GGUF alias stays the daily driver.

## Errors

None — analytical only.

## Decisions

- The post's technology claims are recorded; the post's *performance* claims are not.
- **No fresh probe triggered by today's post.** Re-trigger only on (a) an MTP-bearing EXL3 pack ≤4 bpw head, (b) rig promotion above 8 GB, or (c) upstream embedding-offload / vocab-aware head-quant change.
- **No update** to [`fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md). Its §3.1 *"Why not adopt now"* still holds and now has a contemporary external marker (this social thread circulated 2026-09-W3) — the verdict aged correctly without requiring an edit.
- **No update** to `config.py` (Baseline); daily config unchanged.
- **No wiki entity created** for the post — this analysis is project-scoped; the existing 2026-08-31 probe + §3.1 already carry the durable measured facts.

## Open questions

- Does anyone from the project's authorship / review network publish independent EXL3-vs-llama.cpp benches on ≥16 GB? (Unseen as of 2026-09-23.)
- Does the `komeijishiki/DeepSeek-V4-Pro-Qwen3.5-9B-EXL3-6.50bpw-H8-V8-MTP8` pack actually deliver a fair-MTP delta over llama.cpp-with-MTP? (We can't run it; it's 9.6 GB and the rig is 8 GB.)
- Does upstream ExLlamaV3 add embedding-offload or head-bpw-per-vocab-tier? (Tracked in `2026-08-31-qwen35-9b-exl3-probe.md` open questions.)
- Will a MTP-bearing turboderp *base-model* EXL3 pack ship at ≤4 bpw total? Today's only such pack is from a third-party finetuner over the 9B base, not the 70B-class tier the post's claims really need.
- The post's spark-tier recipe relies on a specific fork (`vcruz305/exllamav3` at `523ecd3`) and a `vllm-exl3` plugin — neither is integrated into the project's pinned ExLlamaV3 baseline. If a future operator wants to validate the Spark row, the fork delta is part of the gap.

## Cross-links

- [`docs/discovery/fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md) §3, §3.1, *"Why not adopt now"* — measured table + verdict + format-lockout / model-coverage / context-regime / harness-contract / Windows-build-burden rationale.
- [`docs/discovery/frontier-class-local-ai-guide.md`](../discovery/frontier-class-local-ai-guide.md) — Qwen3.8-Flash-Next architecture + project-side 19.0–24.4 tok/s measurement @ 12 GB GPU + 64 GB RAM + NVMe.
- [`docs/sessions/2026-08-31-qwen35-9b-exl3-probe.md`](./2026-08-31-qwen35-9b-exl3-probe.md) — the empirical EXL3 probe this analysis depends on.
- [`docs/sessions/2026-08-24-codacus-fork-validation.md`](./2026-08-24-codacus-fork-validation.md) — circuit-breaker protocol (applies to quantizers, not just llama-server).
