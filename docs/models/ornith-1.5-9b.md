# Ornith-1.5-9B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/ornith-1.5-9b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF
**HF URLs (fetched 2026-08-29):** https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (GGUF repo) · https://huggingface.co/ornith-ai/Ornith-1.5-9B (base model repo)
**License:** MIT (https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE)
**Local file:** `models/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf` (5.63 GB)
**Family:** Ornith-1.5 (ornith-ai; Qwen 3.5 architecture)
**Quantization:** official Q4_K_M (only official GGUF repo; no Unsloth/MTP pack yet)
**MTP repo:** none — dense qwen35; the Q4_K_M GGUF had no MTP tensors at the 2026-08-22 scan (0/427 `nextn`); an upstream re-upload later carried an MTP head, but our local file is the original no-MTP artifact (see Architecture note).

## Architecture (verified from local GGUF, `gguf.GGUFReader` + `autoresearch/core/model_arch.py:126` `gguf_has_mtp()`, 2026-08-22)
- Causal LM, Qwen 3.5 arch (`qwen35.*` fields), **dense**
- **`block_count` = 32**
- `kv_f16_mb @ 65536 ctx` = 4096 MB (q4_0 KV ≈ 1 GB at 65k)
-
**2026-08-25 artifact swap:** official `Q4_K_M` was re-uploaded and now carries an MTP head — 4 `nextn` tensors, `block_count` 33 (= 32 + 1 MTP layer; GGUF `block_count` counts the MTP head; base config `num_hidden_layers` is 32), 442 tensors (verified `GGUFReader`). Old no-MTP file kept as `Ornith-1.5-9B-Q4_K_M.premtp-fix.gguf`. Measured vectors refer to the **old** artifact; remeasurement pending. Head verified **trained** 2026-08-26 (harness ladder): acceptance 0.624, decode **61.6 t/s = +49.5 %** vs control @80k — the family dense-MTP win is real. MTP @100k ctx peaks 7.92 GB (rejected by the 7676 keepout); 80k is the 8 GB-class ceiling with the head.
- **TBD:** exact SSM/attention layout (full_attention_interval, hidden dim) — same file size and arch family as Ornith-1.0-9B Q4_K_M; see [ornith-1.0-9b.md](./ornith-1.0-9b.md) for the family layout.

## Hardware requirements
| Quant | Size |
| --- | --- |
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** + KV cache |
| BF16 (publisher) | ≈ 19 GB VRAM (single 80 GB GPU per HF card) |

**Publisher hardware (https://huggingface.co/ornith-ai/Ornith-1.5-9B, fetched 2026-08-29):** BF16 (≈19 GB) targets a single 80 GB GPU; quantized Q4_K_M / Q5_K_M / Q6_K / Q8_0 are quantized variants in the GGUF repo (file sizes inferred from GGUF directory listing — see `models/ornith-ai/Ornith-1.5-9B-GGUF/`). The 8 GB-class envelope is **not** publisher-supported; the 1.0 / 1.5 family are validated on ≥40 GB. The HF README publishes a quantized `Ornith-1.5-9B-Mobile` variant explicitly for edge/mobile deployment.

## Recommended settings (HF card, 2026-08-19; refetched 2026-08-29)
Reasoning model: emits `<think>` blocks; card suggests qwen3-style reasoning/tool-call parsers at the server. Publisher runtime floors: vLLM ≥ 0.19.1 / SGLang ≥ 0.5.9 / Transformers ≥ 5.8.1. Serving recipes use `--reasoning-parser qwen3` + `--tool-call-parser qwen3_xml` (vLLM) or `qwen3_coder` (SGLang) to surface `reasoning_content` and `tool_calls` as OpenAI-style fields.

- **General tasks:** TEMP 1.0, TOP_P 0.95, TOP_K 20, MIN_P 0.0, presence 1.5, repeat 1.0


**Context window (publisher claim, 2026-08-29):** 262,144 tokens native (HF `gguf.context_length=262144`); YaRN `rope_scaling` with `factor: 4.0` extends the effective window to ~1,048,576 tokens (1M). Publisher validates YaRN on bf16; **TBD:** whether YaRN ships in the Q4_K_M GGUF config — our Q4_K_M file inherits the bf16 RoPE, llama.cpp applies scaling at server start.

**Publisher benchmarks (https://huggingface.co/ornith-ai/Ornith-1.5-9B, fetched 2026-08-29):** 5-run averages on the bf16 checkpoint with vLLM / SGLang / OpenHands / Claude-Code harnesses.
- Coding: Terminal-Bench 2.1 (Terminus-2) **46.2** · Terminal-Bench 2.1 (Claude Code) **47** · SWE-bench Verified **70.6** · SWE-bench Pro **47.5** · SWE-bench Multilingual **54.4** · NL2Repo **32.4** · SWE Atlas QnA **20.6**
- Reasoning: HLE no-tools **20.2** · HLE with-tools **30.5** · GPQA Diamond **86.4**
- Agentic: MCP-Atlas **54.2** · Toolathlon-Verified **41.2** · WideSearch **59.5** · BrowseComp **56.4** · ClawEval **66.5**


## Reasoning control

**Template reads only `enable_thinking`** (verified 2026-08-29, embedded `tokenizer.chat_template` via `gguf_dump.py --no-tensors --json` on the local Q4_K_M GGUF): the template contains zero `reasoning_effort` / `thinking_budget` variables, so `--reasoning-effort` is a **silent no-op** on this GGUF. The card already documents the 2026-08-24 no-op finding (see [Trial 2026-08-24](#trial-2026-08-24--reasoning_budget4096-ab-claw-full--coding-10--65k) below); this 2026-08-29 re-confirmation is the third independent check.

**Working levers on this GGUF:**
- `--reasoning on|off` (Baseline `REASONING`): enables/disables the `<think>` block at the server (the rendering itself is a `--reasoning` / `enable_thinking` toggle).
- `--reasoning-budget N` (Baseline `REASONING_BUDGET`): template-independent think cap; server-side forces the end-of-think tag at exhaustion. 4096 seeded for the 2026-08-24 A/B; 2048 in the daily-driver alias; 2048+message in operator-bumped alias.
- `--reasoning-budget-message "..."` (Baseline `REASONING_BUDGET_MESSAGE`): nudge injected at budget exhaustion — prevents silent truncation.
- `--reasoning-preserve` (Baseline `REASONING_PRESERVE`): full-history think preservation; `GET /props` reports `chat_template_caps.supports_preserve_reasoning = true` for this GGUF (sealed in Baseline 2026-08-19, see [Overthinking behavior](#overthinking-behavior--daily-driver-levers-2026-08-25) below).

## MTP (Multi-Token Prediction)
- **MTP head present and trained (2026-08-25 swap + 2026-08-26 probe).** 4 `nextn` tensors, `block_count` 33 (= 32 + 1 MTP layer), `SPEC_TYPE='draft-mtp'` via `--spec-type draft-mtp --spec-draft-n-max 2` (llama.cpp never auto-enables). Measured 2026-08-26 (b10549, harness ladder): @80k pp 1755 t/s, tg **61.6** (+49.5 % vs 41.2 control @80k), acceptance **0.624; @100k MTP peaks 7924 MB > keepout 7676 — VRAM-rejected, 80k is the ceiling. n-max 4 vs 2 VRAM delta negligible (7961 vs 7924 @100k).

## VITRIOL / Split strategy
- Dense — no expert offload. Full GPU: `-ngl 99`, `N_CPU_MOE=None`.

## Our config baseline (Trial 2026-08-19)
- `MODEL = 'Ornith-1.5-9B-Q4_K_M.gguf'`
- `CTX_SIZE = 65536` (65k proven on family; 131072 risks VRAM kill mid-coding on 8 GB)
- `VRAM_LIMIT_MB = 8000.0` + run env: `AUTORESEARCH_SKIP_FREE_CLAMP=1` (free-VRAM clamp too tight with desktop VRAM use) and `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256` (see note below)
- `KV q4_0`, batch 256/128, threads 6/8, `FLASH_ATTN on`, `NO_MMAP True`, `CONT_BATCHING True`, NGL 99
- Sampler: TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0

## Historical runtime evidence (Trial results (2026-08-19, claw-full + coding-10 @ 65k))
| Metric | Value |
| --- | --- |
| Status | **on_front** (both axes) |
## Historical runtime evidence (Trial 2026-08-24 — REASONING_BUDGET=4096 A/B (claw-full + coding-10 @ 65k))
Requested "reasoning effort medium": `--reasoning-effort` is a **silent no-op on this GGUF** — embedded `tokenizer.chat_template` (7828 chars, `<think>`-based) contains zero `reasoning_effort` variables (verified via `gguf_dump.py --no-tensors --json`; b10549 server accepts `LLAMA_ARG_REASONING_EFFORT`, template never reads it). The harness had no `REASONING_EFFORT` knob at the time (plumbed 2026-08-29 as a config.py-only Baseline key; still a silent no-op on this GGUF's template). Nearest working lever = `--reasoning-budget`; seeded 4096 (operator-instructed; prior Trial rows ran `reasoning_budget:null`).

| Metric | Value |
| --- | --- |
| Status | **on_front** (row `56a3c78f`) |
| bench_tg | 44.1 t/s |
| Combined TPS | 53.6 |
| Peak VRAM | 7.0 GB |

Runtime finding: REASONING_PRESERVE expanded prompts to 69052/104611 tokens and triggered HTTP 400 exceed_context_size_error; other responses stopped at the output cap. This is historical transport/context evidence.


**Harness fixes required for correct measurement (2026-08-19):**
- `autoresearch/benchmarks/agentic_runner.py` Claw-loop turn timeout raised **30 s → 120 s** (reasoning-model `<think>` traces + max_tokens=2048 at ~40-55 t/s exceed 30 s; the original request was truncated).
- Agentic `max_tokens` floor raised **2048 → 4096** in `autoresearch/runners/evaluation.py` + `agentic_runner.py`; Claw turn timeout **240 s → 420 s** (2026-08-19). 1.5-class CoT still exhausted 2048 mid-`<think>` — server log `n_decoded = 2048` exactly; turns now decode 3000–4100 tokens.
- `REASONING_PRESERVE=True` seeded in Baseline (2026-08-19): `GET /props` reports `chat_template_caps.supports_preserve_reasoning = true` for this GGUF; full-history think preservation for agentic continuity.
- `autoresearch/core/llama_runner.py` `dedicated_vram_kill_ceil` now honors `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB` (preflight and runtime monitor were inconsistent; at 65k this model's steady state ~7.7 GB exceeds the default 7676 MB ceiling).

## Overthinking behavior + daily-driver levers (2026-08-25)

Operator benchmark sessions (2026-08-20/21, agent loops) are ~90 % thinking by volume: think blocks mean 3.4k chars/turn (median 1.6k, p90 8.7k, max 17.9k), answer+tool text mean 406 chars; one session had a single 146,893-char think (~35k tokens ≈ 13 min at 44 t/s). Long thinks show **zero self-repetition** (no repeated 8-grams) — verbose RL-trained deliberation, not a repetition loop. Vendor evaluates with max_new_tokens up to 131k / 256k ctx, so unbounded thinking was rewarded.

Lever verdicts (mechanisms verified in pinned-build source, 2026-08-25):
- `--reasoning-budget` (already 4096 in the daily-driver alias, operator-bumped from 2048) is the only direct think cap; it forces the end-of-think tag at exhaustion and is template-independent. 4096 ≈ 93 s/turn at 44 t/s. Add `--reasoning-budget-message "Stop thinking and act now."` — forced cutoff without a nudge degrades quality. Per-request `reasoning_budget_tokens` (OpenAI API) allows per-task override without restart.
- Repeat/presence penalties are windowed to the last 64 tokens by default (`--repeat-last-n` 64) — the card's `presence 1.5` / alias `repeat 1.15` cannot see a 4k think as configured; raise `--repeat-last-n` (e.g. 2048) if A/B-ing presence. Measured zero repetition ⇒ modest expected effect.
- KV q4_0→q8_0 does NOT address think length; on this hybrid arch q4_0@131k measured ≥ q8_0@131k (family A/B 2026-08-25), and q8_0 KV @131k adds ~2.1 GB (peak ~9.5 GB) — physically unloadable on 8 GB-class. Not a loop lever.
- Suggested active profile: budget 2048 + budget message; per-request `reasoning_budget_tokens` override for hard tasks. **TBD:** think-token distribution at budget ∈ {1024, 2048, 4096} × {no message, message}.

## Ornith family comparison (2026-08-19, 8 GB-class, results.tsv ground truth)

| Model (quant) | Peak VRAM |
| --- | --- |
| **1.5-9B Q4_K_M** | 7.2–7.4 GB |
| 1.0-9B UD Q4_K_XL | 7.7–7.8 GB |
| 1.0-9B Q4_K_M (deepreinforce) | 7.8–7.9 GB |
| 1.0-9B MTP Q4_K_M | 7.4 GB |
| **1.5-35B Q4_K_M** (MoE `n-cpu-moe`) | 3.6–4.0 GB |
| 1.0-35B UD Q4_K_XL | 4.9–5.2 GB |
| 1.0-35B UD Q3_K_XL | 4.4–5.2 GB |

**Verdict:**
- **1.5-9B Q4_K_M is the best Ornith for 8 GB-class** — top agentic, top coding, lowest VRAM of the 9B tier.


- 1.0-9B MTP remains the **speed** pick only (64 t/s draft) at the cost of agentic.

## Sources / Verification
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (README, 2026-08-19)
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (HF repo API JSON: tags, `safetensors.parameters`, `cardData.license`, downloads, `lastModified`, `siblings` — fetched 2026-08-29)
- https://huggingface.co/ornith-ai/Ornith-1.5-9B (base-model repo API + README.md: benchmarks, usage, recommended settings, hardware table — fetched 2026-08-29)
- Local GGUF metadata via `scripts/model_info.py` (2026-08-19) + `gguf.GGUFReader` field+tensor scan 2026-08-22 (`autoresearch/core/model_arch.py:126` `gguf_has_mtp()` → false, 0/427 `nextn` tensors, no `nextn_predict_layers`)
**Extraction date:** 2026-08-29

## TPS Hill Climb — dense plateau (2026-08-22, b10549, 8 GB-class discrete NVIDIA, ctx >65k, bench-only no validation)


| Config (ctx, batch/ubatch, threads, KV, flags) | median t/s (3 reps) | harness log | decision |
| --- | --- | --- | --- |
| 100000, 512/128, 8/8, q4_0/q4_0, --no-mmap --flash-attn on --cont-batching** | **44.40 [44.3,44.4,44.4] 0.2%** | `llama-server-20260822-204923` (44.4) + `205115` (6/8 44.3) plateau pair | **winner — persisted to `autoresearch/core/config.py:30,38-39` (VRAM_LIMIT 8000)** |
| 70000, 512/128, 6/8, q4_0 | 44.4 [44.3,44.4,44.4] | `llama-server-20260822-203840` | tie — ctx flat 70k–120k |
| 110000, 512/128, 6/8, q4_0 | 44.3 | `llama-server-20260822-2039xx` | tie |
| 120000, 512/128, 6/8, q4_0 | 44.4 | `llama-server-20260822-204015` max viable q4_0 (7768<7932) | tie |
| 100000, 1024/512, 6/8, q4_0 | 44.3 | `llama-server-20260822-204150` | tie — batch neutral |
| 100000, 512/128, 6/8, q4_0 (vs 8/8) | 44.3 vs 44.4 | `205115` vs `204923` | tie ±0.1 (0.2% noise) — reconciled thread discrepancy (alias 6/8 vs probe leftover 8/8); standardized to 8/8 |
| 100000, 512/128, 8/8, q4_0, --mmap (NO_MMAP False) | 44.4 | `llama-server-20260822-205829` (8/8) + `205329` (6/8) | tie — mmap neutral |
| 100000, 512/128, 8/8, q4_0, --cont-batching off | 44.4 | `llama-server-20260822-205653` | tie — cont-batching neutral |
| 100000, 512/128, 8/8, q4_0, --flash-attn off | ConfigError `FLASH_ATTN must be on` | validate_config:99 | closed — not a valid axis |
| 100000, 512/128, 8/8, q8_0 | preflight FAIL `est 9106MB >7932` | bench harness | **reject** — VRAM (repeat confirmed 2026-08-22) |
| 131072, 512/128, 6/8, q4_0 | bench 44.4 but `est 7962>7932` preflight FAIL | — | needs keepout 0 (risky) — out of scope |
| 100000, 512/128, 6/8, f16 | 44.50 +0.10 | prior sweep | bench win but `est 11918>7932` not server-viable |


**Env for reproduction:** `AUTORESEARCH_LLAMA_CPP_ROOT=b10549`, `AUTORESEARCH_SKIP_FREE_CLAMP=1`, `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256`, `VRAM_LIMIT_MB=8000`, `TPS_REPS=3`.
## Open questions
Retrieval-path failure common to the family, not truncation. Possible target for a budget/efficiency A/B later.
- **65k ctx ceiling is the next limiter (now proven)**: the 1.5-35B rerun hit an HTTP 400 `exceed_context_size_error` — `request (124983 tokens) exceeds the available context size (65536)` (row `f8980537`, T053). `REASONING_PRESERVE` think re-renders inflate request size; candidate A/B: ctx 131072 or `REASONING_PRESERVE` off. Long research tasks can fill 65k at 4096 tokens/turn.
- Full SSM layout for 1.5 (interval/hidden) — verify from GGUF when next card edit happens.

## Preserved historical runtime tables

| Model (quant) | bench_tg | Peak VRAM |
| --- | --- | --- |
| **1.5-9B Q4_K_M** | 43.2–44.3 | 7.2–7.4 GB |
| 1.0-9B UD Q4_K_XL | 41.6–48.6 | 7.7–7.8 GB |
| 1.0-9B Q4_K_M (deepreinforce) | 42.5 | 7.8–7.9 GB |
| 1.0-9B MTP Q4_K_M | 64.0 (draft) | 7.4 GB |
| **1.5-35B Q4_K_M** (MoE `n-cpu-moe`) | 25.4–28.8 | 3.6–4.0 GB |
| 1.0-35B UD Q4_K_XL | 22.7–25.7 | 4.9–5.2 GB |
| 1.0-35B UD Q3_K_XL | 24.1–26.0 | 4.4–5.2 GB |

## Historical runtime provenance

Historical dense Q4_K_M CLI decode plateau: **44.3–44.4 t/s**, using a **4096** CLI context cap. Configured server contexts of 70k?120k had similar CLI timing; this does not establish full-context throughput. The recorded 100k q4_0 preflight estimate was **7418MB**, against a 7932 MB ceiling. Publisher benchmark resource settings (48 GB RAM) are quality-harness metadata in the archived source, not a local fit measurement.
