# Ornith-1.5-35B-A3B — Model Card (Local)

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/ornith-1.5-35b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF
**License:** MIT
**Local file:** `models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf` (21.7 GB)
**Family:** Ornith-1.5 (ornith-ai; Qwen 3.5 MoE architecture)
**Quantization:** official Q4_K_M (only official GGUF repo)

## Architecture (verified from local GGUF, `gguf.GGUFReader` + `autoresearch/core/model_arch.py:126` `gguf_has_mtp()`, 2026-08-22)
- Causal LM, Qwen 3.5 MoE arch (`qwen35moe.*`), **MoE**
- **`block_count` = 41** (1.0-35B was 40)
- **~35B total / ~3B activated** per card (MoE, 256 experts, 8 active + shared)
- `kv_f16_mb @ 65536 ctx` = 2624 MB (q4_0 KV ≈ 0.7 GB at 65k)
-
**2026-08-25 artifact swap:** official `Q4_K_M` was re-uploaded (post-2026-08-23 fix) with a new MTP head — byte-diff vs the old file: all 4 `nextn` tensors changed, `output.weight`/`token_embd.weight` bit-identical (verified `GGUFReader`+sha256). Old file kept as `Ornith-1.5-35B-Q4_K_M.premtp-fix.gguf`. Measured vectors refer to the **old** artifact; remeasurement pending. The 2026-08-22 MTP dead-end (acceptance 0.38, −25% decode on `n-cpu-moe`) was measured on the old **untrained** head — re-probed 2026-08-26 on the new trained head (see MTP section): acceptance 0.567 but decode still −9.7% — MTP remains a net loss on this MoE. Note: Q4_K_M tensors are packed quant bytes — kurtosis analysis on raw `GGUFReader` data is invalid (2026-08-25 lesson).
- **Layout (verified `gguf_dump.py --no-tensors --json` 2026-09-02):** `embedding_length` 2048; attention 16 heads / **2 kv heads**, `key_length` = `value_length` 256, rope dim 64 @ `freq_base` 1e7; GDN/SSM: conv 4, state 128, group 16, inner 4096, time-step rank 32; experts 256 × 8 active + shared (ffn 512, shared ffn 512); `context_length` 262144. KV total implies ~20 full-attn blocks (41.9 KB/token f16 = measured `kv_f16_mb` 2624 @ 65k); the rest are GDN (constant state, no KV growth).

## Hardware requirements
| Quant | Size |
| --- | --- |
| **Q4_K_M (our pick)** | **21.7 GB file; MoE expert offload → **peak 4.0 GB VRAM** |
| Q5_K_M / Q6_K / Q8_0 | 25.3 / 29.2 / 37.8 GB — bigger file, marginal expert-quality gain |

**Target:** 8 GB-class discrete NVIDIA + 32 GB-class RAM. Experts offloaded to CPU (`--n-cpu-moe 41` auto), active path on GPU (`-ngl 99`). Host-memory preflight est 21.7 GB fits. RAM is the binding resource (not VRAM).

## Recommended settings (HF card, 2026-08-19)
Reasoning model: emits `<think>` blocks; card suggests qwen3-style reasoning/tool-call parsers.

- **General tasks:** TEMP 0.6, TOP_P 0.95, TOP_K 20
- **To reproduce reported benchmarks:** TEMP 1.0

### Reasoning control (verified local template, 2026-08-29)

- Template reads **only `enable_thinking`** (confirmed by `gguf_dump` of embedded `tokenizer.chat_template`; no `reasoning_effort` variable present). → **`--reasoning-effort` is a silent NO-OP** on this GGUF; do not rely on ladder flags.
- Working levers: `--reasoning on/off` (REASONING), `--reasoning-budget N` (REASONING_BUDGET, server-side / template-independent), `REASONING_PRESERVE` (verified `supports_preserve_reasoning = true` for Ornith-1.5 GGUFs per `/props`; seed for agentic, not coding/search).
- Publisher card (2026-08-19) describes reasoning mode `<think>...</think>` with reasoning parser `qwen3`; sampling for reproduction: `temperature=1.0`, `top_p` default.

## MTP (Multi-Token Prediction)
- **Verified tested (2026-08-22).** `gguf_has_mtp() == true`, `qwen35moe.nextn_predict_layers = 1`, 4 tensors (`blk.40.nextn.eh_proj/enorm/hnorm/shared_head_norm`), 753 total. `SPEC_TYPE='draft-mtp'`, `SPEC_DRAFT_N_MAX=4` (binary b10549). Separate fingerprint from the non-MTP vector.
- **Measured on this file — no speedup on this MoE:**
  - 65k `n=1`: bench **27.6** t/s, peak 3.9 GB — trial `1c1bc293`, validation **PASS**
  - 65k `n=2`: bench **24.6** t/s, peak 3.7 GB — trial `e568c5e0`, validation **PASS** (`draft_accept` 0.54, `mean_len` 2.08)
  - 65k `n=4` pre-fix: **rejected**, est 8369 > 7676 (`1329dc50`)
  - 131k `n=4` pre-fix: **rejected**, est 9104 > 7676 (`60ddaec2`) — direct server on :18081 fit at **4243 MB**, `predicted_per_second` 12.9, draft 346 / accepted 38 (11%); post-fix harness `67cb12d9` (ctx 131072, spec 4) bench **18.1** — rejected under TPS_FLOOR 20.0
- **Pareto-dominated:** TPS −0.7% at `n=1`, −11% at `n=2`, −34% at 131k `n=4` vs the 27.8 t/s non-MTP baseline (`f8980537`).
- **Estimator fixed:** MoE workspace zeroed — the VRAM estimator (not the binary) was the limiter; post-fix runs estimate correctly.
- 1.0-35B had NO MTP — do not carry that assumption to 1.5.
- **2026-08-28 (b10549 load log):** all other `blk.40.*` tensors (attn/ffn/expert/shexp) are unused/ignored at load — `blk.40` is the MTP-head carrier only; effective MoE blocks are 0-39 (`--n-cpu-moe 41` still covers everything).
- **2026-08-26 (new trained head, harness ladder, codacus fork, ctx 131k, n-cpu-moe 41):**
  - MTP n-max 2: pp 213.5 t/s, tg **28.0** (−9.7 % vs 31.0 control), acceptance **0.567** (135/238) — trained head tripled acceptance vs the untrained 0.38 but the net decode loss persists (CPU-offloaded MoE: head overhead > accepted-token savings).
  - Cache-only (48 slots): pp 204.2, tg **36.9** (+19 %) — the 35B winner; unchanged from the old file.
  - Stack MTP+cache: tg 31.2, acceptance 0.634 @ ngl 36/ctx 115k — the only keepout-compliant fit (ngl 99/ctx 131k peaks 7700–7896 MB > 7676); ~4–5 ms/token CPU-attention penalty means the true GPU-resident stack ≈ 35–37 ≈ cache — cancels, no gain.
  - Daily profile: cache-only, MTP off.

## VITRIOL / Split strategy (MoE expert offloading)
- Auto `--n-cpu-moe 41` (GGUF block_count) — experts on CPU/RAM, attention + shared expert + routing on GPU.
- 1.0-35B A/B: n-cpu-moe=block_count beat manual 32. Peak VRAM 4.0 GB @ 65k.
- **2026-08-28 ubatch ladder** (b10549 upstream, ctx 131k, q4_0 KV, threads 6, 4142-tok prompt, warm rep2): pp 305.7 @ ub512 → 716.9 @ ub2048 → **1172.9 @ ub6144; tg flat 32.4–33.6 (CPU-expert-stream bound — placement/ubatch do not move decode); peak VRAM 4318 → 4786 → 6306 MB (keepout 7676 OK, ~1.4 GB margin). Manual `-ot` regex ≡ `--n-cpu-moe 41` within noise at ub512 and ub6144 — upstream implements both as the same `LLM_FFN_EXPS_REGEX` override (`common.h:1130/1142`). ub 6144 on the cache profile: rejected — see next bullet. Session: [2026-08-28](../performance/2026-08-28-ornith-35b-ubatch-ot-ab.md).
- **2026-08-28 cache × ubatch (codacus fork) + alias degradation:** `models/traces/ornith-1.5-35b-merged.csv` had been wiped → the fork silently served WITHOUT cache (`cannot open profile ... expert cache disabled`, ~31 t/s); profile is a READ input (no auto-create) — regenerated via the discovery workflow (`llama-moe-trace` ×2 prompts, 49 360 rows). Cache 32 slots + ub 2048 = pp **534.9** / tg **35.1** @6862 MB (keepout-compliant; +26 MB vs ub512, decode gain kept); cache + ub 6144 rejects (7857 > 7676); plain upstream + ub 2048 = 716.9 pp / 33.6 tg @4786 MB. The 48-slot alias fingerprint @131k measured 7729 > 7676 under a fat desktop baseline — 32 slots fits with ~840 MB. Alias re-pointed 2026-08-28 (operator, decode-first): cache 32 slots + ubatch 2048; upstream ub2048 stays documented as the prefill-first alternative.

## Context ladder (2026-09-02, serving-path probes)

Operator ask: ≥200k window on the 8 GB box. Harness preflight cannot launch it: est @ 204800 q4_0 ≈ 8.4 GB (`gguf_kv_f16_mb` KV + `VRAM_MOE_NON_EXPERT_FRAC` 0.28 × file ≈ 5.8 GB) vs the hard clamp `resolve_vram_limit_mb` applies to any configured budget (physical − 512 keepout = 7676) — the 0.28 frac over-reads this MoE's real ~1.8 GB GPU-resident weights ~3×, so **every ctx > ~136k is preflight-rejected regardless of KV type**. Ladder ran via `model-up` alias probes per the `use-harness-not-raw-llama` carve-out (which names the moe-cache ladder explicitly). Device-wide nvidia-smi; q4_0 KV, ubatch 2048, threads 6, `--no-warmup`, single-shot prompts:

| Config | idle+smoke | peak @ fill | pp | tg @ fill |
| --- | --- | --- | --- | --- |
| codacus cache32 @ 204800 | 7555 | **7710 @ 4k fill — breaks keepout** | 925 | 28.2 @ 4k |
| codacus cache24 @ 204800 | 7075 | 7335 @ 110k | 763 | 17.7 @ 110k |
| plain b10549 @ 204800 | 5816 | 5509 @ 145k | 1045 | 18.0 @ 145k |
| plain b10549 @ 262144 | 6051 | 6135 @ 145k | 1061 | 19.5 @ 145k |

**Winner: plain upstream b10549, ctx 262144 (full native GGUF window), q4_0 KV, ubatch 2048** — daily alias re-pointed; expert-cache flags removed. Final-config numbers: shallow-4k pp 1420 / tg 30.4 @ ~5.96 GB; decode-vs-fill curve 30.4 → 19.5 @ 145k → 14.7 @ ~255k. llama.cpp preallocates the entire KV at load (q4_0 @ 262144 ≈ 2.8 GB inside the ~6.0 GB idle), so filling costs attention speed, not VRAM. At deep fill the codacus expert cache buys nothing (17.7 vs 18.0 tg) while costing +1.8 GB — its 131k shallow-decode edge (35.1) does not survive the 200k window trade: −13% shallow decode, +165% prefill, +131k window. No-Trial-claims (single-shot serving probes, not TPS_REPS medians). Do not lower `VRAM_MOE_NON_EXPERT_FRAC` globally to un-block harness Trials at 200k+ — it protects other arches; re-test trigger: model-arch-derived non-expert sizing.

## Our config baseline (Trial 2026-08-19)
- `MODEL = 'Ornith-1.5-35B-Q4_K_M.gguf'`
- `CTX_SIZE = 65536` (family complete-vector context; MoE keeps VRAM low)
- `VRAM_LIMIT_MB = 8000.0` + run env `AUTORESEARCH_SKIP_FREE_CLAMP=1`
- `KV q4_0`, batch 256/128, threads 6/8, `FLASH_ATTN on`, `NO_MMAP False` (mmap — 21.7 GB file pages), `CONT_BATCHING True`, NGL 99, `N_CPU_MOE=None` (auto → 41)
- Sampler: TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0

## Historical runtime evidence (Trial results (2026-08-19 claw-full + coding-10 @ 65k; 2026-08-20 agentic rerun @ 4096 floor))
| Metric | Value |
| --- | --- |
| Status | **on_front** |
| bench_tg | 27.8 t/s |
| Combined TPS | 36.8 |
| Peak VRAM | 3.6–4.0 GB |


**Remaining agentic failures (2/15, 2026-08-20):**
- **T053 (finance US-Steel): 0.00 — 65k ctx ceiling, now proven.** HTTP 400 captured verbatim: `request (124983 tokens) exceeds the available context size (65536 tokens)` (`exceed_context_size_error`, n_prompt_tokens 124983, n_ctx 65536). Accumulated history + `REASONING_PRESERVE` think re-renders ballooned the request past 65k; server log shows one `truncated = 1` (n_tokens 65535) before it. This is the documented next-limiter, now with direct evidence.
Family-wide task weakness.
- No max_tokens cap hits: **0 turns decoded to 4096** in the run log (the 5 `length_stops` flags were `</s>`-stop-string stops — see harness caveat).

## Overthinking (2026-08-25)

Same family behavior as 1.5-9B: a 2026-08-22 operator session shows 18/19 turns thinking (mean 3.7k chars, max 18.8k ≈ 4.7k tokens ≈ 3 min at 27.8 t/s); another session ran near-zero thinking (mean 27 chars) — variance by session. Daily-driver alias already caps at `--reasoning-budget 4096` (≈ 2 min/turn at 27.8 t/s); add `--reasoning-budget-message` and consider 2048 for interactive use. KV q8_0 fits on this MoE (KV 1.3→2.6 GB @131k) but does not change think length — budget is the lever.

## Sources / Verification

- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF (README + benchmark table, 2026-08-29): publisher claim — coding (SWE-bench Verified 79 / Terminal-Bench 2.1 67.8) / reasoning (HLE 25.6 / GPQA 89.2) / agentic (ClawEval 72.5 / MCP-Atlas 70.2); context window 262144 tokens (GGUF `context_length` verified); ~3B activated / 256 experts × 8 active + shared (~35B total / ~70 GB bf16, from card); quant Q4_K_M = 21.7 GB file (`totalFileSize` 71 GB from HF siblings); license MIT (`license:mit`); sampling `TEMP=0.6` general / `TEMP=1.0` to reproduce benchmarks.
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B (base model card + chat_template.jinja + `config.json` `Qwen3_5MoeForConditionalGeneration`, 2026-08-29): reasoning parser `qwen3`, tool-call parser `qwen3_xml`; `enable_thinking` only (no `reasoning_effort` in Jinja — confirms local `gguf_dump` result); YaRN RoPE scaling (factor 4.0) for ~1M effective window.
- https://huggingface.co/api/models/ornith-ai/Ornith-1.5-35B-A3B-GGUF (JSON: `gguf.architecture=qwen35moe`, `context_length=262144`, `total=35505251456` bytes ≈ 33 GB uncompressed; `lastModified=2026-08-24`; siblings include Q4_K_M / Q5_K_M / Q6_K / Q8_0 / BF16), extraction 2026-08-29.
## Open questions
- MTP (trained head, 2026-08-26): still no net speedup — acceptance 0.567 but decode −9.7 % @131k; stack cancels; cache-only is the winner (36.9). Pre-fix history: 131k n4 fit at 4.24 GB actual vs 9104 est, bench 18.1 < floor.
- T053: **serving side resolved 2026-09-02** — alias window now 262144 (see Context ladder). Harness Trial side remains est-capped ~131k until the MoE preflight frac is recalibrated; REASONING_PRESERVE re-render inflation untested.
- T054: task content/research path is family-wide weak; possible target for a budget/efficiency A/B later.
