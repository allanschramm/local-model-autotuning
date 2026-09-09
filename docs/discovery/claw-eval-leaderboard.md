# Claw-Eval Leaderboard (Local Rig)

**Claw-Eval full** = agentic Objective Vector axis ([ADR 0006](../adr/0006-pareto-frontier-search.md)). Legacy scalar Val Score display only. Claw-Eval quick is smoke — high quick does **not** guarantee high full.

Hardware: discrete **8 GB-class** NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA unless a card says otherwise.

Ground truth: the results store — canonical `results.db` (SQLite) first, legacy `results.tsv` fallback. Ignore rows with `val_score` outside `[0, 1]` (historical Autoloop pollution — TPS leaked into score). Full leaderboard (every complete vector, IQ-first; ADR 0017): [pareto-leaderboard.md](pareto-leaderboard.md). Leaderboard text is a view of the store — never drop a measured GGUF because it is weak or deleted.

**Thinking / hybrid-reasoning models (2026-08-08):** Pre-fix Claw rows for families that emit `reasoning_content` (Ornith, Qwen3.x thinking, Gemma-4 `enable_thinking`, Qwythos/Mythos, KAT-Coder, Nanbeige, Pocket, …) may be **false lows** (empty graders, HTTP 400, `max_tokens=512`). Ornith UD remasured **0.3333 → 0.9333** @ 65k after the harness fix. **2026-08-19:** rows measured between the 2026-08-08 fix and the 4096-token floor (2048 cap, e.g. Ornith-1.5-9B's 0.8000) are slight understatements for long-CoT families — remeasure when agentic matters. Do not treat those old ranks as IQ. Checklist + remasure policy: [thinking-models-claw-harness.md](thinking-models-claw-harness.md).

## Claw-Eval full (n=15) — ranked by agentic score

| Rank | Model | Val Score | bench_tg | peak VRAM | ctx | Session / note |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `POCKET-35B-Q3_K_M.gguf` | **0.6667** | 35.7 | 3.7 GB | 65k | [2026-07-26](../sessions/2026-07-26-pocket-35b-pipeline.md); coding **0.615**; **Night** |
| 1 | `Laguna-XS-2.1-Q3_K_XL.gguf` | **0.6667** | 37.2 | 3.6 GB | 65k | [2026-07-24](../sessions/2026-07-24-claw-full-smoke-high.md); coding **0.195**; GGUF deleted 2026-07-27 |
| 3 | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | **0.6000** | 42.1 | 7.5 GB | 65k | coding **0.570** @ 32k |
| 3 | `Ornith-1.0-35B-UD-Q4_K_XL.gguf` | **0.6000** | 25.7 | 4.9 GB | 65k | `n-cpu-moe 40`; coding **0.580** @ 65k |
| 3 | `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` | **0.6000** | 30.2 | 3.4 GB | 65k | [2026-07-27](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md); coding **0.640** |
| 3 | `LFM2.5-1.2B-Instruct-Q8_0.gguf` | **0.6000** | 166.4 | 3.7 GB | 65k f16 | [top-TPS](../sessions/2026-07-24-claw-full-top-tps.md); coding min fails ADR 0008 Day IQ band |
| 6 | `Qwythos-9B-v2-MTP-Q4_K_M.gguf` | **0.5333** | 34.5 | 7.9 GB | 131k | GGUF may be missing |
| 7 | `Bonsai-27B-Q1_0.gguf` | **0.4667** | 40.2 | 6.5 GB | **65k** | 131k VRAM-kill mid-agentic; GGUF deleted 2026-07-27 |
| 7 | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | **0.4667** | 63.7 | 7.0 GB | **32k** | coding **0.580** @ 32k; **Day** ADR 0008 ([2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md)) |
| 7 | `Ornith-1.0-35B-UD-Q3_K_XL.gguf` | **0.4667** | 25.2 | 4.5 GB | 65k | Q3 A/B; coding **0.555**; GGUF deleted 2026-07-27 |
| 10 | `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | **0.4000** | 29.5 | 5.7 GB | 65k | TEMP=1.0 thinking; GGUF deleted 2026-07-27 |
| 11 | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | **0.3333** | 113.3 | 5.8 GB | 65k MTP | draft-mtp n=4; coding **0.555** |
| 11 | `Qwythos-9B-v2-Q4_K_M.gguf` | **0.3333** | 44.1 | 7.7 GB | **32k** | 65k VRAM-kill mid-full; GGUF deleted 2026-07-27 |
| 11 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | **0.3333** | 43.6 | 8.0 GB | 131k | coding **0.640** |
| 14 | `nanbeige4.2-3b-Q4_K_M.gguf` | **0.2667** | 32.4 | 6.9 GB | 32k | arch fork; GGUF deleted 2026-07-27 |
| 14 | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | **0.2667** | 45.6 | 7.6 GB | 65k | weaker than non-MTP; GGUF deleted 2026-07-27 |
| 16 | `LFM2.5-8B-A1B-Q4_K_M.gguf` | **0.2000** | 178.5 | 7.8 GB | 65k | `n-cpu-moe 0`; GGUF deleted 2026-07-27 |
| 16 | `POCKET-26B-Q4_K_M.gguf` | **0.2000** | 20.5 | 4.5 GB | 65k | GGUF **deleted**; coding 0.490 |
| 18 | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | **0.1333** | 29.2 | 4.1 GB | 65k | coding **0.590**; weak agentic; GGUF deleted 2026-07-27 |
| 18 | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | **0.1333** | 65.0 | 7.5 GB | 32k | coding-10 **rejected** (VRAM); GGUF deleted 2026-07-27 |

## Claw-Eval quick (n=5) — smoke ceiling

| Score | Models (examples) |
| :---: | :--- |
| **1.00** | Laguna-XS only (so far) |
| **0.80** | **POCKET-35B Q3_K_M**, **KAT-Coder IQ4_XS**, LFM2.5-1.2B @ **65k f16**, Bonsai@131k smoke, Ornith-9B, Ornith-35B, some Qwythos |
| **≤0.40** | Prefer sibling / skip full queue unless curious (e.g. LFM2.5-8B-A1B **0.20** → full ≤0.20; Ornith-9B-MTP quick 0.40 → full **0.4667**) |

Smoke ≠ Val Score: Laguna quick 1.00 → full 0.67; Bonsai quick 0.80 → full 0.47; LFM2.5-1.2B quick 0.80 → full **0.60**; LFM2.5-8B quick 0.20 → full 0.13–0.20.

## Failure pattern — harness (thinking models, 2026-08-08)

Claw / agentic loop ignored `reasoning_content`, dropped it from history, and used `max_tokens=512`. Thinking GGUFs scored false-weak until fixed on `main`. Details + regression checklist: [thinking-models-claw-harness.md](thinking-models-claw-harness.md). Evidence: [2026-08-08 session](../sessions/2026-08-08-thinking-claw-harness-fix.md).

## Failure pattern (2026-07-24 queue)

Across Laguna / Ornith / Bonsai full runs:

* **Pass:** tool-heavy easy tasks (email, calendar, todo, contacts, helpdesk, many notes/finance).
* **Fail / near-zero:** long `web_real` research (CVE, OSS compare, regulatory, US Steel, NFLX ARPPU).
* **Implication:** current local leaders are strong at structured tool use; weak at long synthesis / real-web keyword graders.

## Operational lessons

1. **Queue by quick, decide by full.** Run full on quick ≥0.80 first.
2. **Dense @ max ctx can smoke-pass and agentic-fail.** Bonsai 131k: short-gen OK, mid-full `VRAM_LIMIT EXCEEDED used=7906MB > limit=7900MB` → run agentic claw-full at **65k** on the operator host.
3. **MoE prefer `N_CPU_MOE=block_count`.** Ornith-35B A/B: 40 beat 32 (TPS + VRAM). Laguna already at 40.
4. **One Trial at a time.** Shared GPU + port 18080.
5. **Edit `config.py` Baseline, then**  
   `.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full …"`

## Prefer for agentic use (the operator host, 2026-07-27)

1. **`POCKET-35B-Q3_K_M`** — ties Laguna agentic + far stronger coding (Night / balanced).
2. **`Laguna-XS-2.1-Q3_K_XL`** — ties best agentic; weak coding (GGUF deleted; scores kept).
3. **`Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS`** — agentic 0.6000 + coding **0.640** (ties Mythos coding).
4. **`Ornith-1.0-9B-MTP-Q4_K_M`** — Day pick ADR 0008 (~64 t/s, IQ band).
5. **`Ornith-1.0-9B-UD-Q4_K_XL`** / **`Ornith-1.0-35B-UD-Q4_K_XL`** — agentic 0.6000; Q4 beats Q3 on 35B.
6. **`Bonsai-27B-Q1_0`** — usable but weaker full; keep ctx 65k (GGUF deleted; scores kept).

Skip for agentic: **`LFM2.5-8B-A1B`**, **`gemma-4-26B-A4B`**, **`Qwen3.5-9B`** (full ≤0.20 / 0.13).

## See also

* [thinking-models-claw-harness.md](thinking-models-claw-harness.md) — thinking-model Claw false-fail + remasure policy  
* [pareto-leaderboard.md](pareto-leaderboard.md) — full leaderboard + Day/Night (ADR 0017)  
* [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI  
* [coding-leaderboard.md](coding-leaderboard.md) — direct-coding 10-task ranks  
* Session: [2026-08-08](../sessions/2026-08-08-thinking-claw-harness-fix.md) — harness fix + Ornith 0.9333  
* Session: [2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md) — SSD cleanup + Pareto
