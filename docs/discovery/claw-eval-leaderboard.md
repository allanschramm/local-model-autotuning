# Claw-Eval Leaderboard (Local Rig)

Canonical **Claw-Eval full** is the Search Val Score (ADR 0004). Claw-Eval quick is smoke only — high quick does **not** guarantee high full.

Hardware context for numbers below: RTX 4060 **8 GB**, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA unless a card says otherwise.

Ground truth: `results.tsv`. Ignore rows with `val_score` outside `[0, 1]` (historical Autoloop pollution — TPS leaked into score).

## Claw-Eval full (n=15) — ranked KEEP / best discards

| Rank | Model | Val Score | bench_tg | peak VRAM | ctx | Alias | Session / note |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `Laguna-XS-2.1-Q3_K_XL.gguf` | **0.6667** | 37.2 | 3.6 GB | 65k | `laguna-xs` | [2026-07-24](../sessions/2026-07-24-claw-full-smoke-high.md) |
| 2 | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | **0.6000** | 42.1 | 7.5 GB | 65k | `ornith-9b` | same |
| 2 | `Ornith-1.0-35B-UD-Q4_K_XL.gguf` | **0.6000** | 25.7 | 4.9 GB | 65k | `ornith-35b` | `n-cpu-moe 40` |
| 4 | `Qwythos-9B-v2-MTP-Q4_K_M.gguf` | **0.5333** | 34.5 | 7.9 GB | 131k | — | GGUF may be missing; Hub MTP scarce |
| 5 | `Bonsai-27B-Q1_0.gguf` | **0.4667** | 40.2 | 6.5 GB | **65k** | `bonsai` | 131k VRAM-kill mid-agentic |
| 6 | `nanbeige4.2-3b-Q4_K_M.gguf` | **0.2667** | 32.4 | 6.9 GB | 32k | `nanbeige4.2-3b` | arch fork |

Best discard worth noting: Mythos non-MTP full **0.3333** (below Ornith/Laguna keeps).

## Claw-Eval quick (n=5) — smoke ceiling

| Score | Models (examples) |
| :---: | :--- |
| **1.00** | Laguna-XS only (so far) |
| **0.80** | LFM2.5-1.2B @ **65k f16** (alias), Bonsai@131k smoke, Ornith-9B, Ornith-35B, some Qwythos |
| **≤0.40** | Prefer sibling / skip full queue (e.g. Ornith-9B-MTP, LFM2.5-8B-A1B **0.20**) |

Smoke ≠ Val Score: Laguna quick 1.00 → full 0.67; Bonsai quick 0.80 → full 0.47.

## Failure pattern (2026-07-24 queue)

Across Laguna / Ornith / Bonsai full runs:

* **Pass:** tool-heavy easy tasks (email, calendar, todo, contacts, helpdesk, many notes/finance).
* **Fail / near-zero:** long `web_real` research (CVE, OSS compare, regulatory, US Steel, NFLX ARPPU).
* **Implication:** current local leaders are strong at structured tool use; weak at long synthesis / real-web keyword graders.

## Operational lessons

1. **Queue by quick, decide by full.** Run full on quick ≥0.80 first.
2. **Dense @ max ctx can smoke-pass and agentic-fail.** Bonsai 131k: short-gen OK, mid-full `VRAM_LIMIT EXCEEDED used=7906MB > limit=7900MB` → lock agentic alias to **65k**.
3. **MoE prefer `N_CPU_MOE=block_count`.** Ornith-35B A/B: 40 beat 32 (TPS + VRAM). Laguna already at 40.
4. **One Trial at a time.** Shared GPU + port 18080.
5. **Edit `config.py` Baseline, then**  
   `.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full …"`

## Prefer for agentic use (this rig, 2026-07-24)

1. **`laguna-xs`** — best Val Score + lowest VRAM among leaders.
2. **`ornith-9b`** — same Val Score as 35B, higher TPS, denser VRAM.
3. **`ornith-35b`** — same Val Score, more capacity headroom via MoE offload.
4. **`bonsai`** — usable but weaker full; keep ctx 65k for agentic.

## See also

* [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI
* [models/aliases/INDEX.md](../../models/aliases/INDEX.md) — live alias table
* Cards: [laguna-xs-2.1.md](../models/laguna-xs-2.1.md), [ornith-1.0-9b.md](../models/ornith-1.0-9b.md), [ornith-1.0-35b.md](../models/ornith-1.0-35b.md), [bonsai-27b.md](../models/bonsai-27b.md)
