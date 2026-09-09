# Coding Preflight Leaderboard (Local Rig)

Direct-coding score (optional Search preflight):  
`coding = 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode`  
Exactly **10 tasks** per dataset. Ground truth: the results store — canonical `results.db` (SQLite) first, legacy `results.tsv` fallback. Docs are a secondary view — include every fair 10-task row from the store (`on_front` | `dominated` | `incomplete` | `rejected`), including deleted GGUFs.

Hardware: discrete **8 GB-class** NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream CUDA unless noted.

Claw-Eval full is the agentic axis — see [claw-eval-leaderboard.md](claw-eval-leaderboard.md). Full leaderboard (every complete vector, IQ-first; ADR 0017): [pareto-leaderboard.md](pareto-leaderboard.md).

## Ranked (best per model/quant, fair 10-task)

| Rank | Model | coding | LCB | HE | MBPP | BC | ctx | Note |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | ---: | :--- |
| 1 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M` | **0.6400** | 0.50 | 0.90 | 0.90 | 0.10 | 131k | Historical KEEP |
| 1 | `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS` | **0.6400** | 0.50 | 0.90 | 0.90 | 0.10 | **65k** | [2026-07-27](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md); claw-full **0.6000** |
| 3 | `POCKET-35B-Q3_K_M` | **0.6150** | 0.50 | 0.80 | 0.90 | 0.10 | **65k** | [2026-07-26](../sessions/2026-07-26-pocket-35b-pipeline.md); claw-full **0.6667** |
| 3 | `Qwythos-9B-v2-Q4_K_M` | **0.6150** | — | — | — | — | **32k** | claw-full 0.3333 |
| 5 | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | **0.5900** | 0.40 | 1.00 | 0.80 | 0.00 | **65k** | [2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md); claw-full **0.1333** |
| 6 | `Ornith-1.0-35B-UD-Q4_K_XL` | **0.5800** | 0.40 | 0.90 | 0.80 | 0.10 | **65k** | same session; claw-full **0.6000** |
| 6 | `Ornith-1.0-9B-MTP-Q4_K_M` | **0.5800** | 0.40 | 0.80 | 0.90 | 0.10 | **32k** | MTP n=4; claw-full 0.4667 |
| 8 | `ornith-1.0-9b-Q4_K_M` | **0.5800** | 0.40 | 0.80 | 0.90 | 0.10 | 131k | deepreinforce official Q4_K_M (≠ UD); claw-full still missing for this basename |
| 9 | `Ornith-1.0-9B-UD-Q4_K_XL` | **0.5700** | 0.40 | 0.90 | 0.70 | 0.20 | **32k** | LCB patched†; claw @ 65k |
| 10 | `gemma-4-12B-it-qat-UD-Q4_K_XL` | **0.5650** | 0.40 | 0.80 | 0.90 | 0.00 | 131k | Historical KEEP; GGUF may be gone |
| 11 | `Ornith-1.0-35B-UD-Q3_K_XL` | **0.5550** | 0.40 | 0.90 | 0.70 | 0.10 | **65k** | Q3 A/B vs Q4; claw-full **0.4667** |
| 11 | `gemma-4-E4B-it-qat-UD-Q4_K_XL` | **0.5550** | — | — | — | — | **65k** | draft-mtp; claw-full 0.3333 |
| 12 | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M` | **0.5500** | 0.50 | 0.80 | 0.70 | 0.00 | **65k** | claw-full **0.2667**; GGUF deleted 2026-07-27; TSV keep |
| 13 | `POCKET-26B-Q4_K_M` | **0.4900** | 0.50 | 0.60 | 0.60 | 0.10 | **65k** | GGUF **deleted**; claw-full 0.2000 |
| 14 | `Bonsai-27B-Q1_0` | **0.4550** | 0.40 | 0.40 | 0.80 | 0.10 | **65k** | [vs POCKET](../sessions/2026-07-26-bonsai-coding-vs-pocket.md) |
| 15 | `LFM2.5-1.2B-Instruct-Q8_0` | **0.3500** | 0.10 | 0.50 | 0.70 | 0.10 | 65k f16 | LCB patched† |
| 16 | `Laguna-XS-2.1-Q3_K_XL` | **0.1950** | 0.20 | 0.20 | 0.30 | 0.00 | 65k | Best claw-full; weak coding |

† HE/MBPP/BC from original coding-10 Trial; LCB re-measured via `scripts/lcb_only.py` after Windows cache fix — see [session](../sessions/2026-07-24-lcb-patch-gambiarra.md).

### Rejected (no coding vector)

| Model | Attempts | Note |
| :--- | :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL` | 32k+MTP; 65k no MTP | VRAM kill mid-HE both; claw-full 0.1333 only |

## Operational lessons

1. **Dense coding can VRAM-kill at agentic ctx.** Ornith UD claw-full OK @ 65k; coding @ 65k hit 7950 MB → lock coding Baseline to **32k** on the operator host. Qwen3.5-9B: even **65k no MTP** and **32k+MTP** kill mid coding-10.
2. **LCB cache must be a real file on Windows.** Symlink → `WinError 1314`. Harness now `shutil.copy2`.
3. **Sandbox timeout 30s** on generated-code subprocess (infinite loops hang otherwise).
4. **Agentic ≠ coding.** Laguna claw-full **0.6667** / coding **0.195**. Gemma-26B coding **0.59** / claw **0.13**.
5. **Quants are separate Trials.** Ornith-35B Q3 vs Q4 both measured — Q4 wins quality.
6. CLI: edit `config.py`, then  
   `.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 …"`  
   LCB-only remeasure: `.\venv\Scripts\python.exe scripts\lcb_only.py`

## Prefer by job (the operator host)

| Job | Prefer |
| :--- | :--- |
| Agentic / tools | **POCKET-35B** → **KAT-Coder** → **Ornith-9B-UD** / **Ornith-9B-MTP** (Laguna GGUF deleted; scores kept) |
| Direct coding preflight | Mythos / **KAT-Coder** → **POCKET-35B** → Ornith-9B/35B-Q4 @ fit ctx → LFM only if need speed |
| Balanced (agentic + coding) | **POCKET-35B** (claw 0.67 + coding 0.62) or **KAT-Coder** (claw 0.60 + coding 0.64) |
| Day supervised | **Qwen3.8-4B-Q4_K_M** ([pareto-leaderboard.md](pareto-leaderboard.md) Day pick; ADR 0017 IQ-first, TPS breaks near-ties) |

## See also

* [pareto-leaderboard.md](pareto-leaderboard.md) — global frontier + Day/Night  
* [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI  
* [claw-eval-leaderboard.md](claw-eval-leaderboard.md) — agentic ranks  
* Sessions: [coding-10](../sessions/2026-07-24-coding-10-claw-leaders.md), [2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md), [KAT-Coder](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md)
