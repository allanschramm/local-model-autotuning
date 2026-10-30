# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/coding-leaderboard.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model | ctx | Note |
| :--- | ---: | :--- |
| `Bonsai-27B-Q1_0` | **65k** | [vs POCKET](2026-07-26-bonsai-coding-vs-pocket.md) |
| `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS` | **65k** | [2026-07-27](2026-07-27-kat-coder-v2.5-dev-pipeline.md); |
| `LFM2.5-1.2B-Instruct-Q8_0` | 65k f16 | LCB patched† |
| `Laguna-XS-2.1-Q3_K_XL` | 65k | Best claw-full; weak coding |
| `Ornith-1.0-35B-UD-Q3_K_XL` | **65k** | Q3 A/B vs Q4; |
| `Ornith-1.0-35B-UD-Q4_K_XL` | **65k** | same session; |
| `Ornith-1.0-9B-MTP-Q4_K_M` | **32k** | MTP n=4; |
| `Ornith-1.0-9B-UD-Q4_K_XL` | **32k** | LCB patched†; claw @ 65k |
| `POCKET-26B-Q4_K_M` | **65k** | GGUF **deleted; |
| `POCKET-35B-Q3_K_M` | **65k** | [2026-07-26](2026-07-26-pocket-35b-pipeline.md); |
| `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M` | **65k** | ; GGUF deleted 2026-07-27; TSV keep |
| `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M` | 131k | Historical KEEP |
| `Qwythos-9B-v2-Q4_K_M` | **32k** |  |
| `gemma-4-12B-it-qat-UD-Q4_K_XL` | 131k | Historical KEEP; GGUF may be gone |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | **65k** | [2026-07-27](2026-07-27-incomplete-vectors-pareto.md); |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL` | **65k** | draft-mtp; |
| `ornith-1.0-9b-Q4_K_M` | 131k | deepreinforce official Q4_K_M (≠ UD); claw-full still missing for this basename |


| Model | Attempts | Note |
| :--- | :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL` | 32k+MTP; 65k no MTP | VRAM kill mid-HE both; only |


| Job | Prefer |
| :--- | :--- |
| Historical run runtime | **POCKET-35B** → **KAT-Coder** → **Ornith-9B-UD** / **Ornith-9B-MTP** (Laguna GGUF deleted |
| Historical run runtime | Mythos / **KAT-Coder** → **POCKET-35B** → Ornith-9B/35B-Q4 @ fit ctx → LFM only if need speed |
| Day supervised | **Qwen3.8-4B-Q4_K_M** ([pareto-leaderboard.md](pareto-leaderboard.md) Day pick; ADR 0017 IQ-first, TPS breaks near-ties) |
