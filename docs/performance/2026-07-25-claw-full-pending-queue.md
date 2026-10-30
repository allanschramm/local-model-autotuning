# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-25-claw-full-pending-queue.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model (basename) | peak VRAM | ctx | notes |
| :--- | ---: | ---: | :--- |
| `Qwythos-9B-v2-Q4_K_M` | 7.7 GB | **32k** | 65k mid-full `VRAM_LIMIT` → retry 32k |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL` | 5.7 GB | 65k | TEMP=1.0 thinking; `n-cpu-moe 40` MTP n=1 |
| `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M` | 7.6 GB | 65k | Weaker than Mythos non-MTP 0.3333 |
| `Ornith-1.0-9B-MTP-Q4_K_M` | 7.0 GB | 32k | Ties Bonsai; faster than Ornith UD |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | 4.1 GB | 65k |  |
| `Qwen3.5-9B-UD-Q4_K_XL` | 7.5 GB | 32k |  |

## Preserved historical runtime tables

| Model (basename) | bench_tg | peak VRAM | ctx | notes |
| :--- | ---: | ---: | ---: | :--- |
| `Qwythos-9B-v2-Q4_K_M` | 44.1 | 7.7 GB | **32k** | 65k mid-full `VRAM_LIMIT` → retry 32k |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL` | 29.5 | 5.7 GB | 65k | TEMP=1.0 thinking; `n-cpu-moe 40` MTP n=1 |
| `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M` | 45.6 | 7.6 GB | 65k | Weaker than Mythos non-MTP 0.3333 |
| `Ornith-1.0-9B-MTP-Q4_K_M` | 63.7 | 7.0 GB | 32k | Ties Bonsai; faster than Ornith UD |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | 29.2 | 4.1 GB | 65k |  |
| `Qwen3.5-9B-UD-Q4_K_XL` | 65.0 | 7.5 GB | 32k |  |
