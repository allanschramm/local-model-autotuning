# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/claw-eval-leaderboard.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model | peak VRAM | ctx |
| :--- | :---: | :---: |
| `Bonsai-27B-Q1_0.gguf` | 6.5 GB | **65k** |
| `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` | 3.4 GB | 65k |
| `LFM2.5-1.2B-Instruct-Q8_0.gguf` | 3.7 GB | 65k f16 |
| `LFM2.5-8B-A1B-Q4_K_M.gguf` | 7.8 GB | 65k |
| `Laguna-XS-2.1-Q3_K_XL.gguf` | 3.6 GB | 65k |
| `Ornith-1.0-35B-UD-Q3_K_XL.gguf` | 4.5 GB | 65k |
| `Ornith-1.0-35B-UD-Q4_K_XL.gguf` | 4.9 GB | 65k |
| `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | 7.0 GB | **32k** |
| `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | 7.5 GB | 65k |
| `POCKET-26B-Q4_K_M.gguf` | 4.5 GB | 65k |
| `POCKET-35B-Q3_K_M.gguf` | 3.7 GB | 65k |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | 7.5 GB | 32k |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | 5.7 GB | 65k |
| `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | 7.6 GB | 65k |
| `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 8.0 GB | 131k |
| `Qwythos-9B-v2-MTP-Q4_K_M.gguf` | 7.9 GB | 131k |
| `Qwythos-9B-v2-Q4_K_M.gguf` | 7.7 GB | **32k** |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 4.1 GB | 65k |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 5.8 GB | 65k MTP |
| `nanbeige4.2-3b-Q4_K_M.gguf` | 6.9 GB | 32k |
