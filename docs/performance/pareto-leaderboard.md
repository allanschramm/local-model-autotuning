# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/pareto-leaderboard.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model | ctx | TPS |
| ------------------------------------------------- | ------ | ------- |
| Bonsai-27B-Q1_0.gguf | 65k | 40.3 |
| K2-Horizon-0.9B-Q4_K_M.gguf | 65k | 243.3 |
| K2-Horizon-3.7B-Q4_K_M.gguf | 65k | 74.2 |
| K2-Horizon-7B-Q4_K_M.gguf | 65k | 46.0 |
| Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf | 65k | 31.3 |
| LFM2.5-1.2B-Instruct-Q8_0.gguf | 65k | 156.4 |
| LFM2.5-2.6B-Q8_0.gguf | 65k | 78.7 |
| LFM2.5-8B-A1B-Q4_K_M.gguf | 65k | 182.2 |
| Laguna-XS-2.1-Q3_K_XL.gguf | 65k | 37.2 |
| Ling-3.0-tiny-Q4_K_M.gguf | 65k | 53.5 |
| MindSparQ-Coder-1.5B.Q4_K_M.gguf | 65k | 182.3 |
| MiniCPM5-2B-Q4_K_M.gguf | 131k | 133.3 |
| NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | 65k | 78.7 |
| Nanbeige4.2-3B-Q4_K_M.gguf | 32k | 54.4 |
| Nanbeige4.2-3B.Q4_K_M.gguf | 32k | 53.8 |
| Ornith-1.0-35B-UD-Q3_K_XL.gguf | 65k | 26.0 |
| Ornith-1.0-9B-UD-Q4_K_XL.gguf | 65k | 48.6 |
| Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf | 65k | 34.9 |
| Ornith-1.5-35B-Q4_K_M.gguf | 65k | 28.8 |
| Ornith-1.5-9B-Q4_K_M.gguf | 65k | 43.2 |
| POCKET-26B-Q4_K_M.gguf | 65k | 21.8 |
| POCKET-35B-Q3_K_M.gguf | 65k | 35.7 |
| Qwen3.5-4B-MTP-Q4_K_M.gguf | 65k | 84.5 |
| Qwen3.5-4B-Q4_K_M.gguf | 131k | 71.9 |
| Qwen3.8-2B-BF16.gguf | 32k | 60.5 |
| Qwen3.8-4B-Q4_K_M.gguf | 131k | 74.9 |
| Qwen3.8-4B-Q8_0.gguf | 32k | 48.9 |
| Qwen3.8-9B-Q4_K_M.gguf | 8k | 44.2 |
| Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf | 65k | 48.7 |
| Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf | 65k | 45.6 |
| SmolLM3-3B-Q4_K_M.gguf | 131k | 110.0 |
| Spark-X2.5-4B-Q4_K_M.gguf | 131k | 73.1 |
| Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf | 65k | 28.2 |
| granite-4.0-h-tiny-Q4_K_M.gguf | 65k | 50.8 |
| granite-4.1-3b-Q4_K_M.gguf | 65k | 86.6 |
| grug-35b-v2-Q4_K_M.gguf | 16k | 23.7 |
| model-Q4_K_M.gguf | 131k | 74.9 |


| Model | ctx | TPS |
| ------------------------------------------------- | ------ | ------- |
| Bonsai-27B-Q1_0.gguf | 65k | 40.3 |
| K2-Horizon-0.9B-Q4_K_M.gguf | 65k | 243.3 |
| K2-Horizon-3.7B-Q4_K_M.gguf | 65k | 74.2 |
| K2-Horizon-7B-Q4_K_M.gguf | 65k | 46.0 |
| Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf | 65k | 31.3 |
| LFM2.5-1.2B-Instruct-Q8_0.gguf | 65k | 156.4 |
| LFM2.5-2.6B-Q8_0.gguf | 65k | 78.7 |
| LFM2.5-8B-A1B-Q4_K_M.gguf | 65k | 182.2 |
| Laguna-XS-2.1-Q3_K_XL.gguf | 65k | 37.2 |
| Ling-3.0-tiny-Q4_K_M.gguf | 65k | 53.5 |
| MindSparQ-Coder-1.5B.Q4_K_M.gguf | 65k | 182.3 |
| MiniCPM5-2B-Q4_K_M.gguf | 131k | 133.3 |
| NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf | 65k | 78.7 |
| Nanbeige4.2-3B-Q4_K_M.gguf | 32k | 54.4 |
| Nanbeige4.2-3B.Q4_K_M.gguf | 32k | 53.8 |
| Ornith-1.0-35B-UD-Q3_K_XL.gguf | 65k | 26.0 |
| Ornith-1.0-9B-UD-Q4_K_XL.gguf | 65k | 48.6 |
| Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf | 65k | 34.9 |
| Ornith-1.5-35B-Q4_K_M.gguf | 65k | 28.8 |
| Ornith-1.5-9B-Q4_K_M.gguf | 65k | 43.2 |
| POCKET-26B-Q4_K_M.gguf | 65k | 21.8 |
| POCKET-35B-Q3_K_M.gguf | 65k | 35.7 |
| Qwen3.5-4B-MTP-Q4_K_M.gguf | 65k | 84.5 |
| Qwen3.5-4B-Q4_K_M.gguf | 131k | 71.9 |
| Qwen3.8-2B-BF16.gguf | 32k | 60.5 |
| Qwen3.8-4B-Q4_K_M.gguf | 131k | 74.9 |
| Qwen3.8-4B-Q8_0.gguf | 32k | 48.9 |
| Qwen3.8-9B-Q4_K_M.gguf | 8k | 44.2 |
| Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf | 65k | 48.7 |
| Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf | 65k | 45.6 |
| SmolLM3-3B-Q4_K_M.gguf | 131k | 110.0 |
| Spark-X2.5-4B-Q4_K_M.gguf | 131k | 73.1 |
| Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf | 65k | 28.2 |
| granite-4.0-h-tiny-Q4_K_M.gguf | 65k | 50.8 |
| granite-4.1-3b-Q4_K_M.gguf | 65k | 86.6 |
| grug-35b-v2-Q4_K_M.gguf | 16k | 23.7 |
| model-Q4_K_M.gguf | 131k | 74.9 |


| Model | Gap |
| :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | coding |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | coding-10 |
| `Qwythos-9B-v2*` failed trials | — |
| `cesium2-v7-q8_0.gguf` | coding |
| `deepgrove/maple-preview-TQ1_0-head-Q4_K.gguf` | load |
| `nanbeige4.2-3b-Q4_K_M.gguf` | coding-10 |
| `ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf` | load |
| split-config basenames | single-run |


| Job | Prefer |
| :--- | :--- |
| Night / long agent loops | **Qwen3.8-4B-Q4_K_M** (Tiel-Coder ties its quality but runs at 28 t/s; Ornith-1.5-9B close third) |
| Day / supervised | **Qwen3.8-4B-Q4_K_M** (74.9 t/s at top quality; Spark-X2.5-4B close — iq 0.44 @ 73.1 t/s; K2-Horizon-0.9B only if quality 0.44 suffices) |
| Historical run runtime | Qwen3.8-4B ( ties KAT/Tiel at 2.5× their TPS) → POCKET-35B |
| Low-VRAM co-run / fallback | **Ling-3.0-tiny** —  at 2.5 GB peak |
| SSD cleanup | delete weak/`rejected` first (`Cesium`, `MindSparQ`, corrupt `Ornith-FC` Q2_K, unloadable `maple` TQ1_0) |
