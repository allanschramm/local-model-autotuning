# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/best-model-8gb-vram.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model | GGUF Q4_K_M | Arch (verified) | Context |
| :--- | :---: | :--- | :---: |
| **Gemma 4 E4B** | **5.34 GB** | dense, 4.5B effective / 8B w/ embeddings (PLE), sliding-window + global attention, text+image+audio | 128K |
| **LFM2.5-8B-A1B** | **5.16 GB** | hybrid MoE, 8.3B total / **1.5B active** (18 LIV conv + 6 GQA) | 128,000 |
| **Qwen3-8B** | **5.03 GB** | dense 8.2B, GQA 32Q/8KV, thinking + non-thinking | 32,768 native / 131,072 YaRN |
| **Qwen3.5-4B** | **2.74 GB** | dense hybrid: Gated DeltaNet + sparse attention (8/32 full-attn layers), 4B LM + vision encoder, MTP | 262,144 native |


| Model | File @ Q4_K_M | KV headroom note |
| :--- | :---: | :--- |
| LFM2.5-8B-A1B | 5.16 GB | 6 GQA layers of 24 ⇒ good headroom; full 128K needs reduced ctx or quantized KV |
| Qwen3-8B | 5.03 GB | 32K f16 KV (~4.7 GB) ⇒ total ≈ 9.7 GB **exceeds 8 GB; use Q8_0 KV or drop ctx |
| Qwen3.5-4B | 2.74 GB | ~5 GB left for KV; 8/32 full-attn layers ⇒ 32K–128K fits comfortably |


| Model | Reason | Source-verified size |
| :--- | :--- | :--- |
| Gemma 3 12B | 7.3 GB @ Q4_K_M + any KV > 8 GB | bartowski/LM Studio card |
| Mistral Small 3.1/3.x (24B) | fits "a single RTX 4090 or 32 GB MacBook once quantized" — not an 8 GB card | Mistral card |
| Qwen3-14B | **dense** (14.77B), Q4_K_M = **9.0 GB** — exceeds 8 GB; only sub-8 GB is UD-IQ1/Q2. No "14B-A3B" variant exists (verified via `hf` CLI: card + quant list) | Unsloth file list |
| Qwen3-30B-A3B | 18.6 GB @ Q4_K_M; even full expert offload needs ~19 GB+ system RAM, PCIe-bound | Unsloth |
| gpt-oss-20b | 11.6 GB @ Q4_K_M; OpenAI states native MXFP4 "runs within 16 GB of memory" | OpenAI card + Unsloth |


| URL | Used for |
| :--- | :--- |
| https://huggingface.co/Qwen/Qwen3-8B | Qwen3-8B params, context, thinking modes |
| https://huggingface.co/unsloth/Qwen3-8B-GGUF | Q4_K_M = 5.03 GB |
| https://huggingface.co/Qwen/Qwen3-30B-A3B | MoE card: 30.5B/3.3B, 128 experts, ctx |
| https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF | Q4_K_M = 18.6 GB (too large) |
| https://huggingface.co/Qwen/Qwen3.5-4B | Qwen3.5-4B arch, ctx, all benchmark tables |
| https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | Q4_K_M = 2.74 GB |
| https://huggingface.co/google/gemma-3-4b-it |  |
| https://huggingface.co/google/gemma-3-12b-it |  |
| https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF | Q4_K_M = 2.49 GB |
| https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF | Q4_K_M = 7.3 GB (fit fail) |
| https://huggingface.co/google/gemma-3n-E4B-it |  |
| https://huggingface.co/lmstudio-community/gemma-3n-E4B-it-text-GGUF | Q4_K_M = 4.24 GB |
| https://huggingface.co/google/gemma-4-E4B-it |  |
| https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF | Q4_K_M = 5.34 GB |
| https://huggingface.co/LiquidAI/LFM2.5-8B-A1B |  |
| https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF | official Q4_K_M = 5.16 GB |
| https://huggingface.co/openai/gpt-oss-20b | 21B/3.6B, MXFP4, 16 GB memory, eval results |
| https://huggingface.co/unsloth/gpt-oss-20b-GGUF | Q4_K_M = 11.6 GB |
| https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B |  |
| https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF | Q4_K_M = 4.68 GB |
| https://huggingface.co/zai-org/GLM-4-9B-0414 | Z.ai card (32B-series tables; 9B claim only) |
| https://huggingface.co/microsoft/Phi-4-mini-instruct | 3.8B, 128K, benchmark table, test hardware |
| https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503 |  |
| https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | 8B, 128K, benchmark tables |
| https://qwenlm.github.io/blog/qwen3/ | Qwen3 lineup: context, layers, experts, languages |
| https://arxiv.org/abs/2505.09388 |  |
| https://arxiv.org/abs/2508.10925 | gpt-oss-120b & 20b Model Card |
| https://arxiv.org/abs/2511.23404 | LFM2 Technical Report (referenced by Liquid card) |
| https://arxiv.org/abs/2607.02770 | Gemma 4 Technical Report (referenced by Gemma 4 card) |
| https://huggingface.co/Qwen/Qwen3-14B | dense 14.77B; no "A3B" variant exists (verified via `hf` CLI) |
| https://huggingface.co/unsloth/Qwen3-14B-GGUF | file list: Q4_K_M = 9.0 GB (fit fail); only UD-IQ1/Q2 fit |
| https://huggingface.co/ibm-granite/granite-4.0-h-tiny | Granite 4.0 H Tiny arch/ctx/license (primary) |
| https://huggingface.co/blog/nvidia/nemotron-3-nano-4b | 18 tokens/s on Jetson Orin Nano 8GB (8 GB-memory device, not 8 GB-class discrete NVIDIA) |
