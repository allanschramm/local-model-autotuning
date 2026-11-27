# Inkling 2 Mini expert placement, November 23, 2026

`inkling2_loop` is a looped MoE with 24 blocks. Blocks 6–17 run three times per token, so where the experts reside can change throughput even at a fixed GPU block count. Upstream `b12651` includes architecture support and a fix for copying host weights shared by multiple graph nodes. The Unsloth Windows CUDA prebuilt adds model-specific optimizations. [Upstream support](https://github.com/ggml-org/llama.cpp/pull/32694) · [Host copy fix](https://github.com/ggml-org/llama.cpp/pull/32807) · [Unsloth runtime](https://github.com/unslothai/llama.cpp/releases/tag/b12649-mix-c7059d6)

All Trials targeted 65,536 tokens on Windows with an 8 GiB-class CUDA device. They used K/V `q4_0`, Flash Attention, eight inference and batch threads, batch 512, no draft, and frozen sampling (temperature 1.0, top-p 1, top-k 0, min-p 0, repeat penalty 1.0, presence penalty 0). Successful rows confirmed 65,536 tokens per slot. TPS is the median of three 512-token generations from 32-token prompts, with the fixed 2,048-token PPL check. [Tuning protocol](../../autoresearch/TUNING.md)

| GGUF quant | Runtime | GPU expert blocks | CPU expert blocks | Ubatch | Median TPS | PPL | SQLite run prefix |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `UD-IQ3_XXS` | `b12651` | 20–23 | 20 | 64 | Watchdog | 4.8506 | `3713d01e` |
| `UD-IQ3_XXS` | `b12651` | None | 24 | 16 | 14.83 | 4.8784 | `21b27904` |
| `UD-Q4_K_M` | `b12651` | 20–23 | 20 | 16 | 14.87 | 4.8969 | `e3fdda09` |
| `UD-Q4_K_M` | `b12649-mix-c7059d6` | 20–23 | 20 | 16 | 16.02 | 4.8969 | `f96ba54c` |
| `UD-Q4_K_M` | `b12649-mix-c7059d6` | 6–9 | 20 | 16 | 17.78 | 4.8821 | `a7b3b13a` |
| `UD-Q4_K_M` | `b12649-mix-c7059d6` | 6–11 | 18 | 16 | 19.58 | 4.8490 | `6a19fa2f` |
| `UD-Q4_K_M` | `b12649-mix-c7059d6` | 6–12 | 17 | 16 | 20.59 | 4.8762 | `abdb02ba` |

The model files were `Inkling-2-Mini-UD-IQ3_XXS.gguf` (11.66 GB) and `Inkling-2-Mini-UD-Q4_K_M.gguf` (16.61 GB). The Q4 fit with host-resident experts despite exceeding dedicated GPU capacity on disk. The failed IQ3 attempt exceeded the shared-memory watchdog limit. The runtime-only Q4 swap gained 7.7%, the four-block placement swap gained another 11%, and adding three looped blocks brought the total gain to 38.4% over the original Q4 control. [Publisher GGUFs](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)

Pass explicit `--override-tensor` rules before `--n-cpu-moe`; the first applicable rule wins. Audit the loaded tensor map, as `N_CPU_MOE` alone does not identify the blocks. The seven-block `model-up` check audited 88 expert tensors and completed a 1,163-token prefill check. Its observed peak was about 7.50 GiB of dedicated VRAM and 78 MiB shared. Eight GPU blocks, deep prompt fill, vision and tool calling remained untested.

Reproduce the control and placement changes with one bounded [tuning plan](../../autoresearch/TUNING.md) per configuration. Query the listed prefixes in local `results.db`; the Trial artifacts contain the exact placement and samples.
