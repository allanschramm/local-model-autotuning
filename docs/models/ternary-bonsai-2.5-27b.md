# Ternary Bonsai 2.5 27B

- The validated text GGUF is `Ternary-Bonsai-2.5-27B-PTQ1_0.gguf` (5.99 GB, Apache 2.0). Its SHA-256 is `929560ec9d37cf3cadd4b00cc87b51c570082a628dd830b4692dc0b2ed18dbbe`. [Publisher artifact](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF/blob/main/Ternary-Bonsai-2.5-27B-PTQ1_0.gguf)
- GGUF architecture `qwen35` is dense and hybrid. Every fourth block uses full attention, for 16 full-attention and 48 recurrent blocks. The publisher lists a maximum context of 262,144 tokens. [Publisher model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF)
- Validation used the PrismML `prism-b12232-f702476` Windows x64 CUDA 13.3 prebuilt, with its matching runtime DLLs and the GGUF reader from the same tag for `PTQ1_0` tensor metadata. The publisher also offers `Q2_0` for upstream llama.cpp `b11556` or later. [PrismML release](https://github.com/PrismML-Eng/llama.cpp/releases/tag/prism-b12232-f702476) · [Upstream compatibility](https://huggingface.co/prism-ml/Ternary-Bonsai-2.5-27B-GGUF#which-llamacpp-runs-which-file)

## Runtime validation

On 2026-11-27, one bounded [`tuning-runtime-v1` Trial](../../autoresearch/TUNING.md) passed on an 8 GiB-class CUDA device. It used one slot, `N_GPU_LAYERS=99`, K and V caches in `q4_0`, Flash Attention on, batch 128, ubatch 64, eight inference and batch threads, and mmap. Speculative decoding was disabled. Sampling used temperature 1.0, top-p 0.95, top-k 20, min-p 0.0, repeat penalty 1.0 and presence penalty 0.0.

| Measurement | Result |
| --- | --- |
| Configured and observed slot context | 100,096 tokens |
| Fixed-text reference PPL | 7.3213 at 2,048 tokens |
| Generation throughput | 37.90 tokens/s median (38.10, 37.90, 37.89) |
| Occupancy per generation sample | 32 input tokens, 512 generated tokens, zero truncation |

The Trial confirms allocation and generation with short inputs. Prefill near 100K, long-context retrieval, and MTP performance are unmeasured as of the validation date.

## Memory preflight

The original Trial was rejected during preflight because the generic KV estimate charged every block for attention. PrismML's `llama-fit-params --fit-print on` projected 7,427 MiB for CUDA weights, context and buffers at 100,000 tokens with the same cache and batch settings. The projection is an estimate. The corrected preflight counts context-scaled KV on the periodic full-attention blocks, and the subsequent monitored Trial passed at 100,096 tokens. [Runtime attention layout](https://github.com/PrismML-Eng/llama.cpp/blob/prism-b12232-f702476/src/models/qwen35.cpp) · [Harness estimate](../../autoresearch/core/model_arch.py)

To reproduce, use a local Baseline with the settings above, pin `AUTORESEARCH_LLAMA_CPP_ROOT` to the validated release, and run `python -m autoresearch.tuning --plan <plan.json>` from the project venv. Use a plan with `minimum_context` 100000, `max_trials` 1, and `search_space` set to `{"BATCH_SIZE":[128]}`. The tuning command records the Trial and evidence paths in local `results.db`.
