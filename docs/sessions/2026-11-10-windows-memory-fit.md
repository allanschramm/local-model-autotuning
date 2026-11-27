# Windows memory-fit Trials, November 10–12, 2026

The local SQLite and Trial artifacts record these runs on Windows with an 8 GiB-class CUDA device. All candidates requested 65,536 tokens, K/V `q4_0`, Flash Attention, eight inference and batch threads, and no speculative draft. Successful Trials used upstream `b11577` and confirmed the actual slot allocation. TPS is the median of three 512-token generations under [tuning-runtime-v1](../../autoresearch/TUNING.md). PPL is the fixed-text check.

## TwIL-8B

The GGUF was `TwIL-8B-Q4_K_M-balanced.gguf`. The sampler used temperature 1.0, top-p 0.95, top-k 40, min-p 0.05, repeat penalty 1.0 and presence penalty 0. Fully GPU-resident weights with batch/ubatch 512/128 hit the Windows shared-memory watchdog twice and later passed. A partial placement diagnostic with 16 GPU layers stopped on a Windows invalid-argument error. The bounded Search kept all layers on the GPU.

| Stage | Batch/ubatch | TPS | PPL |
| --- | ---: | ---: | ---: |
| Standalone validation | 512/128 | 28.84 | 12.5026 |
| Search Baseline | 512/128 | 25.18 | 12.5026 |
| Search, smaller batch | 256/128 | 29.67 | 12.5026 |
| Search, smaller ubatch | 512/64 | 27.34 | 12.1222 |

All Search Trials were admitted. With context held constant, 256/128 had the highest measured throughput. The spread between nominally identical 512/128 runs makes a repeat important before attributing a gain to batch size alone.

## Nemotron 3.5 Lightning 30B A3B

The GGUF was `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-IQ4_NL.gguf`. The sampler matched TwIL except for top-k 0 and min-p 0.

| CPU expert blocks | Batch/ubatch | Outcome | TPS | PPL |
| ---: | ---: | --- | ---: | ---: |
| Unset | 512/128 | RAM preflight refused the launch | — | — |
| 42 | 512/128 | RAM circuit breaker stopped the Trial | — | — |
| 42 | 256/64 | PPL completed, then RAM preflight refused the generator launch | — | 5.6731 |
| 40 | 256/64 | Admitted | 22.09 | 5.6731 |

Two fewer CPU expert blocks let the last attempt pass at the same context. Repeat the RAM checks at each process launch, including after perplexity. A successful PPL stage establishes only that stage's fit.

To locate the evidence in local `results.db`, query `tuning_trials` for `run_id` prefixes `74fe083d`, `279de582`, `79581863`, `8da611d7`, `49dad565`, `108e949d`, `6f6db9d2`, `7e0aa696` and `f10fff64`. The associated artifacts contain exact settings, diagnostics and samples. Long-prompt generation and retrieval were outside this test.
