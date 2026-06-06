# Agent Experiment Memory

## 1. Experiment Log (TSV Summary)

| Commit | Model | Config (KV, Ctx, Threads, MTP) | VRAM (GB) | TPS | Score | Status | Notes |
|---|---|---|---|---|---|---|---|
| `127ea10` | 9B-MTP | default, ctx=128k | 7.8 | - | 0.430657 | **Keep** | Baseline 9B-MTP 128k |
| `027bcc3` | Qwopus-9B | default, ctx=128k | 7.7 | - | 0.374961 | **Keep** | Baseline Qwopus-9B |
| `027bcc3` | Qwen-2B | default, ctx=128k | 3.5 | - | 0.317142 | **Keep** | Baseline Qwen-2B |
| `429179` | Qwen-2B | f16 KV cache | 3.8 | - | 0.355324 | **Keep** | Improved from 0.317 |
| `430933` | Gemma-4B | f16 KV cache | 6.1 | - | 0.366415 | **Keep** | Improved from 0.343 |
| `a8461ef` | Qwen3.5-9B-IQ4_NL | turbo3, ctx=65536, threads=12 | 7.8 | 159.1 | 0.340196 | **Keep** | retrieval=0.5914, agency=0.1727 |
| `33cc25f` | g4-opt-it | kv=q4_0, ctx=16384, threads=12 | 7.7 | 24.3 | 0.000000 | **Discard** | Combined TPS (24.3) fell below 30.0 threshold. |
| `33cc25f` | Qwen3.5-9B-MTP | kv=q4_0, ctx=65536, threads=12 | 7.0 | 43.2 | 0.188909 | **Discard** | MTP test, no improvement over 9B baseline. |

## 2. Blocked Configurations (Fails / Low TPS)

*   **Low Throughput Bottleneck**: `g4-opt-it` with standard `q4_0` KV cache at 16k context gets only **24.3 TPS** (below the 30.0 TPS threshold), zeroing the score. Requires `turbo` quant formats or higher thread parallelism to meet the throughput target.
*   **VRAM Limits**: 9B models with `f16` KV cache at large context (64k+) will exceed the 7.9 GB safety budget. Lower quantization (like `turbo3` or `q4_0`) is strictly required.

## 3. Working Hypotheses

*   **TurboQuant Efficiency**: `turbo3` KV cache quant allows 9B models to execute at 65k context within 7.8 GB of VRAM while sustaining very high throughput (159.1 TPS).
*   **Thread Configuration**: 12 threads provides an optimal processing rate without causing thread scheduling contention on the host CPU.
