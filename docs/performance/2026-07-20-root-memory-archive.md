# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-20-root-memory-archive.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Commit | Model | KV | Ctx | VRAM (GB) | TPS |
| --- | --- | --- | --- | --- | --- |
| `bab91e9` | Qwen3.5-9B-MTP | q4_0 | 32768 | 6.9 | 157.3 |
| `0c8e5fc` | gemma4-v2 | turbo4 | 65536 | 8.0 | 29.5 |
| `a06b9f9` | Ornith-1.0-9B | q4_0 | 131072 | 7.4 | 52.2 |
| `d5ddbd1` | Ornith-1.0-35B | q4_0 | 131072 | 7.7 | 31.5 |


* **Qwen3.5-9B-MTP**: `kv=q4_0`, `ctx=32768`, `threads=12`
* **gemma4-v2**: `kv=turbo4 K+V`, `ctx=65536`, `threads=8`, `temp=0.4`, `batch=128/ubatch=64`
* **Ornith-1.0-9B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4`
* **Ornith-1.0-35B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4`, `n-cpu-moe=32` (VITRIOL)


* **gemma4-v2 ctx below 65536 with Nexus**: 50K token padding breaks context. Nexus requires ctx at least 65536.
* **Low Throughput**: g4-opt-it with q4_0 at 16k → 24.3 TPS (below then-threshold 30.0).
* **VRAM Limits**: Models over 9B with f16 KV at 64k+ exceed 7.9 GB. Use turbo4.

* **Turbo4 Gemma4 12B dense**: VRAM flat at ~7.9GB up to 128K ctx thanks to aggressive TurboQuant. KV cache cost amortized by nvidia-smi 10Hz sampling.
* **MTP absent in Gemma4**: Model lacks `nextn_predict_layers`. Speculative decoding doesn't help.
* **Threads**: 8 threads better than 12 on Gemma4 12B (unregistered baseline issue with 12).
* **Batch sizing**: batch=128, ubatch=64 more stable on Gemma4 than 512/256 under short trial budget.
* **Indentation in reasoning models (Ornith/Qwen 3.5)**: `<think>` blocks generate 8+ space indented drafts (inside markdown lists). Truncated output → fallback parser carries extra indentation → IndentationError when concatenating signature.
* **VITRIOL MoE Streaming 35B**: Keeping routed experts in CPU/RAM via `--n-cpu-moe 40` enables 35B model loading on 8GB VRAM (8 GB-class discrete NVIDIA) with acceptable throughput (23.6 TPS) and peak VRAM only 4.1 GB.
* **UBATCH sweet spot**: 256 beats 512 on discrete 8 GB-class NVIDIA for Gemma4. llama-bench: ub=256 pp1922 tg49.8 vs ub=512 pp1940 tg41.0. Higher ubatch = better prompt processing but worse token generation.
* **Category column in results.tsv** (2026-07-01): Added to enable fair cross-model comparisons.
* **Docs path discipline** (2026-07-01): User-facing docs must not contain hardcoded paths (`/home/<user>/...`). Use `./llama.cpp/`, `$LLAMA_CPP`, or `AUTORESEARCH_LLAMA_CPP_ROOT`.

* `--n-cpu-moe` value matters: 32 works for 35B MoE; too high wastes RAM without benefit on dense models.
* `config.py` write helpers preserve section headers and comments — safe to call repeatedly.
* GEN_KWARGS filtered to non-None values only before passing to llama-server.
* Never run `llama-server` or `llama-bench` directly — harness manages paths, lifecycle, VRAM monitoring, result logging.
