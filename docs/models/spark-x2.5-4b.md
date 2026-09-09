# Spark-X2.5-4B

## Header Block

- **Source repo**: [`XHToken/Spark-X2.5-4B-GGUF`](https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF)
- **Base model**: [`XHToken/Spark-X2.5-4B`](https://huggingface.co/XHToken/Spark-X2.5-4B)
- **Upstream PR**: [ggml-org/llama.cpp#27868](https://github.com/ggml-org/llama.cpp/pull/27868) (landed in `b10828`)
- **License**: Apache 2.0
- **Local file path**: `models/XHToken/Spark-X2.5-4B/Spark-X2.5-4B-Q4_K_M.gguf`
- **Family**: Spark-X2.5 series (~4.1B parameters)
- **Quantization**: `Q4_K_M` (file size: 2,479.8 MiB)

---

## Architecture

*Verified directly from GGUF metadata via `gguf.GGUFReader` and `scripts/model_info.py`.*

- **GGUF Architecture**: `spark2_5`
- **Architecture Class**: Dense causal language model (no MoE layers)
- **Block Count (`spark2_5.block_count`)**: 36 transformer layers
- **Native Context Length (`spark2_5.context_length`)**: 1,048,576 tokens (1M native context)
- **Embedding Dimension (`spark2_5.embedding_length`)**: 2560
- **Feed-Forward Dimension (`spark2_5.feed_forward_length`)**: 10240
- **Attention Heads (`spark2_5.attention.head_count`)**: 16
- **Tensor Count**: 290
- **Attention Pattern**: Hybrid attention architecture with gated attention mechanisms
- **KV Cache Estimate (`f16` @ 64k)**: 1,179.0 MB

---

## Hardware Requirements

| Parameter | Value |
|---|---|
| Model size | ~4.1B parameters |
| Quantization size | 2.42 GiB (`Q4_K_M`) |
| Physical VRAM required | $\ge 4.0\text{ GB}$ (tight) / $\ge 6.0\text{ GB}$ (comfortable at 131k context) |
| Runtime Requirement | **`llama.cpp` $\ge$ `b10828`** (tested and validated on `upstream@b10867`). Older builds (e.g. `b10819`) fail with `unknown model architecture: 'spark2_5'`. |

---

## Recommended Settings

From model card and GGUF metadata:

| Setting | Value | Notes |
|---|---|---|
| `TEMP` | 0.8 / 1.0 | 0.8 general fallback; 1.0 publisher default |
| `TOP_P` | 0.95 | Publisher recommended |
| `TOP_K` | 40 | Publisher default is -1 (unbounded); 40 for bounded sampling |
| `MIN_P` | 0.05 | Universal fallback floor |
| `REPEAT_PENALTY` | 1.0 | Neutral |
| Thinking mode | `--think=false` / disabled | Publisher documents disabling thinking mode for direct agentic/coding response |

---

## MTP Section

- **MTP Tensors in this GGUF**: None (standard dense causal LM without extra prediction heads).
- **Embedded MTP Support**: Not present in `Spark-X2.5-4B-Q4_K_M.gguf`.
- **Harness Speculative Decoding**: `--spec-type none`.

---

## MoE Split

- **Architecture**: Dense (not MoE).
- **`N_GPU_LAYERS`**: 99 (all 36 layers offload to GPU).
- **`N_CPU_MOE`**: `None` (harness validates dense models must fit in VRAM).

---

## Our Config Baseline (Trial Validated — on_front)

Measured on discrete 8 GB-class NVIDIA GPU via `benchmark_search.py --agentic-full` on 2026-09-08:

- **Status**: **`on_front`** (Day #17 / Night #14 Pareto frontier)
- **Engine**: `upstream@b10867` (CUDA 13.3)
- **Context Size**: `131072` (tested at full 131k context)
- **KV Cache**: `q4_0`
- **Batch / Ubatch**: 512 / 128
- **Threads**: 8
- **Flash Attention**: `on`
- **Measured Throughput (`bench_tg` 512)**: **`71.3 t/s`** (combined TPS: **`83.5 t/s`**)
- **Peak VRAM**: **`6.1 GB`** (fits comfortably in 8 GB VRAM at 131k context)
- **Agentic Score (Claw-15 full)**: **`0.8667`** (13/15 passed in 1333s)
- **Coding Score (Coding-10)**: **`0.4400`**
  - **HumanEval+**: `0.7000` (7/10)
  - **MBPP+**: `0.5000` (5/10)
  - **LiveCodeBench**: `0.4000` (4/10)
  - **BigCode Hard**: `0.0000` (0/10)

---

## Sources / Verification

- **Primary HF Repo**: `https://huggingface.co/XHToken/Spark-X2.5-4B-GGUF` (retrieved 2026-09-08)
- **Upstream PR**: `https://github.com/ggml-org/llama.cpp/pull/27868` (merged in `b10828`, 2026-09-07)
- **GGUF Inspection**: Verified via `gguf.GGUFReader` on local weights `models/XHToken/Spark-X2.5-4B/Spark-X2.5-4B-Q4_K_M.gguf` (2026-09-08)
- **Validation Run**: Executed via `benchmark_search.py --validation` logged to canonical SQLite `results.db` (2026-09-08)
- **Full Trial Run**: Executed via `benchmark_search.py --agentic-full` logged to canonical SQLite `results.db` (2026-09-08)

---

## Open Questions

- **Speculative Draft Pairing**: No native draft model currently exists for `spark2_5`. Evaluation of ngram-cache or prompt-lookup on this architecture remains unmeasured.
