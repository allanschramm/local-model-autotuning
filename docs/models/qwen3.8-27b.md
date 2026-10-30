# Qwen3.8-27B — Model Card

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/qwen3.8-27b.md). Historical performance retains its original protocol.

**Source repo:** https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
**Base model:** Qwen/Qwen3.8-27B
**License:** Apache-2.0
**Local file:** `models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf` (5.8 GB; 6192222208 bytes, UD-IQ1_S)
**Family:** Qwen 3.5 (qwen35 architecture)
**Quantization:** Unsloth Dynamic IQ1_S (UD-IQ1_S)

## Architecture (from GGUF via harness)

- **Dense** (`block_count` = **64**, `expert_count` = 1) — harness `is_moe_model` reads dense ⇒ no `N_CPU_MOE`.
- Hybrid attention: Gated DeltaNet layers with sparse full-attention interval (Qwen3.5 pattern, 4/64 full-attn per GGUF) ⇒ ~1/16 dense KV footprint at equal ctx.
- Native context class 262K-equivalent; runs 262144 configured on this rig.
- `qwen35.attention.head_count` = 24, `qwen35.attention.head_count_kv` = 4
- `qwen35.attention.key_length` = 256, `qwen35.attention.value_length` = 256
- `qwen35.embedding_length` = 5120, `qwen35.feed_forward_length` = 17408
- `qwen35.rope.dimension_count` = 64, `qwen35.rope.freq_base` = 10000000.0
- `qwen35.full_attention_interval` = 4
- `qwen35.nextn_predict_layers` = 0 (no MTP tensors)
- SSM (Gated DeltaNet): `qwen35.ssm.conv_kernel` = 4, `qwen35.ssm.group_count` = 16, `qwen35.ssm.inner_size` = 6144, `qwen35.ssm.state_size` = 128, `qwen35.ssm.time_step_rank` = 48

## Hardware requirements (publisher table)

- Publisher tier: UD-IQ1_S → 4–6 GB GPUs; fits entirely (`N_GPU_LAYERS 99`).
- Measured @65536 context (from results.db): prefill ~28.4 t/s (tps) with VRAM headroom; fits 8 GB-class VRAM.
- No OOM observed in validation runs; no requirement for CPU offload.

## Recommended settings (publisher card)

- **Thinking mode (reasoning enabled):** `TEMP 1.0 · TOP_P 0.95 · TOP_K 20 · MIN_P 0.0`
- **Instruct/non-thinking mode:** `TEMP 0.7 · TOP_P 0.80 · TOP_K 20 · MIN_P 0.0`
- Chat template embedded; requires `JINJA=True`.
- **Reasoning control (verified 2026-08-29 from embedded `tokenizer.chat_template`):** the GGUF template reads `reasoning_effort` and `enable_thinking` (both present). Ladder: `xhigh` (default) → `medium` → `low`; `'high'` aliases to `xhigh`; any other value raises an exception in Jinja. Default render injects the `xhigh` instruction into every prompt. Working levers: `--reasoning on/off` (Baseline REASONING), `--reasoning-budget N` (REASONING_BUDGET, server-side template-independent), and REASONING_PRESERVE (where `/props` reports `supports_preserve_reasoning` — verified true for this GGUF). Common misconception guard: do not attempt `--reasoning-effort medium/low` expecting a ladder change without confirming the template; use `--reasoning-budget` for fine-grained control.

## MTP

- **No MTP tensors in this GGUF** (`qwen35.nextn_predict_layers` = 0). Separate draft models exist in the HF repo under `MTP/` directory (e.g., `mtp-Qwen3.8-27B-Q4_0.gguf`) but are not embedded in this UD-IQ1_S file. Do not add spec decoding without an explicit A/B confirming benefit on this architecture.

## MoE split ("VITRIOL split")

- N/A — dense model, all layers on GPU. No expert tensors detected in GGUF metadata.

## Our config baseline

- **TBD:** pending first full Trial (Claw-Eval full + coding-10). Baseline will be populated after successful validation and ENGINE_DEFAULTS/SAMPLER_DEFAULTS extraction from the model card.

## Sources / Verification

- HF card `unsloth/Qwen3.8-27B-GGUF` — file size, quantization info, license, imatrix, conversational tags. Extracted 2026-08-29.
- HF README `unsloth/Qwen3.8-27B-GGUF` — hardware guidance, usage notes, dynamic v3.0 details. Extracted 2026-08-29.
- Base model card `Qwen/Qwen3.8-27B` — architecture details, license Apache-2.0, image-text-to-text pipeline tag, benchmark tables for text and VL performance. Extracted 2026-08-29.
- Base model README `Qwen/Qwen3.8-27B` — highlights, model overview, benchmark results, quickstart. Extracted 2026-08-29.
- Local GGUF dump: `./venv/Scripts/python.exe llama.cpp/gguf-py/gguf/scripts/gguf_dump.py --no-tensors --json models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ1_S.gguf` (2026-08-29).

## Open questions

- **TBD:** Capability eval (agentic, coding) missing — results.db currently holds only throughput rows (no capability numbers). Need full Trial to fill claw-full and coding-10 axes.
- **TBD:** Practical context length ceiling with YaRN on b10549 — verify from effective server log before raising `CTX_SIZE` beyond 262144.
