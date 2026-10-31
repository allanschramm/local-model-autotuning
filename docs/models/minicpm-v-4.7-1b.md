# MiniCPM-V-4.7-1B

## Header Block
- Source: [openbmb/MiniCPM-V-4.7-1B-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf); Apache-2.0.
- File: `models/openbmb/MiniCPM-V-4.7-1B-gguf/MiniCPM-V-4_7-1B-Q4_K_M.gguf`; Q4_K_M, 640.9 MiB. Text benchmarks do not use the separate vision projector.

## Architecture
Local GGUF: name `Hf_V47`, dense `qwen35`, block_count 25, embedding 1024, attention heads 8 / KV heads 2, native context 262144. Declares nextn_predict_layers=1 but contains no final-layer/MTP tensors: the file contains the 24-block trunk.

## Hardware Requirements
Measured at context 65536, q4_0 KV, full GPU load on upstream b11577. Full Trial peak VRAM 2.1 GB. b11149 failed loading `blk.24.attn_norm.weight`; b11577 loads this trunk-only export.

## Recommended Settings
No dedicated text sampler recommendation in the retrieved publisher GGUF card. Use the universal fallback: TEMP 0.8, TOP_P 0.95, TOP_K 40, MIN_P 0.05, repeat 1, presence 0, frequency unset. Speculation disabled.

## MTP Section
Metadata alone does not prove embedded MTP: there are no MTP tensors in this file. Do not enable draft-mtp from nextn_predict_layers=1. Upstream [#30097](https://github.com/ggml-org/llama.cpp/pull/30097) adds trunk-only qwen35 loading; b11577 includes it. No harness flag addition needed.

## MoE Split
Dense: N_CPU_MOE=None, N_GPU_LAYERS=99.

## Our Config Baseline
2026-10-12 SQLite Trial: upstream b11577, context 65536, q4_0 KV, batch 512 / ubatch 128, threads 8, flash attention on, mmap, fallback sampler, speculation disabled. Claw-15 + coding-10: ; generation TPS 275.2, bench_tg 248.8, status on_front.

Throughput measurements during overlapping launches are not comparable to isolated Trials. These benchmarks evaluate text only.

## Sources / Verification
- [Publisher GGUF card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf), hf retrieval 2026-10-12.
- [Upstream trunk-only loading change](https://github.com/ggml-org/llama.cpp/pull/30097), checked 2026-10-12.
- Local GGUF tensor inventory and results.db, checked 2026-10-12.

## Open Questions
- Vision projector and image/video quality have not been evaluated; they are outside the text Trial.
