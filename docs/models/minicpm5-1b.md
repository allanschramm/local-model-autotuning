# MiniCPM5-1B

Runtime scope: quality results remain in the [archived full card](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/models/minicpm5-1b.md). Historical performance retains its original protocol.

## Header Block
- Source: [openbmb/MiniCPM5-1B-GGUF](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF); Apache-2.0.
- File: `models/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q4_K_M.gguf`; Q4_K_M, 656.2 MiB. No symlink required.

## Architecture
Verified from the local GGUF: `llama`, dense, 24 blocks, embedding 1536, attention heads 16 / KV heads 2, native context 131072. No expert tensors or MTP tensors.

## Hardware Requirements
Full GPU load was measured at context 65536 with q4_0 KV on upstream b11577; peak VRAM 2.2 GB. This is a measured configuration, not a universal minimum.

## Recommended Settings
Publisher: thinking `TEMP=0.9`, no-thinking `TEMP=0.7`; both `TOP_P=0.95`. Use `TOP_K=0`, `MIN_P=0`, `REPEAT_PENALTY=1`, `PRESENCE_PENALTY=0` as neutral unspecified controls. Keep reasoning/template mode aligned with the selected sampler.

## MTP Section
No embedded MTP. DSpark for MiniCPM5-2B is not a companion for this model. Speculation disabled in the recorded Trials; no new upstream flag mapping required.

## MoE Split
Dense: `N_CPU_MOE=None`, `N_GPU_LAYERS=99`.

## Our Config Baseline
2026-10-12, SQLite `results.db`: upstream b11577, context 65536, q4_0 KV, batch 512 / ubatch 128, threads 8, flash attention on, mmap, speculation disabled. Complete Claw-15 + coding-10: ; generation TPS 259.1, bench_tg 271.1, status on_front.


## Sources / Verification
- [Publisher GGUF card](https://huggingface.co/openbmb/MiniCPM5-1B-GGUF), retrieved with hf on 2026-10-12.
- Local GGUF metadata and canonical SQLite Trials, checked 2026-10-12.

## Open Questions
- None for the measured text configuration. No vision evaluation applies to this text model.
