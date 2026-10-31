# MiniCPM-V-4.7-35B-A3B

## Header Block
- Source: [openbmb/MiniCPM-V-4.7-35B-A3B-gguf](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf); Apache-2.0.
- File: `models/openbmb/MiniCPM-V-4.7-35B-A3B-gguf/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf`; Q4_K_M, 20185.7 MiB. No symlink required. Vision projector is separate and unused by text Trials.

## Architecture
Verified from local GGUF: name `35a3_hf_v47_0923`, hybrid MoE `qwen35moe`, block_count 41, embedding 2048, attention heads 16 / KV heads 2, native context 262144, 256 experts with 8 active. Declares nextn_predict_layers=1 but has no MTP or block-40 tensors: this is the 40-block trunk export.

## Hardware Requirements
Validation at context 65536, q4_0 KV, upstream b11577: 31.3 bench_tg, peak VRAM 3.4 GB with experts on CPU. Entire 21.17 GB GGUF and host workspace still require host RAM; passing the hardware gates is required before launch. This is measured offload behavior, not full GPU fit.

## Recommended Settings
Retrieved publisher GGUF card supplies no dedicated text sampler. Universal fallback: TEMP 0.8, TOP_P 0.95, TOP_K 40, MIN_P 0.05, repeat 1, presence 0, frequency unset. SPEC_TYPE=None / SPEC_DRAFT_N_MAX=0.

## MTP Section
No embedded MTP tensors despite metadata declaring a nextn layer. Do not infer draft-mtp availability from the header. Upstream [#30097](https://github.com/ggml-org/llama.cpp/pull/30097) supports trunk-only qwen35moe exports; b11577 loaded this file. Publisher DSpark companions are separate drafts, not standalone benchmark targets. No new harness flag mapping required.

### Exact DSpark Companion

The matching draft is [openbmb/MiniCPM-V-4.7-35B-A3B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark), revision `5facb75c251ffd524524a544c1f5fd97ba0d6c5c`. It is not the MiniCPM5-2B draft. Publisher config: Qwen3DSparkModel, 5 draft layers, block_size 15, target hidden-state layers [3, 11, 19, 27, 35], BF16.

Converted locally to `MiniCPM-V-4.7-35B-A3B-DSpark-BF16.gguf` with the read-only upstream converter at `11fe02151f79c41d0d4af7da708755d73b9c0da6`: architecture dflash, 62 tensors, approximately 779 MB, including Markov/confidence heads. GGUF tokens and token types match the target exactly. Exported extraction indices are [4, 12, 20, 28, 36] (converter's one-based graph-layer convention).

Conversion uses `--outtype bf16` and `--target-model-dir` containing the target tokenizer. This converter does not register the MiniCPMV4_7 wrapper; the temporary tokenizer directory's config is therefore the unchanged target `text_config`, with its matching registered architecture `Qwen3_5MoeForCausalLM`. Preserve the original wrapper config separately. No checkpoint tensors or vendor code are edited.

Candidate engine/sampler Baseline differs from the no-spec Trial only in SPEC_TYPE='draft-dspark', SPEC_DRAFT_N_MAX, and SPEC_DRAFT_MODEL pointing to this GGUF. Context remains 65536; sampler/KV/offload/batch/thread settings stay fixed.

2026-10-13: max15 loaded but was rejected at the throughput gate (bench_tg 17.9 < TPS_FLOOR 20). Max7 passed full-context validation: 5/5, bench_tg 30.4, peak VRAM 5.0 GB. That task statistic is not a suite-wide acceptance rate. The 30.4 bench result is below the historical no-spec 32.7; no speed improvement is established.

The server connection was forcibly closed during T006; T008 onward returned connection refused. The agentic artifact records 2/15 and 0.1333, but 13 tasks contain HTTP errors: **agentic is invalid as a quality measurement**. The store nevertheless labels this row OK/on_front; that mechanical status must not justify a launcher or quality claim. No OOM diagnostic was found, and peak VRAM alone does not identify the crash cause.

Evidence: canonical description `trial MiniCPM-V-4.7-35B-A3B exact DSpark BF16 max7 b11577 controlled`, artifact `autoresearch/runners/logs/agentic-20261013-004051-MiniCPM-V-4_7-35B-A3B-Q4_K_M.json`. Runtime server log ends without an explicit OOM report. The measured DSpark throughput is below the historical no-spec result (36.4 vs 37.5 generation TPS; 27.2 vs 32.7 bench_tg). The no-spec Baseline/fingerprint/launcher was restored; no speed improvement or complete quality vector was established for DSpark.

This Trial used MPLBACKEND=Agg; the historical no-spec Trial did not pin it. The operator chose to keep the historical baseline rather than repeat it. Converter dependency versions were restored before the full Trial.

## MoE Split
N_CPU_MOE=None resolves to 41 through the harness, with N_GPU_LAYERS=99. Experts stay on CPU; preserve configured context when tuning resources.

## Our Config Baseline
2026-10-12: context 65536, q4_0 KV, batch 512 / ubatch 128, threads 8, flash attention on, mmap, fallback sampler, no speculation. SQLite validation passed 5/5 at 31.3 t/s, peak VRAM 3.4 GB. Validation is not a complete agentic/coding vector.

Full Claw-15 + coding-10 Trial completed with outcome OK / status on_front: ; generation TPS 37.5, bench_tg 32.7, peak VRAM 3.7 GB. Description: `trial MiniCPM-V-4.7-35B-A3B b11577 serial`. Speculation remained disabled. Image/video quality remains outside this text evaluation.

## Sources / Verification
- [Publisher GGUF card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf), hf retrieval 2026-10-12.
- [Upstream trunk-only loading change](https://github.com/ggml-org/llama.cpp/pull/30097), checked 2026-10-12.
- Local GGUF metadata/tensor inventory and SQLite validation, checked 2026-10-12.

## Open Questions
- **TBD:** identify the server termination and the harness's false OK/on_front classification after HTTP failures. Do not treat the failed agentic portion as measured quality or repeat the full Trial without addressing the failure.
- Vision projector and image/video quality are untested.
