# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-24-claw-full-top-tps.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Run Claw-Eval **full** (n=15) on the three highest measured TPS models from `results.tsv`: LFM2.5-1.2B, LFM2.5-8B-A1B, Gemma-4-E4B (draft MTP).

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA (`build-cuda`).


| Model | Key Baseline knobs |
| :--- | :--- |
| `LFM2.5-1.2B-Instruct-Q8_0.gguf` | ctx **65k**, KV **f16**, dense, `TPS_FLOOR=15` |
| `LFM2.5-8B-A1B-Q4_K_M.gguf` | ctx **65k**, KV **q4_0**, `N_CPU_MOE=0`, temp 0.2 / top_k 80 / rp 1.05 |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | ctx **65k**, draft-mtp n=4, batch 256/128, threads 6 |


| Model | peak VRAM |
| :--- | ---: |
| LFM2.5-1.2B | 3.7 GB |
| LFM2.5-8B-A1B (run A) | 7.3 GB |
| LFM2.5-8B-A1B (run B) | 7.8 GB |
| Gemma-4-E4B MTP | 5.8 GB |


1. **Config race:** Shell launched in parallel with `config.py` StrReplace → first “Gemma” job actually re-ran LFM 8B. Fix: verify `ENGINE_DEFAULTS['MODEL']` print before harness.
2.

* LFM 1.2B @ 65k f16 joins Ornith tier on ) with ~4× TPS — strong speed+agentic Pareto candidate.


## Preserved historical runtime tables

| Model | bench_tg | peak VRAM |
| :--- | ---: | ---: |
| LFM2.5-1.2B | 166.4 | 3.7 GB |
| LFM2.5-8B-A1B (run A) | 184.3 | 7.3 GB |
| LFM2.5-8B-A1B (run B) | 178.5 | 7.8 GB |
| Gemma-4-E4B MTP | 113.3 | 5.8 GB |
