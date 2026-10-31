# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-02-qwythos-claude-mythos-100k.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Complete the throughput, coding-10, and Claw full Objective Vector for `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` at 100000 context for the Qwythos comparison.

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP disabled, per model-card recommendation.
- Batch/ubatch: `16/8`.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `20`, repeat penalty `1.05`.

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
```


| Metric | Result |
| --- | ---: |
| Backend bench | 42.1 t/s |
| Historical run runtime | 47.9 t/s |
| LiveCodeBench | 0.5000 |
| Peak VRAM | 7.3 GB |


- Preserve this complete 100k point for comparison with both Qwythos v2 artifacts.
- Normal v2 and Claude-Mythos are batch-matched at `16/8`; their measured TPS is directly comparable under this setup.
- The complete MTP point uses `32/16`. Its strict `16/8` Trial crossed the physical-VRAM gate, so report MTP's higher TPS as a best-feasible result, not a batch-matched speedup.
