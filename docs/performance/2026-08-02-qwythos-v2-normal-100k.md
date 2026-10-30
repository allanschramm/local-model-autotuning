# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-02-qwythos-v2-normal-100k.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Complete the throughput, coding-10, and Claw full Objective Vector for the no-MTP Qwythos v2 artifact at 100000 context.

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

- Model: `Qwythos-9B-v2-Q4_K_M.gguf`.
- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP disabled.
- Batch/ubatch: `16/8`.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `20`, repeat penalty `1.05`.

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
```


| Metric | Result |
| --- | ---: |
| Backend bench | 41.6 t/s |
| Historical run runtime | 47.3 t/s |
| LiveCodeBench | 0.4000 |
| Peak VRAM | 7.2 GB |


- Batch/ubatch `32/16` crossed the physical-VRAM gate at 7902 MB in the preceding Trial.

- Keep `16/8` as the complete no-MTP 100k point.
- Do not merge this row with the embedded-MTP point; they are distinct model basenames and Fingerprints.
- A strict MTP speed A/B must use matching batch/ubatch in addition to matching context, sampler, KV, and runtime.
