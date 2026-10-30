# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-26-bonsai-coding-vs-pocket.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Fair coding-10 on `Bonsai-27B-Q1_0` at the same ctx/sampler recipe as POCKET-35B Q3_K_M, to close the head-to-head Objective Vector.

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, upstream CUDA.

```python
MODEL = 'Bonsai-27B-Q1_0.gguf'
CTX_SIZE = 65536
KV_CACHE_K = KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # dense
THREADS_BATCH = 12
CONT_BATCHING = False
SPEC_TYPE = None
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
```


| Model | VRAM |
| --- | ---: |
| **POCKET-35B Q3_K_M** | 3.7 GB |
| Bonsai-27B Q1_0 | 7.5 GB |


- POCKET +35% coding, +43% claw-full, ~half the VRAM.
- Bonsai +19% TPS only.

- Prefer **POCKET-35B** for agentic + coding on the operator host (higher claw + coding; lower peak VRAM).
- Bonsai remains a higher-TPS / lower-quality point on the same axes.

## Preserved historical runtime tables

| Model | tg | VRAM |
| --- | ---: | ---: |
| **POCKET-35B Q3_K_M** | 35.0 | 3.7 GB |
| Bonsai-27B Q1_0 | **41.7** | 7.5 GB |
