# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-26-pocket-26b-pipeline.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Champion path for `POCKET-26B-Q4_K_M.gguf` (Gemma4-26B-A4B Korean-tuned quant): validation → claw-full → coding-10.

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, upstream CUDA.

```python
MODEL = 'POCKET-26B-Q4_K_M.gguf'
CTX_SIZE = 65536
N_CPU_MOE = None  # → 30
TEMP = 1.0
TOP_P = 0.95
TOP_K = 64  # from GGUF sampling
SPEC_TYPE = None
```


| Stage | Result |
| --- | --- |
| Validation | , tg **21.8**, VRAM **4.5 GB** |
| Historical run runtime | **0.2000** (3/15), tg **20.5**, VRAM **4.5 GB** |
| Historical run runtime | **0.4900** , tg 20.4, VRAM 4.7 GB |

## Preserved historical runtime tables

| Model | tg |
| --- | ---: |
| **POCKET-35B Q3_K_M** | ~36 |
| POCKET-26B Q4_K_M | ~21 |
| gemma-4-26b-a4b (Unsloth + MTP) | ~29 |
