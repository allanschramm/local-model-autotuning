# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-01-gemma4-v2-q3km-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Config: `CTX_SIZE=131072` (config default), `cache_type_k/v=q4_0`, `batch=1024`, `ubatch=256`, `ngl=99`, `flash_attn=on`, `cont_batching`.

## Preserved historical runtime tables

| Dataset | TPS | Notes |
| --- | --- | --- |
| LiveCodeBench | 36.4 | 1/2 passed |
| **Overall** | **38.4** |  |
| **VRAM** |  | Fits 8 GB at full 131k ctx |
