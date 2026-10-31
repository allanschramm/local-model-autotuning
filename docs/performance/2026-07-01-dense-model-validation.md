# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-01-dense-model-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Model | TPS | tg t/s | VRAM |
| --- | --- | --- | --- |
| **gemma-4-12B-fable5-Q3_K_M** | **43.0** | 31.2 | 7.4 GB |
| **ornith-1.0-9b-Q4_K_M** | **50.2** | 43.7 | 7.1 GB |
| gemma-4-12B-it-qat-UD-Q4_K_XL | 36.0 | 33.4 | 8.0 GB |
| Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M | 50.4 | 43.7 | 7.1 GB |
