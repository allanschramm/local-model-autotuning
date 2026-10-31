# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-23-ling-tiny-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```powershell
hf download bartowski/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q4_K_M.gguf --local-dir models/bartowski/Ling-3.0-tiny-GGUF
```
