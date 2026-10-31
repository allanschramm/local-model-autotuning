# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-20-llama-cli-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

1. **Throughput Measurements:**
   - Shorter prompt generation (-n 128) without MTP: **69.9 t/s**.
   - Shorter prompt generation (-n 128) with MTP: **136.6 t/s** (+95.4% throughput speedup).
   - Sustained prompt generation (-n 512 - Quantum Mechanics Tutorial) with MTP: **113.4 t/s**.

2. **VRAM Fit:**
   - Fully offloaded to GPU (NGL=99), Peak VRAM usage of **6.0 GB** tracked via NVML.
