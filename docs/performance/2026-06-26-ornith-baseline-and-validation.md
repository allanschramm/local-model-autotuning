# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-06-26-ornith-baseline-and-validation.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

**Date**: 2026-06-26
**Focus**: Establishing a reliable baseline for the new model `Ornith-1.0-9B` and creating a fast validation mode to test new models' VRAM and TPS.

- **Model**: `ornith-1.0-9b-Q4_K_M.gguf` (Offloaded to GPU with `NGL=99`)
  - **LiveCodeBench**: `0.4000`
  - **BigCodeBench Hard**: `0.1000`
- **Peak VRAM**: `7.9 GB`
- **TPS**: `49.4`
- **Status**: `KEEP` (committed locally to `main`)
