# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/discovery/capability-extraction-harness.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.


| Technique | Vendor claim (2026) | In-repo measurement (8 GB-class) |
| --- | --- | --- |
| DFlash (block-diffusion drafter, NVIDIA/DeepSeek DeepSpec) | 3–6× lossless; up to 15× Blackwell | 12.5 vs 24.6 t/s (−49%) on Qwen3.6-35B-A3B with `n-cpu-moe`; needs 35B target fully on GPU |
| DSpark (semi-autoregressive drafter + confidence head + hardware-aware scheduler, DeepSeek) | 60–85% faster per-user, up to 6.6× throughput | 19.2 vs 37.3 t/s (−48.5%) on Bonsai-27B Q1_0 (quantization-speed inversion: Q4 draft slower than Q1 target) |
