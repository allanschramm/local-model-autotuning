# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-24-claw-full-smoke-high.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Descs used:
- `claw-full Laguna-XS-2.1-Q3_K_XL n-cpu-moe40 ctx65k`
- `claw-full Ornith-1.0-9B-UD-Q4_K_XL ctx65k`
- `claw-full Ornith-1.0-35B-UD-Q4_K_XL n-cpu-moe40 ctx65k`
- `claw-full Bonsai-27B-Q1_0 ctx131k` → **FAIL** VRAM
- `claw-full Bonsai-27B-Q1_0 ctx65k (131k VRAM-kill retry)` → KEEP


| Model (basename) | peak VRAM | ctx |
| :--- | :---: | :---: |
| `Laguna-XS-2.1-Q3_K_XL` | 3.6 GB | 65k |
| `Ornith-1.0-9B-UD-Q4_K_XL` | 7.5 GB | 65k |
| `Ornith-1.0-35B-UD-Q4_K_XL` | 4.9 GB | 65k |
| `Bonsai-27B-Q1_0` | 6.5 GB | **65k** |


## Preserved historical runtime tables

| Model (basename) | bench_tg | peak VRAM | ctx |
| :--- | :---: | :---: | :---: |
| `Laguna-XS-2.1-Q3_K_XL` | 37.2 | 3.6 GB | 65k |
| `Ornith-1.0-9B-UD-Q4_K_XL` | 42.1 | 7.5 GB | 65k |
| `Ornith-1.0-35B-UD-Q4_K_XL` | 25.7 | 4.9 GB | 65k |
| `Bonsai-27B-Q1_0` | 40.2 | 6.5 GB | **65k** |
