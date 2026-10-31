# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-24-coding-10-claw-leaders.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA.


| Model | ctx | KV | Notes |
| :--- | ---: | :--- | :--- |
| `Laguna-XS-2.1-Q3_K_XL` | 65k | q4_0 | `N_CPU_MOE=40` |
| `LFM2.5-1.2B-Instruct-Q8_0` | 65k | f16 | dense |
| `Ornith-1.0-9B-UD-Q4_K_XL` | **32k** | q4_0 | 65k VRAM-kill mid-coding |


| Model | VRAM |
| :--- | ---: |
| Laguna | 4.5 GB |
| LFM 1.2B | 4.0 GB |
| Ornith 32k | 7.4 GB |


1. **Ornith @ 65k:** `VRAM_LIMIT EXCEEDED used=7950MB > limit=7900MB` during HumanEval → fail row. Retry @ **32k** OK.
2. **LCB all three:** `WinError 1314` symlink. Fixed later via copy2 + [LCB patch session](2026-07-24-lcb-patch-gambiarra.md).

* Coding alias for Ornith dense on 8 GB: prefer **ctx ≤32k** (agentic alias stays 65k).

## Preserved historical runtime tables

| Model | bench_tg | VRAM |
| :--- | ---: | ---: |
| Laguna | 35.2 | 4.5 GB |
| LFM 1.2B | 167.6 | 4.0 GB |
| Ornith 32k | 38.4 | 7.4 GB |
