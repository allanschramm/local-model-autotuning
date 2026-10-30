# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-01-day-models-131k-pipeline.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Measure three DAY candidates with throughput, coding-10, and Claw-Eval full on the same 131072-context fingerprint.

- Discrete NVIDIA GPU with 8 GB physical VRAM.
- Host-memory preflight budget: 27790 MB.
- Official `llama.cpp` CUDA build.

- `CTX_SIZE=131072`; KV K/V `q4_0`; batch / ubatch `1024 / 256`; threads `8 / 8`; flash attention on.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `0`, min-p `0`, repeat `1.0`, presence `0`, frequency unset.
- Granite's official cards and generation configs published no sampler values; the operator explicitly authorized this universal fallback.


| GGUF basename | Context | Combined TPS | Peak VRAM |
| --- | ---: | ---: | ---: |
| `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 131072 | 90.6 | 5.6 GB |
| `granite-4.0-h-tiny-Q4_K_M.gguf` | 131072 | 79.9 | 3.6 GB |
| `granite-4.1-3b-Q4_K_M.gguf` | 131072 | 66.9 | 7.2 GB |


Granite 4.0 H Tiny resolved `N_CPU_MOE=None` to `--n-cpu-moe 40`. Granite 4.1 3B and Nemotron were dense and remained within physical VRAM.

- Nemotron's first Claw full attempt failed when the web mock inherited the Windows legacy code page. The harness now starts mock services with `PYTHONUTF8=1`; focused runner tests passed `7/7`, and the retry completed.
- Granite 4.0's first full attempt ended after one coding request with `[Errno 22] Invalid argument`. The server was healthy; the launcher had closed the output handle. Redirecting stdout/stderr at process creation produced a complete retry without changing the fingerprint.

## Preserved historical runtime tables

| GGUF basename | Context | Bench tg | Combined TPS | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 131072 | 79.2 | 90.6 | 5.6 GB |
| `granite-4.0-h-tiny-Q4_K_M.gguf` | 131072 | 51.2 | 79.9 | 3.6 GB |
| `granite-4.1-3b-Q4_K_M.gguf` | 131072 | 92.0 | 66.9 | 7.2 GB |
