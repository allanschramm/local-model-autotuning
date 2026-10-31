# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-23-nanbeige42-tps-matrix.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```bash
# Fork (regular directory, not submodule)
git clone --branch nanbeige42 --single-branch https://github.com/Nanbeige/llama.cpp.git llama.cpp-nanbeige42
# Build CUDA Release into llama.cpp-nanbeige42/build-cuda (vcvars64 + CUDA 13.3 + Ninja)

hf download owao/nanbeige4.2-3b-gguf nanbeige4.2-3b-Q4_K_M.gguf \
  --local-dir models/owao/nanbeige4.2-3b-gguf
```

```bash
llama.cpp-nanbeige42/build-cuda/bin/llama-bench.exe \
  -m models/owao/nanbeige4.2-3b-gguf/nanbeige4.2-3b-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk <KV> -ctv <KV> -b <B> -ub <UB> \
  -p 512 -n 128,512 -r 3
```


| KV | b | ub | pp512 (t/s) | tg128 (t/s) | tg512 (t/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| q4_0 | 512 | 128 | 2410 ± 20 | 54.72 ± 0.47 | 54.55 ± 0.19 |
| q8_0 | 512 | 128 | 2395 ± 11 | 54.75 ± 0.23 | 54.53 ± 0.20 |
| **f16** | **512** | **128** | **2512 ± 43** | **55.92 ± 0.17** | **55.19 ± 0.31** |
| q8_0 | 256 | 128 | 2298 ± 47 | 55.06 ± 0.08 | 54.64 ± 0.06 |
| f16 | 256 | 128 | 2471 ± 16 | 55.82 ± 0.38 | 55.35 ± 0.22 |
| **f16** | **512** | **256** | **2727 ± 78** | **55.17 ± 0.34** | **55.17 ± 0.13** |
| f16 | 1024 | 512 | 2673 ± 28 | **56.00 ± 0.05** | 55.16 ± 0.32 |
| q4_0 | 256 | 64 | 1872 ± 17 | 55.00 ± 0.09 | 54.47 ± 0.12 |
