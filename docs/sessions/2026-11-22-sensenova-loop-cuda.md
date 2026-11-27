# SenseNova Loop 9B CUDA validation, November 22, 2026

The Windows trials used the publisher's `SenseNova-Loop-9B-Q4_K_M.gguf` on an 8 GiB-class CUDA device. The tested `b11577` and `b12633` binaries lacked the `sensenova_loop` architecture. The session used isolated CUDA 13.5 builds of the publisher-recommended fork revision `24ea7bb` and the optimized `9804dbf` branch. Check architecture support in the exact binary before setting up a Trial. [Publisher GGUF](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF/blob/main/SenseNova-Loop-9B-Q4_K_M.gguf) · [Fork recommendation](https://huggingface.co/sensenova/SenseNova-Loop-9B-GGUF/discussions/3) · [Upstream implementation](https://github.com/ggml-org/llama.cpp/pull/32588) · [CUDA optimization](https://github.com/blueswhen/llama.cpp/commit/9804dbfd10d1be9db5f4ccb58460ed2ade263c68)

All three Trials allocated 65,536 tokens with K/V `q4_0`, Flash Attention, batch/ubatch 256/64 and eight inference and batch threads. MTP was disabled. Frozen sampler settings were temperature 1.0, top-p 0.95, top-k 40, min-p 0.05, repeat penalty 1.0 and presence penalty 0. Each successful Trial used the fixed 2,048-token PPL check and three 512-token generations from 32-token prompts. [Tuning protocol](../../autoresearch/TUNING.md)

| Runtime | GPU layer setting | Median TPS | PPL | SQLite run prefix |
| --- | ---: | ---: | ---: | --- |
| Fork `24ea7bb`, partial placement diagnostic | 40 | 4.81 | 13.1913 | `918c841b` |
| Fork `24ea7bb`, full placement control | 99 | 10.68 | 13.2930 | `d35ff375` |
| Optimized CUDA `9804dbf`, full placement | 99 | 27.36 | 12.9995 | `2bcc2ba9` |

Full placement on the original fork improved throughput 2.22× and established the GPU-only control. The matched runtime swap then improved throughput 2.56×. The model stores 32 physical layers and executes 56 logical layer steps. The optimized branch revises CUDA routing and fuses operations in the mHC path. The measured gain belongs to the whole runtime change. Individual kernel and fallback costs were unmeasured.

An earlier `b12633` Trial used the TwIL Baseline model and hit the Windows shared-memory watchdog. Its recorded context was 65,536, PPL was 12.2835 and TPS was unavailable. Check the selected GGUF in the resolved Baseline before interpreting a runtime upgrade or failure. See run prefix `1635cbea` in local `results.db`.

A later `model-up` check verified API model identity and text generation with the optimized Trial settings.
