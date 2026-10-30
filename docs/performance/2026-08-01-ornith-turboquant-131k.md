# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-01-ornith-turboquant-131k.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Run the complete Ornith pipeline at 131072 context without crossing the 7900 MB physical-VRAM gate, changing only runtime and sampler Baseline settings.

- NVIDIA 8 GB-class discrete NVIDIA, 8 GB physical VRAM
- VRAM gate: 7900 MB

- Model: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Runtime: TurboQuant+ `tqp-v0.3.0`, Windows x64 CUDA 12.4 release
- Context: 131072
- KV: Turbo2/Turbo2
- Batch/ubatch: 64/32
- Sampler: temperature 0.4, top-p 0.95, top-k 20, min-p 0, repeat penalty 1

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = '<repo>\llama.cpp-releases\turboquant\tqp-v0.3.0'
```


| Trial ID | Speculation | Bench | Peak VRAM | Result |
| --- | ---: | ---: | ---: | --- |
| `8d1db64d-8df5-42e8-ba06-bf78a9f76ef2` | MTP n=2 | 62.3 t/s | 8090 MB | VRAM gate |
| `8881827c-de13-4d66-b7e1-201ae58d435f` | MTP n=1 | not recorded | — | native exit `0xC0000409` |
| `95a8213b-4fd0-4b06-8d16-fcb7b782dced` | disabled | 43.3 t/s | 7.2 GB | complete pipeline OK |


The successful Objective Vector measured combined 53.8 TPS, (; LiveCodeBench 0.4, ), and . Runtime was 1823 seconds.

- MTP n=2 exceeded the gate by 190 MB.
- MTP n=1 is not a viable fallback in this release/model combination because `llama-cli` exits with `0xC0000409`.

- Keep the successful local Baseline at 131072, Turbo2/Turbo2, batch/ubatch 64/32, with speculative decoding disabled.
- The embedded-MTP GGUF can serve 131k safely, but MTP itself is not active in the successful Fingerprint.
