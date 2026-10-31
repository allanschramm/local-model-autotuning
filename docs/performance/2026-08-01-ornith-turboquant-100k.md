# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-01-ornith-turboquant-100k.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Test the prebuilt TurboQuant+ release for embedded-MTP Ornith 9B at the 100000-token context floor and 7900 MB physical-VRAM gate.

- NVIDIA 8 GB-class discrete NVIDIA, 8 GB physical VRAM
- VRAM gate: 7900 MB

- Model: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Runtime: TurboQuant+ `tqp-v0.3.0`, Windows x64 CUDA 12.4 release
- Context: 100000
- Sampler: temperature 0.4, top-p 0.95, top-k 20, min-p 0, repeat penalty 1
- MTP: `draft-mtp`

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = '<repo>\llama.cpp-releases\turboquant\tqp-v0.3.0'
```


| Trial ID | KV K/V | MTP n | Batch/ubatch | Bench | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `1a1a1d27-babc-4ea4-b74b-47b7fa4e2257` | turbo3/turbo3 effective due to parser defect | 4 | 256/128 | 57.1 t/s | server reached 7978 MB; cleanup then raised the historical `None.poll` error |
| `fc24f889-939b-417f-9619-252ca670bdd7` | q8_0/turbo3 | 4 | 256/128 | not run | preflight 8508 MB > 7900 MB |
| `ad0b55dd-68ed-4d9f-bc42-c5c22c2b0aed` | turbo2/turbo2 | 4 | 256/128 | not recorded | native `llama-cli` exit `0xC0000409` |
| `4b8f2aab-9b0e-45cc-91dd-6f06f62ecb07` | turbo3/turbo3 | 2 | 256/128 | 40.3 t/s | server reached 7935 MB > 7900 MB |
| `ba8d83cd-8ad8-42fc-9e1e-976296d4addf` | turbo3/turbo3 | 2 | 128/64 | 64.5 t/s | server reached 7936 MB > 7900 MB |


The final effective command included `-ctk turbo3 -ctv turbo3`, `--spec-type draft-mtp`, `--spec-draft-n-max 2`, `-b 128`, and `-ub 64`. Coding-10 and Claw full did not run because the server crossed the physical-VRAM gate.

- Keep upstream `llama.cpp/` as the only llama.cpp source clone.
- Store alternate prebuilt runtimes under gitignored `llama.cpp-releases/<engine>/<tag>/` and select them with `AUTORESEARCH_LLAMA_CPP_ROOT`.
- Reject this Ornith point at 100k: every viable TurboQuant+ server attempt exceeded 7900 MB.
- Do not reduce MTP to n=1 without an explicit speed/quality trade-off decision.
