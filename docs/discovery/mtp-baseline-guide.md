# MTP Speculative Decoding & llama-bench Guide

This guide explains how to verify and baseline speculative decoding (Multi-Token Prediction / MTP) on your local hardware before starting a full model autotuning run.

---

## 1. MTP Verification Concept

MTP speeds generation by drafting multiple tokens per step. Packaging differs by model family:

1. **Embedded `nextn`:** tensors inside the main GGUF (e.g. local `Qwen3.5-9B-UD-Q4_K_XL.gguf`, Ornith/Mythos `*-MTP*.gguf`). Flags: `--spec-type draft-mtp --spec-draft-n-max N` (no draft file).
2. **External draft:** separate assistant GGUF (Gemma-4 E4B → `models/draft/mtp-gemma-4-E4B-it.gguf`). Main UD file has **no** `nextn`. Flags: add `--spec-draft-model <draft>`.

Detect: scan GGUF metadata for `nextn` / `gemma4-assistant`. Inventory + measured TPS: [small-model-mtp-tps.md](./small-model-mtp-tps.md).

Prefer the harness gate (`run_llama_bench_validation`) over raw `llama-bench` — bench binaries do not accept MTP draft flags. Canonical `n_max` on this rig for speed matrix: **4** (not 2).

---

## 2. Test MTP Speedup via `llama-cli`

### Embedded MTP (Qwen) — WITH:
```bash
./llama.cpp/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```

### Gemma external draft — WITH:
```bash
./llama.cpp/build-cuda/bin/llama-cli.exe \
  -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-model models/draft/mtp-gemma-4-E4B-it.gguf \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```

### WITHOUT MTP (Baseline):
```bash
./llama.cpp/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```
(Omit `--spec-type` / draft flags entirely, or set `--spec-type none` if your build requires an explicit disable.)

If the MTP run throws `failed to create MTP context` (or similar), that GGUF has no usable MTP heads — download a `*-MTP-GGUF` variant or the matching Gemma draft.

---

## 3. Base Benchmarking via `llama-bench`

To benchmark the base raw performance of the models at a specific context depth (e.g., 65k context) without speculative decoding, use `llama-bench`.

### Test 9B Model at 65k context depth:
```bash
./llama.cpp/build-cuda/bin/llama-bench.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```

### Test 35B MoE Model at 65k context depth (CPU Offload):
```bash
./llama.cpp/build-cuda/bin/llama-bench.exe \
  -m models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 40 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```

### Test 26B MoE Model at 65k context depth (CPU Offload):
```bash
./llama.cpp/build-cuda/bin/llama-bench.exe \
  -m models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -ncmoe 30 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```
*Note: Gemma-4-26B requires `--n-cpu-moe 30` on 8 GB GPUs to prevent VRAM swapping and thrashing.*
