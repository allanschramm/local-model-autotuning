# MTP Speculative Decoding & llama-bench Guide

This guide explains how to verify and baseline speculative decoding (Multi-Token Prediction / MTP) on your local hardware before starting a full model autotuning run.

---

## 1. MTP Verification Concept

MTP speculative decoding speeds up text generation by predicting multiple tokens per step using prediction heads built into the model. However, it requires:
1. **Compatible GGUF file:** The model must be downloaded from the `-MTP-GGUF` repository variant (e.g. `unsloth/Qwen3.5-9B-MTP-GGUF`), which includes the draft prediction head tensors directly in the `.gguf` file.
2. **Speculative flag:** You must run the server or CLI with `--spec-type draft-mtp` and specify `--spec-draft-n-max 2` (recommended baseline).

---

## 2. Test MTP Speedup via `llama-cli`

To test if MTP is working and measure the exact speedup, run a test prompt with and without MTP using the `llama-cli` binary.

### Run WITH MTP:
```bash
./llama.cpp/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0
```
*Note the generation speed at the end (e.g. `Generation: 69.1 t/s`).*

### Run WITHOUT MTP (Baseline):
```bash
./llama.cpp/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --spec-type none \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0
```
*Note the generation speed (e.g. `Generation: 39.1 t/s`). Compare the speedup ratio.*

If the MTP run throws an error like `failed to create MTP context`, the loaded GGUF file does not contain the MTP tensors. You must download the MTP version of the GGUF file.

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
