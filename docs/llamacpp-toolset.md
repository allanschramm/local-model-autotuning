# llama.cpp Toolset Reference

Vendored runtime at `./llama.cpp/` (repo root). Built with CUDA in `build-cuda/`.

**Path resolution:** The autoloop resolves `llama-server` via `autoresearch/core/llama_runner.py`. If your build is elsewhere, set `export AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/llama.cpp`.

## Build

Windows toolchain (installed 2026-07-17): VS 2022 Build Tools (MSVC 14.44, VCTools workload), NVIDIA CUDA Toolkit 13.3, Ninja. nvcc on Windows requires MSVC host compiler; configure must run inside `vcvars64.bat` env. Helper `llama.cpp/rebuild-cuda.bat` (untracked, machine-specific) wraps this: `rebuild-cuda.bat configure` = full configure+build, `rebuild-cuda.bat` = incremental build. Keep `build-cuda/` as a real in-repo directory (no external junctions).

```bash
# Full build with CUDA — inside vcvars64 env, CUDA bin on PATH
cmake -S ./llama.cpp -B ./llama.cpp/build-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build ./llama.cpp/build-cuda --config Release

# Full build for CPU-only (no GPU hardware required)
cmake -S ./llama.cpp -B ./llama.cpp/build-cpu -G Ninja -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ./llama.cpp/build-cpu --config Release

# Build single target
cmake --build ./llama.cpp/build-cuda --target llama-bench
```

Targets are prefixed `llama-` (e.g. `llama-quantize`, `llama-perplexity`).

## Built binaries location

All binaries → `$LLAMA_CPP/build-cuda/bin/` or `$LLAMA_CPP/build-cpu/bin/` (or `./llama.cpp/build-cuda/bin/` / `./llama.cpp/build-cpu/bin/` by default)

| Binary | Purpose | Use case |
|--------|---------|----------|
| `llama-server` | OpenAI-compatible HTTP server | Autotuning inference target |
| `llama-bench` | Performance benchmarking | TPS/throughput measurement |
| `llama-cli` | CLI inference | Manual model testing |
| `llama-quantize` | GGUF quantization | Model compression |
| `llama-perplexity` | Perplexity evaluation | Quality measurement |
| `llama-imatrix` | Importance matrix generation | Quantization optimization |
| `llama-gguf-split` | GGUF split/merge | Large model handling |

## Hardware

- GPU: RTX 4060 (8 GB VRAM, CUDA 8.9) — adapt paths and flags for your hardware
- Optional external forks: clone elsewhere and set `AUTORESEARCH_LLAMA_CPP_ROOT` (not vendored in this repo)

---

## llama-bench

Performance testing. Measures prompt processing (pp) and text generation (tg) in tokens/sec.

### Basic usage

```bash
# Default: pp512 + tg128 with 5 reps
./bin/llama-bench -m /path/to/model.gguf

# Custom prompt/gen lengths, 3 reps
./bin/llama-bench -m model.gguf -p 1024 -n 256 -r 3

# TG only (no prompt processing)
./bin/llama-bench -m model.gguf -p 0 -n 128,256,512

# PP only
./bin/llama-bench -m model.gguf -n 0 -p 512

# Combo test (prompt then generate)
./bin/llama-bench -m model.gguf -pg 512,128
```

### GPU offloading

```bash
# Full GPU
./bin/llama-bench -m model.gguf -ngl -1

# Partial offload
./bin/llama-bench -m model.gguf -ngl 20,30,35

# Disable GPU (CPU only)
./bin/llama-bench -m model.gguf -ngl 0
```

### Key flags

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model path | models/7B/ggml-model-q4_0.gguf |
| `-p` | Prompt tokens | 512 |
| `-n` | Generate tokens | 128 |
| `-pg` | Combo pp,tg | (none) |
| `-d` | KV cache prefilled depth | 0 |
| `-b` | Batch size | 2048 |
| `-ub` | UBatch size | 512 |
| `-t` | Threads | system-dependent |
| `-ngl` | GPU layers (-1 = all) | -1 |
| `-fa` | Flash attention on/off/auto | auto |
| `-ctk/-ctv` | KV cache type (f16/q8_0) | f16 |
| `-sm` | Split mode (layer/row/tensor) | layer |
| `-r` | Repetitions | 5 |
| `-o` | Output format (md/csv/json/jsonl/sql) | md |

### Output formats

```bash
# Machine-readable
./bin/llama-bench -m model.gguf -o json
./bin/llama-bench -m model.gguf -o csv

# SQLite import
./bin/llama-bench -m model.gguf -o sql | sqlite3 results.db
```

### Multi-value ranges

```bash
# Comma-separated
./bin/llama-bench -m model.gguf -t 4,8,12

# Range
./bin/llama-bench -m model.gguf -t 1-8

# Range with step
./bin/llama-bench -m model.gguf -pg 64,128 64,256+64

# Range with multiplier
./bin/llama-bench -m model.gguf -b 128-1024*2  # 128,256,512,1024
```

### Hugging Face direct

```bash
./bin/llama-bench -hf ggml-org/Qwen2.5-7B-Instruct-GGUF:Q4_K_M
./bin/llama-bench -hf user/repo -hff my-model.gguf
```

### Multi-model comparison

```bash
./bin/llama-bench -m model1.gguf -m model2.gguf -p 0 -n 128,256
```

### List devices

```bash
./bin/llama-bench --list-devices
```

### MTP (speculative decoding) testing

`llama-bench` does **not** take `--spec-type` / `--spec-draft-model`. For MTP TPS use `llama-cli` (harness: `run_llama_bench_validation`) or see [mtp-baseline-guide.md](discovery/mtp-baseline-guide.md) and [small-model-mtp-tps.md](discovery/small-model-mtp-tps.md).

Upstream KV types only (`q4_0`, …). `turbo*` KV cache types are **not** in upstream `llama.cpp`.

```bash
# Context-depth bench WITHOUT speculation:
./bin/llama-bench -m model.gguf -pg 512,128 -d 0
# -d prefills KV cache to simulate context state
```

---

## llama-server

OpenAI API-compatible HTTP server — primary inference target for autotuning.

### Basic server

```bash
./bin/llama-server \
  -m /path/to/model.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  -c 8192 \
  -ngl -1
```

### Key flags for autotuning

| Flag | Description | Typical value |
|------|-------------|---------------|
| `-c` | Context size | 8192+ |
| `-ngl` | GPU layers (-1 = all) | -1 |
| `-b` | Batch size | 2048 |
| `-ub` | UBatch size | 512 |
| `-t` | CPU threads | 8 |
| `-fa` | Flash attention | on |
| `-ctk` | K cache type | f16 |
| `-ctv` | V cache type | f16 |
| `-sm` | Split mode | layer |
| `-mg` | Main GPU | 0 |
| `-nkvo` | No KV offload | 0 |
| `-mmp` | MMAP | 1 |
| `--jinja` | Jinja2 chat templates | (often used) |
| `--prio` | Process priority | 1 |
| `--no-kv-offload` | KV on CPU | 0 |

### Server management

```bash
# Find running instance
pgrep -f "llama-server"

# Kill
pkill -f "llama-server"

# Check API
curl http://127.0.0.1:8081/v1/models
```

### Chat completion (API)

```bash
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

### Configuration server (this repo)

```bash
# Launch from repo root using autotuning config
python3 scripts/serve-config.py serve
```

This reads `autoresearch/core/config.py` and spawns `llama-server` with the current tuning flags.

---

## llama-cli

Command-line inference. Useful for quick manual testing without HTTP.

```bash
./bin/llama-cli -m model.gguf -p "Your prompt here" -n 256 -ngl -1
```

---

## llama-quantize

GGUF quantization and requantization.

```bash
# Basic quantize
./bin/llama-quantize model.f16.gguf model.q4_k_m.gguf Q4_K_M

# With importance matrix (better quality)
./bin/llama-quantize --imatrix imatrix.dat model.f16.gguf model.q4_k_m.gguf Q4_K_M

# Keep output tensor unquantized
./bin/llama-quantize --leave-output-tensor model.f16.gguf model.q4_k_m.gguf Q4_K_M

# Pure quantization (no K-quant mixing)
./bin/llama-quantize --pure model.f16.gguf model.q4_k_m.gguf Q4_K_M
```

### Common quantization types

| Type | BPW | Quality |
|------|-----|---------|
| Q2_K | 2.56 | lowest |
| Q3_K_M | 3.35 | low |
| Q4_K_M | 4.35 | balanced |
| Q5_K_M | 5.34 | good |
| Q6_K | 6.56 | high |
| Q8_0 | 8.50 | highest |
| F16 | 16.0 | full |

---

## llama-perplexity

Perplexity evaluation for quality measurement.

```bash
./bin/llama-perplexity -m model.gguf -f test_data.txt -ngl -1
```

---

## llama-imatrix

Generate importance matrix for smarter quantization.

```bash
./bin/llama-imatrix -m model.f16.gguf -f calibration_data.txt -o imatrix.dat -ngl -1
```

---

## llama-gguf-split

Split or merge GGUF files.

```bash
# Split large model (e.g., for 2GB chunks)
./bin/llama-gguf-split --split --split-max-size 2G model.gguf model-split.gguf

# Merge splits back
./bin/llama-gguf-split --merge model-split.gguf.1 model-merged.gguf
```

---

## Autotuning-relevant combos

### Measure baseline TPS

```bash
./bin/llama-bench -m /path/to/model.gguf -p 512 -n 128 -ngl -1 -fa on -o json
```

### Compare flag variants

```bash
./bin/llama-bench -m model.gguf -ngl -1 -fa on,off -b 1024,2048 -p 512 -n 128 -o csv
```

### Validate server config

```bash
# Match server flags in bench
./bin/llama-bench -m model.gguf -p 512 -n 128 \
  -b 2048 -ub 512 -t 8 -ngl -1 -fa on -ctk f16 -ctv f16
```

### Quick server smoke test

```bash
./bin/llama-server -m model.gguf -c 8192 -ngl -1 --host 127.0.0.1 --port 8081 &
sleep 3
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model","messages":[{"role":"user","content":"hi"}],"max_tokens":10}' | head -c 200
kill %1
```

## Validation gate (this repo)

`run.py --validation` now runs **llama-cli** instead of `llama-bench`.
- **Reason:** `llama-bench` does not support custom speculative decoding arguments (such as `--spec-draft-model` and `--spec-draft-n-max`). Transitioning to `llama-cli` allows the validation gate to run real speculative decoding configurations and measure correct, accelerated MTP throughput values.
- **Workflow:** It runs a single-turn generation test with a 512-token generation limit.
- **Process Cleanup:** It uses the `--single-turn` flag to ensure the process exits cleanly and unloads the model from VRAM immediately upon completion.
- **Encoding:** Subprocesses are configured to decode output in UTF-8 to prevent CP1252 Windows encoding crashes when math notations are generated.

```bash
# Quick validation (default: tg >= 20 t/s threshold over 512 tokens)
python -m autoresearch.runners.run --validation --desc "test config"

# Custom threshold
python -m autoresearch.runners.run --validation --desc "test config" --bench-tts-threshold 25
```

The validation gate runs automatically before every full evaluation. If the `llama-cli` test generation speed is below the threshold, the trial is skipped immediately.

```bash
# Full run with automatic bench pre-check
python -m autoresearch.runners.run --desc "sweep batch 1024" --batch-size 1024
# Output: [cli-bench] ... tg 113.4 t/s → passes → starts server → runs coding
```

---

## Path resolution (this repo)

`llama-server` binary found via (in order):

1. `$AUTORESEARCH_LLAMA_CPP_ROOT/build-cuda/bin/llama-server`
2. `<repo>/llama.cpp/build-cuda/bin/llama-server`
3. `<repo_parent>/llama.cpp/build-cuda/bin/llama-server`

Default: `./llama.cpp/build-cuda/bin/`

---

## Build shortcuts

```bash
# Set this once per session (or add to .bashrc)
LLAMA_CPP="${AUTORESEARCH_LLAMA_CPP_ROOT:-./llama.cpp}"

# Build all tools (CUDA) — incremental, wraps vcvars64 env
alias lc-build="cmd //c $LLAMA_CPP/rebuild-cuda.bat"

# Build single tool (edit target name into rebuild-cuda.bat call if needed, or run full lc-build)

# Quick bench
alias lbench="$LLAMA_CPP/build-cuda/bin/llama-bench"
```
