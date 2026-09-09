# llama.cpp Toolset Reference

Vendored runtime at `./llama.cpp/` (repo root). Built with CUDA in `build-cuda/` — **or use a prebuilt release (recommended)**.

**Path resolution:** The autoloop resolves `llama-server` via `autoresearch/core/llama_runner.py`. If your build is elsewhere, set `export AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/llama.cpp`.

## Install (release first)

Do **not** compile llama.cpp to start — no Visual Studio, no CUDA Toolkit, no cmake.

1. Grab the latest tag from [ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases) (e.g. `b10247`).
2. Download the matching asset (Windows NVIDIA: `cudart-llama-bin-win-cuda-12.4-x64.zip` — CUDA runtime bundled, only the NVIDIA driver is needed).
3. Extract under `llama.cpp-releases/<engine>/<tag>/build-cuda/bin/` (harness looks for `build-cuda/bin`, `build-cpu/bin`, `build/bin`).
4. Point the harness at it: `export AUTORESEARCH_LLAMA_CPP_ROOT=<repo>/llama.cpp-releases/<engine>/<tag>` — or pin per alias with `llama_cpp_root` in `models/aliases/<name>/config.yaml` (`model-up`). Keep engine and release tag in Trial evidence.
5. Verify: `python scripts/serve-config.py print-cmd` shows the resolved `llama-server`.

Examples installed: `llama.cpp-releases/upstream/b10867/` (nightly CUDA 13.3, supports `spark2_5` + Nemotron-3.5 MTP), `llama.cpp-releases/turboquant/tqp-v0.3.0/`, and `llama.cpp-releases/prismml/prism-b10669-6f1c690/` (Windows CUDA 12.4/13.3 prebuilts).

## Build (only to fix something urgent)

Compile from source **only** when no release covers the need (unpublished bugfix, experimental flag).

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

- GPU: 8 GB-class discrete NVIDIA (8 GB VRAM, CUDA 8.9) — adapt paths and flags for your hardware
- **Release first**: the runtime is a prebuilt release under `llama.cpp-releases/<engine>/<tag>/build-cuda/bin/` (see Install above). `llama.cpp/` stays as the source clone/submodule but is not the default binary source — build it only for urgent fixes.
- Point the harness at a release root via `AUTORESEARCH_LLAMA_CPP_ROOT`, or pin the same root per alias with `llama_cpp_root` in `models/aliases/<name>/config.yaml` (`model-up`). Keep engine and release tag in Trial evidence.
- Installed examples: `llama.cpp-releases/upstream/b10867/` (CUDA 13.3), `llama.cpp-releases/turboquant/tqp-v0.3.0/`, and `llama.cpp-releases/prismml/prism-b10669-6f1c690/` (Windows CUDA 12.4/13.3 prebuilts).

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

Measured `-ctk/-ctv` allowed values (b10362/b10375): f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1. TurboQuant `tqp-v0.3.0` adds `turbo2`, `turbo3`, `turbo4`. Measured turbo2 ≈ 0.137×f16 on tqp (harness `VRAM_QUANT_FACTORS` 0.10 under-reads; [issue #58](https://github.com/allanschramm/local-model-autotuning/issues/58)).

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
| `-kvu` / `--kv-unified` (`-no-kvu` / `--no-kv-unified`) | One KV buffer shared across all slots: a single sequence can use the full `-c`; otherwise each slot is capped at `-c/N`. Total KV memory is unchanged — per-token size is set by `-ctk`/`-ctv`. Default on when slots are auto; inert at `-np 1`. Env `LLAMA_ARG_KV_UNIFIED`. | on (auto slots) |
| `-sm` | Split mode | layer |
| `-mg` | Main GPU | 0 |
| `-nkvo` | No KV offload | 0 |
| `-mmp` | MMAP | 1 |
| `--jinja` | Jinja2 chat templates | (often used) |
| `--prio` | Process priority | 1 |
| `--no-kv-offload` | KV on CPU | 0 |
| `--repack` / `-nr` (`--no-repack`) | CPU weight repack on/off (default: on; env `LLAMA_ARG_REPACK`) | profile-dependent — see [`cpu-inference-guide.md`](discovery/cpu-inference-guide.md) §6 |

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
---

## Flag inventory & harness audit (upstream-first)

**Rule:** harness is passthrough mapper in `autoresearch/core/llama_runner.py:895 _build_cmd` (~27 wired) vs `llama.cpp/common/arg.cpp:1424` (~348 `add_opt`, ~310 distinct flags). Before adding any `ENGINE_*/SAMPLER_*` estimator, `grep add_opt` — `90%` already exists. Full matrix in `docs/llamacpp-flags-audit.md`.

**Currently wired (~27):** `--model -c/--ctx-size -b/--batch-size -ub/--ubatch-size -t/--threads --threads-batch --parallel -ngl/--n-gpu-layers --numa --cache-type-k/v --flash-attn --no-mmap --mlock --jinja --reasoning/--reasoning-effort/--reasoning-budget/message/--reasoning-preserve --cont-batching --cache-reuse --spec-type --spec-draft-n-max --spec-draft-type-k/v --spec-draft-model --n-cpu-moe` (+ bench `-p -n -r -o`). Everything else untapped.

**Upstream lacks — keep (value-add):** `preflight_host_memory:662` full `GGUF(21.7GB)+KV+draft` vs `RAM-max(6144,0.2RAM)` unified / `max(4096,0.15RAM)` discrete fail-closed `autoresearch/core/hardware.py:506`; `physical-512 keepout` + `SHARED 2048` kill + `GGML_CUDA_NO_PINNED=1` `llama_runner.py:305`; `free-at-start - headroom` clamp issue #10; `thermal` wait; `TPS_FLOOR/REPS`; Pareto Day/Night `scripts/rank_results.py` (`common/fit.h:14` assumes unlimited host, so host/WDDM/Pareto gates are the only keepers).

### Safety gaps (wire before new code)

| Upstream | Opportunity |
|---|---|
| `--fit on/off` `arg.cpp:2813` `--fit-target MiB` `:2842` `--fit-ctx N` `:2866` `--fit-print` `:2827` `fit.h:14` `common_fit_params fit.cpp:178` VRAM-only via `ggml_backend_dev_memory` | `fit-print`/`tools/fit-params` for manual sizing only; Trials keep deterministic reject (`fit=on 256` would silently shrink `c=65536` but still `mmap 21.7GB` -> host OOM). |
| `--load-mode none/mmap` `--mmap/--no-mmap` `--mlock` `--direct-io` `--defrag-thold` `:2402` | `NO_MMAP bool` conflates; `load-mode` superset. `mmap+mlock` locks 21.7GB on 16GB unified -> OOM. Default `mmap`+`mlock False`. |
| `--warmup/--no-warmup` | Not wired; `THERMAL_WAIT` only. `--no-warmup` required 8GB tight (empty warmup OOM). |
| `--cache-ram -1` `--kv-unified/--no-kv-unified` `--no-kv-offload` `--repack/--no-repack` `--op-offload/--no-op-offload` `:1704` | `cache-ram` caps KV MiB, `kv-unified` ~512MB @65k save, `no-kv-offload` keeps KV on CPU for MoE (0.7GB `q4_0` vs 2.6GB `f16`), `repack` +3% tg (see `docs/discovery/cpu-inference-guide.md:6`). |

### Perf gaps

| Flag | Opportunity |
|---|---|
| `--cache-type-k/v f16/q8_0/q4_0/turbo2/3/4` `:1735` | We `q4_0` only; `turbo3` `76t/s` vs `73t/s` gemma `results.tsv:94` (tqp fork only). |
| `--swa-full --ctx-checkpoints/--swa-checkpoints --checkpoint-min-step` `:1678` | `swa-full` +15% long ctx SWA. |
| `--threads --threads-batch --cpu-mask/--cpu-range/--cpu-strict --poll/--poll-batch --prio --threads-http` `:1439` | We `THREADS 8`; pin `cpu-mask` P-cores + `poll` +5% on 5800X. |
| `--spec-type mtp/draft-mtp/draft/dflash/ngram` `--spec-draft-n-max/min --spec-draft-p-min/--p-split --spec-draft-backend-sampling` `--spec-ngram-*` `--lookup-cache-static/dynamic` `:1613,2201` | We `draft-mtp` only; `ngram` +8-20% no draft VRAM, `p-min 0.8 p-split 0.9` +10% acceptance. |
| `--flash-attn on/off/auto` `:1735` | We hard `on`; `auto` avoids SWA fallback loss. |
| `--tensor-split --split-mode --main-gpu --device --override-tensor` | Single-GPU defer; `override-tensor blk.40.*=CPU` explicit vs fit regex. |

### Quality gaps

| Flag | Opportunity |
|---|---|
| `--rope-scaling linear/yarn --rope-freq-base/scale --yarn-* --grp-attn-n/w` | `yarn-ext-factor 1.5` extends `65k->98k` for T053 `124k` overflow. |
| `--samplers --sampling-seq --temp/--top-p/--top-k/--min-p/--repeat-penalty --dry-* --xtc-* --top-nsigma/--typical-p --dynatemp-* --mirostat` `:2004` | We 7 keys only; `dry 1.1` fixes loops, `xtc 0.1` diversity. |
| `--override-kv --override-tensor --lora --control-vector --grammar/--json-schema` | `override-kv tokenizer.add_bos=false` fix; `lora` without retrain. |

**Prioritize:** Safety `load-mode/no-warmup/fit-print` > Perf `turbo/repack/kv-unified/ngram/poll/swa` > Quality `dry/xtc/yarn`. Full `~310` list + per-flag file:line in `docs/llamacpp-flags-audit.md` + `docs/discovery/` guides (`cpu-inference-guide`, `mtp-baseline-guide`, `speculative-decoding-formats`).

