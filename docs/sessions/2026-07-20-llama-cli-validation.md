# Session Log: llama-cli Validation Gate Migration (2026-07-20)

## Goal
Transition the candidate pre-check validation gate in `run.py` and `evaluation.py` from `llama-bench` to `llama-cli` to enable real-world Speculative Decoding (MTP) throughput measurements, and ensure it runs cleanly on Windows without process blocks or memory leaks.

## Hardware
- GPU: RTX 4060 (8 GB VRAM)
- OS: Windows 11 / PowerShell

## Setup & Configuration
- **Target Model:** `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`
- **MTP Draft Model:** `draft/mtp-gemma-4-E4B-it.gguf`
- **Draft Tokens Max:** 4
- **KV Cache Quantization:** `q4_0` (Key & Value)

---

## Commands Run
```powershell
# Run manual validation check of candidate config
python -m autoresearch.runners.run --validation --desc "Test MTP 512-token generation" --model gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf --kv-k q4_0 --kv-v q4_0 --threads 6 --threads-batch 8 --batch-size 256 --ubatch-size 128 --spec-type draft-mtp --spec-draft-model draft/mtp-gemma-4-E4B-it.gguf --spec-draft-n-max 4 --no-mmap
```

---

## Findings

1. **Throughput Measurements:**
   - Shorter prompt generation (-n 128) without MTP: **69.9 t/s**.
   - Shorter prompt generation (-n 128) with MTP: **136.6 t/s** (+95.4% throughput speedup).
   - Sustained prompt generation (-n 512 - Quantum Mechanics Tutorial) with MTP: **113.4 t/s**.

2. **VRAM Fit:**
   - Fully offloaded to GPU (NGL=99), Peak VRAM usage of **6.0 GB** tracked via NVML.

---

## Errors & Corrections

### 1. KV Cache Mapping Bug
- **Error:** When parsing candidate configurations, the `ServerIntent.from_config` mapping logic fell back to `q4_0` if `kv_k`/`kv_v` properties were checked, but ignored `KV_CACHE_K` and `KV_CACHE_V` from config maps.
- **Correction:** Resolved mapping logic in `llama_runner.py` so keys are correctly normalized, enabling proper cache settings to propagate to backend commands.

### 2. Path Resolution Symphonies
- **Error:** Python's `.resolve()` followed directory symlinks from the `D:` drive to the physical compilation directory `C:\builds\`, resulting in long paths pointing to the `C:` drive in logging outputs.
- **Correction:** Updated resolvers in `llama_runner.py` to return `candidate.absolute()`, keeping paths relative and scoped within the `D:` drive repository workspace.

### 3. Server-Only Command Flag
- **Error:** Passing `--cont-batching` to `llama-cli.exe` caused an instant exit status 1 since continuous batching is a server-only HTTP orchestration feature.
- **Correction:** Removed the continuous batching flag from the `llama-cli` validator command construction in `evaluation.py`.

### 4. Interactive Loop Hangs
- **Error:** `llama-cli` defaults to an interactive terminal loop waiting for the next user prompt (`> `) when loaded with a chat template, causing the python subprocess to block indefinitely and hold VRAM memory.
- **Correction:** Appended the `--single-turn` flag to the `llama-cli` validator arguments list to guarantee the model process terminates and frees VRAM immediately upon completion of the generation turn.

### 5. Windows CP1252 Decoding Crash
- **Error:** Python subprocess decoding defaulted to system default `cp1252` encoding on Windows, which threw `UnicodeDecodeError` when `llama-cli` generated Greek letters or mathematical notations during the quantum mechanics tutorial.
- **Correction:** Configured `subprocess.run` inside `evaluation.py` to decode output using `encoding="utf-8"`.

---

## Decisions
- Transitioned candidate pre-check validation completely to `llama-cli` utilizing the `--single-turn` parameter.
- Set default generation length to `512` tokens (`BENCH_N_GEN = 512`) to ensure sustained metrics.
- Updated baseline model card in `docs/models/gemma-4-e4b.md` and `config.yaml` to specify standard `q4_0` cache formatting, ensuring immediate out-of-the-box compatibility without custom TurboQuant dependency crashes.
