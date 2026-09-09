---
name: validation
description: >
  Download and validate a local or Hugging Face GGUF model via smoke evaluation.
  Validates model load, throughput (bench tg t/s), VRAM fit, and basic Claw quick
  smoke. Use whenever the user asks to "validate this model", "smoke test this model",
  "validate <hf-url|model>", or wants a fast sanity check before a full Trial.
---

# validation

Operator skill to **acquire, inspect, seed, and validate** a GGUF model on any
local rig using the unified benchmark runner with `--validation`.

This skill provides a fast sanity check (model loading, VRAM fit, generation
throughput, and 5-task Claw quick smoke) without committing to a full 25-task
Objective Vector Trial.

---

## Hard Gates & Critical Invariants

1. **NEVER DELETE OR PRUNE THE `models/` DIRECTORY OR ITS TARGET:**
   - In many environments, `models/` is a symlink or NTFS junction pointing to an
     external drive, secondary storage, or LM Studio directory
     (e.g., `D:\models`, `~/.cache/lm-studio/models`, `/mnt/models`).
   - **NEVER** run directory removal commands (`rmtree`, `rm -rf`, `Remove-Item`,
     `rmdir /s`) on `models/` or any directory containing model weights.
   - Files under `models/` are user assets. Never attempt to "clean up" GGUFs or
     parent directories unless explicitly commanded by the operator.

2. **Standard GGUF Store Layout:**
   - Place models in the canonical nested layout:
     `models/<publisher>/<model-name>/<filename.gguf>`
   - The harness resolves models by basename automatically across subdirectories.

3. **No Flag Soup / Mutate Baseline via `config.py`:**
   - Always seed engine and sampler defaults in `autoresearch/core/config.py`.
   - Do not pass ad-hoc CLI flag overrides for model settings.

4. **Always Use Environment Python & Harness Tools:**
   - Windows: `.\venv\Scripts\python.exe`
   - Linux / macOS: `./venv/bin/python`
   - Never use global system Python or run raw `llama-server` / `llama-bench`
     directly for evaluation.

5. **No Command Timeouts:**
   - Benchmarks, model loads, and downloads must run until completion. Never set
     execution timeouts.

---

## Inputs

Resolve from the user's prompt:

| Input | Description | Example |
|---|---|---|
| **Target Model** | HF URL, repo ID, or local GGUF filename | `https://huggingface.co/org/repo`, `org/repo`, `Model-Q4_K_M.gguf` |
| **Quantization** | Specific quant if repo contains multiple | `Q4_K_M`, `UD-Q4_K_XL`, `F16`, `IQ4_XS` |
| **Context Length** | Desired test context (defaults to model native or 32k/65k) | `32768`, `65536` |

---

## Step-by-Step Procedure

### 1. Acquire Model (Local Check or Multi-Method Download)

#### A. Check if Already Present Locally
First, check if the model is already in `models/` (or provided via local path):
```bash
# Windows
.\venv\Scripts\python.exe scripts/model_info.py <filename.gguf>

# Linux / macOS
./venv/bin/python scripts/model_info.py <filename.gguf>
```
If `model_info.py` finds the file, skip downloading and proceed to Step 2.

#### B. If Downloading from Hugging Face
If the model must be downloaded from a Hugging Face repo, use one of the
following methods:

**Method 1: Python `huggingface_hub` (Universal fallback — works in any venv without extra CLI binaries):**
```python
from pathlib import Path
from huggingface_hub import hf_hub_download

dest = Path("models/<publisher>/<model-name>").resolve()
dest.mkdir(parents=True, exist_ok=True)
hf_hub_download(repo_id="<org>/<repo>", filename="<filename.gguf>", local_dir=str(dest))
```

**Method 2: `hf` or `huggingface-cli` (if installed/preferred):**
```bash
# Windows
.\venv\Scripts\hf.exe download <org>/<repo> <filename.gguf> --local-dir models/<publisher>/<model-name>

# Linux / macOS
./venv/bin/hf download <org>/<repo> <filename.gguf> --local-dir models/<publisher>/<model-name>
```

**Method 3: Direct Download / Manual Placement:**
- Place downloaded `.gguf` files directly into `models/<publisher>/<model-name>/`.

---

### 2. Inspect GGUF Metadata (Read-Only)

Query the GGUF header through the harness to determine architecture, block count,
native context, and KV cache sizing:

```bash
# Windows
.\venv\Scripts\python.exe scripts/model_info.py <filename.gguf>

# Linux / macOS
./venv/bin/python scripts/model_info.py <filename.gguf>
```

Record:
- **Architecture**: `dense` vs `MoE`
- **Block Count**: Number of transformer layers
- **KV Cache Estimate**: Estimated memory for target context length

---

### 3. Seed Baseline in `autoresearch/core/config.py`

Update `autoresearch/core/config.py` with baseline parameters:

- `ENGINE_DEFAULTS['MODEL'] = '<filename.gguf>'` (basename only)
- `ENGINE_DEFAULTS['CTX_SIZE'] = <context_size>` (floor 2048)
- `ENGINE_DEFAULTS['SPEC_TYPE'] = None` (or `'draft-mtp'` / `'draft-dspark'` if supported)
- `ENGINE_DEFAULTS['N_CPU_MOE'] = None` (dense models must fit physical VRAM; MoE auto-offloads)
- `SAMPLER_DEFAULTS` = Seed from model card Recommended settings (or `UNIVERSAL_FALLBACK_SAMPLER` from `config.py.example`)

---

### 4. Run Validation Benchmark

Execute the unified validation harness:

```bash
# Windows
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validation <filename.gguf>"

# Linux / macOS
./venv/bin/python benchmark_search.py --validation --desc "validation <filename.gguf>"
```

---

### 5. Check and Report Results

Read the latest row from the canonical results store (`results.db` via `autoresearch.core.results_db.load_rows`, with legacy `results.tsv` as fallback) and report the metrics:

```markdown
| Field | Measured Value | Requirement / Threshold | Status |
|---|---|---|---|
| **Throughput (`bench_tg` 512)** | `XX.X t/s` | $\ge 20.0\text{ t/s}$ (Baseline `TPS_FLOOR`) | PASS / FAIL |
| **Peak VRAM** | `X.X GB` | Fits physical VRAM without shared spill | PASS / FAIL |
| **Claw Quick Smoke (5 tasks)** | `0.XXXX` (N/5 passed) | Sanity check (T002, T004, T006, T008, T010) | Completed |
| **Category** | `validation` | Logged to `results.db` / `results.tsv` | `incomplete` |
```

---

## When to Hand Off

- **Full Objective Vector Trial**: If the operator wants full quality gating
  (`Claw-15` tasks + `coding-10` dataset suite) for Pareto frontier ranking,
  hand off to `.agents/skills/trial` (`benchmark_search.py --agentic-full`).
- **Speed Optimization**: If the operator wants to hill-climb engine knobs
  (batch, threads, KV cache precision) after a successful smoke validation,
  follow `docs/discovery/good-enough-tuning.md`.
- **Launcher Alias**: Once validated, create a local launcher alias in
  `models/aliases/<name>/config.yaml` using `.agents/skills/local-model-alias`.

---

## Anti-Patterns

- **NEVER** delete `models/`, external target folders, or model files to "clean up".
- **NEVER** assume `hf` CLI binary exists — use Python `huggingface_hub` fallback when needed.
- **NEVER** launch `autoloop.py` autonomously when asked to validate a model.
- **NEVER** pass CLI parameter overrides for engine/sampler settings — edit `config.py`.
- **NEVER** run raw `llama-server` or `llama-bench` manually for validation.
- **NEVER** set execution timeouts on model validation commands.
