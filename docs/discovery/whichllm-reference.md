# whichllm CLI Reference

Hardware-aware LLM finder. Auto-detects GPU/CPU/RAM, ranks models by fit and quality.

**Version:** 0.5.12  
**Install:** `uvx whichllm@latest` (requires `uv`/`uvx`)

## Global Options

| Flag | Short | Description |
|:---|:---|:---|
| `--top` | `-n` | Number of top models to show (default: 10) |
| `--context-length` | `-c` | Context length for KV cache estimation (e.g. 4096, 64k, 128k) |
| `--quant` | `-q` | Filter by quantization type (e.g. Q4_K_M) |
| `--speed` | | Speed preset: `any` \| `usable` \| `fast` |
| `--min-speed` | | Exact minimum tok/s filter |
| `--fit` | | Runtime fit filter: `any` \| `gpu` \| `full-gpu` |
| `--gpu-only` | | Only models that fit fully in GPU VRAM |
| `--evidence` | | Benchmark evidence filter: `strict` \| `base` \| `any` |
| `--direct` | | Alias for `--evidence strict` |
| `--min-params` | | Minimum effective parameter size in billions |
| `--profile` | | Ranking profile: `general` \| `coding` \| `vision` \| `math` \| `any` |
| `--details` | | Show Downloads metadata instead of runtime columns |
| `--json` | | Output as JSON |
| `--markdown` | `-m` | Output as GitHub-Flavored Markdown |
| `--refresh` | | Ignore cache and re-fetch models |
| `--gpu` | | Simulate GPU(s): `'RTX 4090'`, `'2x RTX 4090'`, or repeat flag |
| `--vram` | | Override VRAM in GB (requires `--gpu`) |
| `--vram-headroom` | | Reserve GPU memory: `auto` \| `none` \| `1GB` \| `10%` |
| `--ram-budget` | | RAM budget for offload: `available` \| `8GB` \| `50%` |
| `--cpu-only` | | Ignore GPU, run CPU-only mode |

## Commands

### `whichllm` (default — rank models)

Rank models by hardware fit. Outputs a table with tok/s estimates, VRAM usage, and quality scores.

```bash
# Basic: top 10 models for your hardware
uvx whichllm@latest

# Only full-GPU models, coding profile, top 5
uvx whichllm@latest --gpu-only --profile coding -n 5

# Simulate different GPU
uvx whichllm@latest --gpu "RTX 4090"

# Filter by quant
uvx whichllm@latest -q UD-Q4_K_M

# Markdown output for docs
uvx whichllm@latest -m --gpu-only
```

### `whichllm plan <model>`

Show VRAM requirements and quant options for a specific model.

```bash
# Default: what GPU do I need for qwen 3.5 9b?
uvx whichllm@latest plan "qwen 3.5 9b"

# Specific quant and context
uvx whichllm@latest plan "qwen 3.6 35b" -q UD-Q4_K_M -c 128k
```

### `whichllm hardware`

Show detected hardware info only.

```bash
uvx whichllm@latest hardware
```

### `whichllm upgrade <gpu...>`

Compare current machine against GPU upgrades. Shows quality/speed jump per option.

```bash
# Is upgrading from current GPU to RTX 4090 worth it?
uvx whichllm@latest upgrade "RTX 4090"

# Compare multiple GPUs
uvx whichllm@latest upgrade "RTX 4090" "RTX 5090" "H100"
```

### `whichllm run [model]`

Download and chat with a model. Auto-picks best if no model specified.

```bash
# Auto-pick best model for your hardware
uvx whichllm@latest run

# Specific model
uvx whichllm@latest run "qwen 3.5 9b" -q Q4_K_M
```

### `whichllm snippet [model]`

Print a ready-to-run Python script for a model.

```bash
uvx whichllm@latest snippet "qwen 3.5 9b"
```

## Profiles

| Profile | Use case |
|:---|:---|
| `general` (default) | Broad intelligence ranking |
| `coding` | Code generation, agentic coding |
| `vision` | Multimodal / image tasks |
| `math` | Mathematical reasoning |
| `any` | No profile filter |

## Speed Presets

| Preset | Meaning |
|:---|:---|
| `any` (default) | No speed filter |
| `usable` | Practical for interactive use |
| `fast` | High throughput only |

## Integration with local-model-autotuning

```bash
# 1. Find candidates
uvx whichllm@latest --gpu-only --profile coding -n 5

# 2. Plan specific model
uvx whichllm@latest plan "qwen 3.6 35b" -q UD-Q4_K_M -c 128k

# 3. Cross-check with SWE-bench / Aider before committing
# 4. Download GGUF, set MODEL in config.py, run autoloop
```

## Related Docs

- [`discover-models.md`](./discover-models.md) — end-to-end workflow using whichllm
- [`quantization-cascade.md`](./quantization-cascade.md) — how to pick the right quant format
