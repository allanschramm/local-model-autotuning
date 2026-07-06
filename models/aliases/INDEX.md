# Model Aliases

Per-user aliases for the global `model-up` launcher. **Each user maintains their own `models/aliases/<name>/config.yaml`** — these are gitignored (paths are machine-specific).

This `INDEX.md` is **kept tracked as a template/schema reference**. Copy it to your own per-alias config or use it as a guide.

## Template: `models/aliases/<your-alias-name>/config.yaml`

```yaml
alias: <your-alias-name>           # OpenAI-compatible model name
model: models/<model-filename>.gguf  # relative path under repo root, or absolute
port: 18080                        # OpenAI-compatible port
host: 127.0.0.1
description: <one-line description>
flags:
  - --jinja               # required for Pi Agent (chat templates, tool calling, reasoning_content)
  - --ctx-size 8192
  - --n-gpu-layers 99
  - --n-cpu-moe 32
  - --cache-type-k q4_0
  - --cache-type-v q4_0
  - --flash-attn on
  - --threads 8
  - --threads-batch 8
  - --batch-size 512
  - --ubatch-size 128
metrics:                            # optional, for your own tracking
  tps: <measured-tokens-per-sec>
  measured_at: <YYYY-MM-DD>
  notes: <free-text>
status: ready                      # ready | untested | deprecated
```

Each flag string in `flags:` may contain a value (`--ctx-size 8192`); the launcher (`model-up`) splits with `shlex` before passing to `llama-server`.

## Commands

```bash
model-up                # start default alias (first found, or DEFAULT_ALIAS)
model-up <name>         # start a specific alias
model-up list           # list all aliases
model-up status         # health check of running server
model-down              # kill server (alias to model-up stop)
```

## Plug into a harness (Pi Agent, Hermes Agent, Claude Code)

After `model-up <name>`, point your harness at:

```yaml
base_url: http://127.0.0.1:18080/v1
model: <alias-name>     # matches the `alias:` field in the config.yaml
```

The launcher prints the exact connection string on success.

## Why a per-user alias system

The autoloop rewrites `autoresearch/core/config.py` on every keep — useful for automated search, painful for manual use. The alias system is the manual-use counterpart: pick a known-good config, name it, and start the model with one shell command from anywhere.

## Registered Aliases

| Name | Model GGUF | Port | Score | TPS | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `qwythos-9b` | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 18080 | **0.6400** | 50.9 | 10-task, b1024/ub256 |
| `o9` | `ornith-1.0-9b-Q4_K_M.gguf` | 18080 | 0.5450 | 49.8 | 10-task, b1024/ub256 |
| `o35` | `ornith-1.0-35b-Q4_K_M.gguf` | 18080 | 0.5550 | 31.5 | MoE, VITRIOL n-cpu-moe 32 |
| `qwen3.6-35b-q3xl` | `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | 18080 | — | — | Q3_K_XL 16 GB, MTP, n-cpu-moe 32. Untested. |
| `gemma-4-e4b` | `gemma-4-E4B-it-Q8_0.gguf` | 18080 | — | — | Q8_0 7.6 GB, near-lossless. Untested. |

## See also

- [`../docs/discovery/discover-models.md`](../docs/discovery/discover-models.md) — full workflow including model selection.
- `model-up` launcher lives in `models/aliases/`; add that folder to PATH on Windows.


