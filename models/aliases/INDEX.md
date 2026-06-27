# Model Aliases

Per-user aliases for the global `qwen-up` launcher. **Each user maintains their own `models/aliases/<name>/config.yaml`** — these are gitignored (paths are machine-specific).

This `INDEX.md` is **kept tracked as a template/schema reference**. Copy it to your own per-alias config or use it as a guide.

## Template: `models/aliases/<your-alias-name>/config.yaml`

```yaml
alias: <your-alias-name>           # OpenAI-compatible model name
model: models/<model-filename>.gguf  # relative path under repo root, or absolute
port: 18080                        # OpenAI-compatible port
host: 127.0.0.1
description: <one-line description>
flags:
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

Each flag string in `flags:` may contain a value (`--ctx-size 8192`); the launcher (`qwen-up`) splits with `shlex` before passing to `llama-server`.

## Commands

```bash
qwen-up                # start default alias (first found, or DEFAULT_ALIAS)
qwen-up <name>         # start a specific alias
qwen-up list           # list all aliases
qwen-up status         # health check of running server
qwen-up stop           # kill server
```

## Plug into a harness (Pi Agent, Hermes Agent, Claude Code)

After `qwen-up <name>`, point your harness at:

```yaml
base_url: http://127.0.0.1:18080/v1
model: <alias-name>     # matches the `alias:` field in the config.yaml
```

The launcher prints the exact connection string on success.

## Why a per-user alias system

The autoloop rewrites `autoresearch/core/config.py` on every keep — useful for automated search, painful for manual use. The alias system is the manual-use counterpart: pick a known-good config, name it, and start the model with one shell command from anywhere.

## Registered Aliases (Templates)

| Name | Model GGUF | Port | Baseline Score | TPS | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `ornith-1.0-9b` | `models/ornith-1.0-9b-Q4_K_M.gguf` | 18080 | 0.5800 | 52.2 | Full GPU Offload |
| `ornith-1.0-35b` | `models/ornith-1.0-35b-Q4_K_M.gguf` | 18080 | 0.5550 | 27.9 | VITRIOL (`--n-cpu-moe 36`) |

## See also

- [`../docs/discovery/discover-models.md`](../docs/discovery/discover-models.md) — full workflow including model selection.
- `qwen-up` launcher — installed at `~/.local/bin/qwen-up`. Source not in this repo; copy from your local install or write your own.
