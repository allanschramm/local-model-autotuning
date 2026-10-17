# `models/` — Shared GGUF store (harness)

## Layout (canonical)

Models live at `publisher/model/file.gguf`; the repo resolves by **basename**, so nested paths work:

```
models/
  <publisher>/<model>/<file>.gguf                  # canonical store layout
  local/<name>/<file>.gguf                             # other GGUFs
  draft/<file>.gguf                                    # speculative drafts (keep flat here)
  vision/mmproj-*.gguf                                 # multimodal projectors
  aliases/<name>/config.yaml                           # model-up launcher configs
```

Config / CLI still use short names: `MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` — harness finds nested path via `resolve_model_path`.

> **Store safety:** `models/` is a real directory. Recursive/forced deletes on the `models/` root are forbidden; per-file deletes inside `models/<publisher>/` are normal maintenance.

## Download rules

**Repo (`hf`):** put files under publisher/model, not the root:

```bash
hf download <org>/<repo> <file>.gguf --local-dir models/<publisher>/<model-name>
```

Example:

```bash
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models/lmstudio-community/qwen3.5-9b-gguf
```

Special dirs stay as-is: `--local-dir models/draft` for MTP/DFlash drafts; `--local-dir models/vision` for mmproj.

## Aliases (`model-up`)

**Local only — never commit.** Per-user `aliases/<name>/config.yaml`, optional `aliases/INDEX.md` / `REMOVED.md` for launcher notes, plus root `models/REMOVED.md` for SSD keep/delete inventory. Tracked docs carry **model** scores and Pareto picks (GGUF basename), not which alias name you use or what is on disk.

The local launcher (`models/aliases/model-up.cmd`, **never committed**) runs `scripts/model_up.py` with the **repo venv** (`venv/Scripts/python.exe`), falling back to bare `python` only when the venv is absent. The system interpreter has no `gguf`, and without it every MoE GGUF is misread as dense — `--n-cpu-moe` disappears and the experts spill into Shared GPU memory/host RAM. See `autoresearch/core/model_arch.py` `GGUFMetadataError`.

## Do not

- Drop main GGUFs in `models/` root — keep the `publisher/model/` layout.
- Put drafts under `publisher/model` unless you also update `SPEC_DRAFT_MODEL` paths.
- Commit alias names, ports, or personal `config.yaml` into the repo.
