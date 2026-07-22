# `scripts/` — Utilities and Runner Scripts

## Purpose
Operator scripts for running setup health checks, monitoring GPU metrics, discovering candidate models, and managing the server daemon.

## Ownership
Repository operators and developers.

## Local Contracts
- Scripts must be runnable from the repository root.
- `setup-check.sh` is the canonical readiness verification script (supports GPU acceleration and CPU-only builds).
- `hooks/block-adhoc-eval.ps1` — shell policy (python allowlist, config-only Baseline, cwd, no gate rewrite).
- `hooks/block-gate-tamper.ps1` — deny Edit/Write/Delete on gate wiring paths.
- **Rollback:** [docs/discovery/agent-shell-hard-gates.md](../docs/discovery/agent-shell-hard-gates.md) §3.

## Work Guidance
- Use `serve-config.py` as the preferred CLI helper to start/stop the llama-server daemon based on the mutable Baseline in `config.py`.
- Use `build-llamacpp.py` (`python scripts/build-llamacpp.py --cpu` or `--cuda`) to build runtime binaries for local inference.
- Use `check_hardware.py` to diagnose local GPU/VRAM/RAM and give conservative GGUF/context guidance. Dense models must fit physical VRAM and use full GPU placement; never suggest partial dense offload.
- Use `verify_setup.py` to validate local API server health and benchmark real-time TPS. Its default port matches `serve-config.py` (18080).
- Maintain helper commands documented in README.md.

## Verification
- Test script changes locally by executing them.
- Ensure `bash scripts/setup-check.sh` passes before declaring environment readiness.

## Child DOX Index
- [hooks/block-adhoc-eval.ps1](hooks/block-adhoc-eval.ps1) — shell hard-gate.
- [hooks/block-gate-tamper.ps1](hooks/block-gate-tamper.ps1) — gate-file hard-gate.
