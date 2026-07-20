# `scripts/` — Utilities and Runner Scripts

## Purpose
Operator scripts for running setup health checks, monitoring GPU metrics, discovering candidate models, and managing the server daemon.

## Ownership
Repository operators and developers.

## Local Contracts
- Scripts must be runnable from the repository root.
- `setup-check.sh` is the canonical readiness verification script.

## Work Guidance
- Use `serve-config.py` as the preferred CLI helper to start/stop the llama-server daemon based on the mutable Baseline in `config.py`.
- Maintain helper commands documented in README.md.

## Verification
- Test script changes locally by executing them.
- Ensure `bash scripts/setup-check.sh` passes before declaring environment readiness.

## Child DOX Index
None

