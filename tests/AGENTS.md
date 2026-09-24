# `tests/` — Test Suite Contract

## Purpose
Verification of autotuning orchestration, search strategy, configuration, client-server communication, and student utilities.

## Ownership
Repository developers.

## Local Contracts
- Every test file must follow `test_*.py` naming conventions and run under `pytest`.
- Always execute tests using the project virtual environment (`.\venv\Scripts\pytest.exe` or `.\venv\Scripts\python.exe -m pytest tests/`). Do not run system-global python/pytest.
- Mock external resources (CUDA/NVML libraries, llama-server instances) to ensure tests can run in CPU-only or restricted environments.
- Always mock `ctypes.CDLL` in VRAM/memory tracking tests to avoid querying host hardware directly.

## Work Guidance
- Run `.\venv\Scripts\python.exe -m pytest tests/` locally before completing tasks. All collected tests must pass.
- Add tests for any new options, features, or regression bugs.
- Platform-split code (`sys.platform`, `fcntl` vs `msvcrt`) needs a test that forces the other OS branch; Windows pytest will not execute the Linux `fcntl` path otherwise.
- Run `node --test teach/progress.test.js` for browser progress-contract changes.

## Verification
- All tests must pass with `.\venv\Scripts\python.exe -m pytest tests/`.

## Child DOX Index
None — `tests/` is a leaf directory.

> Per-test-file pointer (not enumerated above because `tests/` stays a leaf): `tests/test_mini_swe_agent.py` exercises the pinned DM-Code-Agent loader, single-tool-call mini-swe-agent config, per-task trajectory/no-progress abort, guarded subprocess timeout, scope rules, and suite fingerprint.
