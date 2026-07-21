# `tests/` — Test Suite Contract

## Purpose
Verification of all autotuning orchestrations, search strategy algorithms, configuration parsers, and client-server communication.

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

## Verification
- All tests must pass with `.\venv\Scripts\python.exe -m pytest tests/`.

## Child DOX Index
None — `tests/` is a leaf directory.
