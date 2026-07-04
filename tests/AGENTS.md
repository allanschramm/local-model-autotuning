# `tests/` — Test Suite Contract

## Purpose
Verification of all autotuning orchestrations, search strategy algorithms, configuration parsers, and client-server communication.

## Ownership
Repository developers.

## Local Contracts
- Every test file must follow `test_*.py` naming conventions and run under `pytest`.
- Mock external resources (CUDA/NVML libraries, llama-server instances) to ensure tests can run in CPU-only or restricted environments.
- Always mock `ctypes.CDLL` in VRAM/memory tracking tests to avoid querying host hardware directly.

## Work Guidance
- Run `pytest` locally before committing. All collected tests must pass.
- Add tests for any new options, features, or regression bugs.

## Verification
- All tests must pass with `pytest`.

## Child DOX Index
None
