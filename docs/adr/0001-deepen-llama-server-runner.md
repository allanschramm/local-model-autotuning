# ADR 0001: Deepen LlamaServerRunner with ServerIntent

**Date:** 2026-06-03
**Status:** Accepted

## Context
The project relies on running benchmarks against local models using `llama.cpp`. Previously, the execution scripts (`benchmark_search.py` and `benchmark_search_claw.py`) manually constructed the complex command-line arguments for the `llama-server` process. This included determining optimal context sizes, handling `LD_LIBRARY_PATH`, and injecting hardware-specific optimizations like MTP detection and VITRIOL/MoE offloading. 

This resulted in a shallow `LlamaServerRunner` module, forcing callers to manage low-level details. It caused severe logic duplication across the benchmark scripts and leaked hardware logic across the execution seam.

## Decision
We will push all process orchestration and hardware configuration behind the `LlamaServerRunner` seam. 
- We introduce `ServerIntent`, a pure data object, to represent the high-level intent of the benchmark (e.g., `model_name`, `ctx_size`, `kv_cache`).
- `LlamaServerRunner` will accept a `ServerIntent` and internalize all logic related to building the `cmd` array, detecting hardware capabilities, and managing environment variables.
- We maintain the Python context manager (`with LlamaServerRunner(...)`) to ensure safe process teardown.

## Consequences
- **Locality:** Hardware optimization logic (MTP, VITRIOL) now lives in exactly one place.
- **Leverage:** Callers (benchmark scripts) are drastically simplified, replacing ~50 lines of brittle setup with a simple `ServerIntent` declaration.
- **Dependency Isolation:** The dependencies on the file system (to find the binary) and `nvidia-smi` are fully managed behind the seam.
