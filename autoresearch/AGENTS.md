# `autoresearch/` — Core Autotuning Package

## Purpose
Core codebase containing search strategy optimization logic, llama.cpp server wrappers, API client integrations, and evaluation benchmark harnesses (Nexus, Claw, Coding).

## Ownership
Repository developers.

## Local Contracts
- Do not modify internal evaluation logic or benchmarks under `autoresearch/benchmarks/` without authorization.
- `autoresearch/core/config.py` owns immutable defaults and validation. The ignored `.autoresearch_state.json` is the only mutable Search state.
- Do not add hardcoded user or absolute directory paths in the source files.
- **Use the harness, not raw binaries**: Run `benchmark_search.py` or `autoloop.py` for evaluation. Do not invoke `llama-server` or `llama-bench` directly — the harness resolves paths, translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results.
- **Perplexity-Guided Tuning Guard**: When using `--perplexity-val` to maximize throughput (TPS), enforce a strict quality ceiling: any candidate configuration resulting in more than a 1% increase in perplexity (PPL) compared to the baseline must be discarded.

## Work Guidance
- Implement mock classes for system hardware calls (like GPU VRAM) to ensure code remains testable across environments.
- Keep dependencies minimal and avoid adding new third-party libraries.

## Verification
- Run `pytest` on tests checking core runners (`test_llama_runner.py`, `test_llama_client.py`).
- Run `pytest tests/test_search_strategy.py` for core optimization loop verification.

## Child DOX Index
None
