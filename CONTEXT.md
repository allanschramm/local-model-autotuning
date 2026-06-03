# Domain Language (Ubiquitous Language)

This file defines the core concepts and boundaries for `autoresearch-public`. Use these terms exactly in all code, documentation, and communication.

## Execution and Runtime

*   **ServerIntent**: A pure data object describing the high-level runtime requirements for a benchmark (e.g., model name, context size, KV cache type). It is the only configuration object that crosses the seam into the runner.
*   **LlamaServerRunner**: A deep module responsible for translating a `ServerIntent` into a concrete `llama-server` subprocess. It encapsulates all hardware necromancy (MTP detection, MoE/VITRIOL offloading), environment orchestration (`LD_LIBRARY_PATH`), and process lifecycle (health checks, VRAM sampling).
*   **HardwareAdapter** (Internal): An internal abstraction within the runner used to query environment specifics (like `nvidia-smi` VRAM sampling), isolating OS/Hardware logic from the subprocess lifecycle.

## Evaluation Harness

*   **BenchmarkHarness**: A deep module that orchestrates the "Dual-Pass" execution policy. It owns the logic for sequencing Accuracy (Pass 1) and Throughput (Pass 2) measurements and calculating the final `val_score`.
*   **EvalTask**: A module defining a single evaluation unit. It contains the prompt template, ground truth, and logic to score a model's response.
*   **LlamaClient**: A deep module that handles HTTP communication, payload formatting, and retry logic for the `llama-server`.

## Orchestration

*   **Orchestrator**: A deep module (`run_grid.py`) that manages the multi-config execution loop and server lifecycles. It is responsible for booting the `LlamaServerRunner` once per configuration and executing a suite of benchmarks against it.
*   **Unified Benchmark Interface**: A standardized Python entry point (conventionally `run_benchmark`) provided by all benchmark modules. It accepts a `LlamaClient` and returns a `BenchmarkResult` dataclass, eliminating the need for CLI-based regex scraping.
