# ADR 0003: In-Process Benchmark Orchestration

## Status
Accepted

## Context
`run_grid.py` used `subprocess.run` to call benchmark scripts and regex-scraped their stdout. This was fragile and slow (rebooting the server for each script).

## Decision
Establish a **Unified Benchmark Interface** where all benchmarks export a `run_benchmark(client: LlamaClient)` function. Refactor `run_grid.py` to own the `LlamaServerRunner` lifecycle, booting it once per configuration and running all benchmarks in-process.

## Consequences
- **Positive**: Saved ~4 minutes of redundant server boot time per grid cell.
- **Positive**: Eliminated fragile regex scraping in favor of type-safe `BenchmarkResult` dataclasses.
- **Positive**: Errors bubble up through standard Python stack traces, improving locality.
