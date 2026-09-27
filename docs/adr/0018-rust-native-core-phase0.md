# ADR 0018: Rust Native Core — Phase 0 (PyO3 + Three Modules)

**Status:** Accepted
**Date:** 2026-09-26

## Context & Problem Statement

The `ailocal-model-autotuning` loop spends a meaningful fraction of wall-clock time on three pure-Python modules in `autoresearch/core/` — `pareto.py` (4-axis Pareto nucleus), `state.py` (visited memory + atomic JSON), and `fingerprint.py` (private-data scrub, dump, load, mismatch gate). The harness already loads the autoloop under GIL, repeatedly runs `pareto.compute` per round, calls `fingerprint.mismatch_reason` per Trial gate, and persists `.autoresearch_state.json` per neighbor. Pure-Python implementations pay function-call overhead, dict-allocation churn, and serial regex passes — and they are not maintainable as a separate hot-path code base.

The wiki `[[Rust-Refactoring-Strategy]]` (last updated 2026-09-25) originally scoped a Rust port via PyO3 + maturin against the python autoloop. The plan was deferred ("ten times faster one module at a time"). This ADR formalizes the **Phase 0** slice: the three CPU-bound modules first, with byte-identical Python contract preserved through `autoresearch_core.{pareto,state,fingerprint}` submodules and `autoresearch.core.*` shims.

## Decision

We will add a Rust workspace at `rust/` with the following structure:

- `Cargo.toml` workspace, edition 2021, rust-version 1.82.
- `autoresearch-core/` crate — pure-Rust core + PyO3 0.22 bindings.
- Submodules: `pareto` (`ObjectiveVector`, `Trial`, `fingerprint_hash`, `dominates`, `pareto_set`, `merge`), `state` (`SearchState` with atomic JSON), `fingerprint` (`SCHEMA_VERSION`, `dump`, `load`, `mismatch_reason`, `path_for`, `FingerprintError`).
- Stack: `serde` + `serde_json` (canonical JSON), `pyo3` (submodule-registered), `sha2` (Fingerprint + Pareto hashes), `regex` (6 scrub rules ported 1:1), `criterion` (isolated speedup measurement). No `tokio`/`sqlx`/`reqwest` in Phase 0.

Strategies:

1. **Pure Rust first, PyO3 second.** Each module lands in `src/<mod>/*.rs` with full unit coverage, then gains a PyO3 binding in `src/py/<mod>.rs`. PyO3 is the surface contract; Rust is the implementation.
2. **Shim pattern (zero-ruptura).** `autoresearch/core/{pareto,state,fingerprint}.py` re-exports 100% of the public symbols from `autoresearch_core.{pareto,state,fingerprint}`. `autoloop.py` and every existing import continues to work unchanged.
3. **Submodule registration.** `PyModule.add_submodule` plus `sys.modules["autoresearch_core.<module>"] = submodule` is invoked from the lib's `#[pymodule]`, so Python's `from autoresearch_core.pareto import …` resolves cleanly without an `__init__.py` (maturin-generated).
4. **No storage migration.** `results.db` (SQLite canonical) and `results.tsv` (legacy export) stay Python-side. `apply()` keeps a thin Python wrapper around Rust `load()` + Python `config.write_baseline()` (the operator's mutable Baseline lives in `autoresearch/core/config.py`; the wiki marks TOML as deferred).
5. **Parity-first.** The `compact` Pareto hash payload is `{"engine": …, "sampler": …}` (matching the Python `dict` shape, not `[engine, sampler]`). `ObjectiveVector` is `Option<f64>` per axis. `FingerprintError` extends `ValueError`. Error messages preserve the Python wording verbatim so existing tests' `match=` regex still fires.

## Consequences

### Positive

- **67/67 pytest verde** in `tests/test_pareto.py`, `tests/test_state.py`, `tests/test_classify.py`, `tests/test_fingerprint*.py`, **67** total after Sprint 0.3; the operator sees no contract change.
- **50/50 Rust unit tests verdes** in `cargo test` for the pure-Rust layer.
- **Criterion isolated speedup** measured: `pareto::dominates` ~234 ps, `pareto::fingerprint_hash` ~3.76 µs, `pareto::pareto_set::1k` 239 µs, `pareto::merge::500` 42.9 µs (sample_size=50, --quick).
- **Zero migration for the operator** — `.venv/Scripts/python.exe -m maturin develop --release` then all existing commands work.
- **Pre-Phase 0 docs SQL-first** (`README.md`, `program.md`, `GOLDEN-RULES.md`) plus `scripts/check_no_tsv_readers.py` guard rail for the SQLite-first contract.

### Negative

- **Maturin requires a venv.** The develop flow needs `.venv`; a `python -m venv .venv` is a one-time setup cost in CI and locally (Cargo `b11149` is built abi3 so the wheel is reusable across CPython 3.11–3.13).
- **PyO3 0.22 cold-build is heavy.** First Cargo build downloads ~30 crates and compiles PyO3 macros (~10 s on this machine). Subsequent incremental builds are <2 s; CI cost is the same as the first build.
- **Scrub messages verbatim**: matching Python error wording becomes a soft contract. The fingerprint tests literally `match="absolute path"` / `"contact/host"` / `"GPU SKU"` / `"hostname"`. Any message rewrite must update tests.
- **PyO3 `#pyo3(name=…)` is silent:** the original `py_dump` exposed as `py_dump` because we forgot the rename. Fixed by `#[pyfunction(name = "dump")]`. New bindings must be wrapped with the rename attribute.

### Neutral

- **Cargo workspace branch `allanschramm/rust`** is the live worktree (Allan's worktree answer to the 17:45 map).
- **Carry-over T1–T7 from the 17:45 map are still open** (`llama_runner.py` port, parallelism, JSON-vs-PyO3 interface, TOML config, TSV export, CI matrix). T1/T3/T6 schedule to Phase 1; T4 stays Phase 2; T5 was confirmed irrelevant.
- **Open R1–R6 from this Phase 0 ADR remain documented** in `…/artifacts/plan.md` (rev. 2): PyClass attrs (mitigated via PyClass `frozen`), `ctx → Option<f64>` (consumers pass `float`, dataclass tolerated), canonical JSON byte-for-byte (verified via fixture), regex Unicode mode (rust regex crate uses Unicode per default), Windows-vs-POSIX path branches (`basename()` falls back to `Path::file_name` after `\\→/` normalization).

## Acceptance criteria (verified)

- ✅ `cargo fmt --check` clean
- ✅ `cargo clippy --all-targets -- -D warnings` (only pyo3-internal `gil-refs` warnings remain — known noise)
- ✅ `cargo test` → 50/50 unit tests pass
- ✅ `cargo bench --bench pareto -- --quick` → Criterion sample produced
- ✅ `maturin develop --release` → wheel built and installed; `from autoresearch_core.pareto import …`, `…fingerprint import dump, load, …`, `…state import SearchState` all resolve
- ✅ `pytest tests/test_pareto.py tests/test_state.py tests/test_fingerprint.py tests/test_fingerprint_apply.py tests/test_fingerprint_trial.py tests/test_classify.py` → **67 / 67 pass**

## Connections

- Plan: `…/artifacts/plan.md` (rev. 2 with sprint ordering by gain).
- Wiki: `Obsidian Vault/wiki/entities/Rust-Refactoring-Strategy.md` (strategy baseline).
- Wiki: `Obsidian Vault/wiki/entities/AILOCAL-Model-Autotuning.md` (project context).
- Source ADRs preserved: 0005 (mutable Baseline), 0006 (Pareto nucleus), 0012 (basename Pareto Point), 0014 (Fingerprint bus), 0016 (Morris screen).

## Implementation tracking

| Sprint | Module                                | Status   | Tests          |
|--------|----------------------------------------|----------|----------------|
| 0      | Pre: SQLite-first docs + guard rail   | ✅       | (manual)       |
| 1      | Workspace Rust + 28 unit tests        | ✅       | 28/28          |
| 2      | `fingerprint` port                     | ✅       | 16/16 + 34 pytest |
| 3      | `state` port                          | ✅       | 4/4 + 2 pytest  |
| 4      | `pareto` port                          | ✅       | 17/17 + 11 pytest |
| 5      | CI Ubuntu + ADR + wiki                 | pending  | TBD           |
| 6      | Allan handoff + speedup report        | pending  | TBD           |
