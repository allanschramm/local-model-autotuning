# Rust native core — `rust/`

<!-- Scope: any agent editing the Rust workspace at `rust/`.
DOX hierarchy: README.md → local Cargo / PyO3 contracts → this AGENTS.md. -->

## Purpose

Rust workspace for the CPU-bound modules of `ailocal-model-autotuning`.
Phase 0 (closed 2026-09-27) lands the three hot-path modules behind a
PyO3 binding exposed as the `autoresearch_core` Python package. Phase 1
(additive) will add an `autoresearch-loop` binary that bypasses PyO3 in
the trial loop, plus the eventual port of `llama_runner.py` (1 323 LOC,
**T1 carry-over, Allan decision pending**).

## Ownership

- Repo root: `ailocal-model-autotuning` repo.
- Working tree branch: `allanschramm/rust` (separate worktree — see root
  AGENTS.md for cross-worktree isolation rules).
- Owner: Allan + Mavis development agents.
- Cargo workspace root: [`rust/Cargo.toml`](Cargo.toml).
- Subcrate: [`rust/autoresearch-core/`](autoresearch-core/) (pure-Rust
  core + PyO3 0.22 bindings + maturin 1.15 backend).

## Local Contracts

- **Edition 2021, MSRV 1.82, channel `stable`** — fixed via
  `rust-toolchain.toml`. Update only with explicit Allan approval.
- **Zero-ruptura (mandatory):** every Rust port keeps the Python public
  surface intact. The `autoresearch/core/*` shims in the parent repo
  re-export 100 % of the public symbols from `autoresearch_core.*`.
  `autoloop.py` and the rest of the harness must continue to work
  unchanged. Per-surface parity is gated by the Python test suite
  (`67 pytest` contract) and the parity smoke scripts.
- **`maturin develop --release`** is the develop-install flow. The wheel
  is built and installed into the active `.venv`. abi3-py311 means the
  wheel is reusable across CPython 3.11–3.13.
- **PyO3 submodule registration:** `PyModule::add_submodule` plus
  `sys.modules["autoresearch_core.<module>"] = submodule` is invoked
  from the lib's `#[pymodule]`. Python's
  `from autoresearch_core.pareto import …` resolves cleanly without an
  `__init__.py`.
- **Canonical JSON contract:** `pareto.fingerprint` hash payload is
  `{"engine": …, "sampler": …}` (dict, sorted keys, compact separators).
  Hash parity with the previous pure-Python implementation is byte-for-
  byte. The `scripts/smoke_bindings.py` 100-config sweep is the parity
  oracle; any change to canonicalization must update it together with
  this AGENTS.md.
- **Scrub messages verbatim:** `fingerprint::mismatch_reason` text must
  match the Python wording so existing pytest `match=` regex keeps
  firing.
- **`FingerprintError extends PyValueError`** — preserves the Python
  exception hierarchy.
- **No GPU/SKU/host fingerprint in source or tests.** Phase 0 kernel
  is CPU-only.
- **No vendor-tree edits inside this worktree.** `llama.cpp/`,
  `VITRIOL/`, `llama.cpp-releases/` remain external read-only.

## Work Guidance

### One-time setup

```powershell
# 1. Bootstrap venv (root AGENTS.md contract: never use system-global Python).
python -m venv .venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Install maturin into the venv (one-time).
.\venv\Scripts\python.exe -m pip install maturin==1.15
```

### Build, install, test (dev loop)

```powershell
# Rebuild and install the wheel into the active venv. Cwd must point at
# the subcrate; maturin 1.15 chokes on the cargo-workspace root (no
# `[package]` field there) and reports a TOML parse error.
cd rust\autoresearch-core
..\..\venv\Scripts\python.exe -m maturin develop --release
cd ..\..

# Rust kernel: unit + parity golden tests.
cargo test --all-targets

# Criterion isolated speedup (local only; not run in CI).
cargo bench --bench pareto -- --quick
cargo bench --bench fingerprint -- --quick
cargo bench --bench state -- --quick

# Phase-0 acceptance suite (Python-side parity, must remain 67/67).
cd ..
.\venv\Scripts\python.exe -m pytest \
    tests/test_pareto.py \
    tests/test_state.py \
    tests/test_fingerprint.py \
    tests/test_fingerprint_apply.py \
    tests/test_fingerprint_trial.py \
    tests/test_classify.py

# Smoke (E2E parity + import surface).
.\venv\Scripts\python.exe scripts\smoke_inspect.py
.\venv\Scripts\python.exe scripts\smoke_fp.py
.\venv\Scripts\python.exe scripts\smoke_bindings.py
```

### Workflow conventions

- New module file → put it under `src/<area>/<topic>.rs` with full
  `#[cfg(test)]` coverage first; only after parity does it earn the
  PyO3 binding in `src/py/<area>.rs`.
- Pure-Rust first, PyO3 second.
- New error class → use `#[pyclass(extends=PyValueError)]` or a
  similarly narrow exception subclass; never a generic `PyRuntimeError`.
- Reference upstream-first (root AGENTS.md): any new
  `ENGINE_*`/`SAMPLER_*` flag or estimator must first be checked
  against `llama.cpp/common/arg.cpp` `add_opt`. Phase 0 has no engine
  flags; Phase 1 will introduce them — keep that contract.

## Verification

- `cargo fmt --check` clean.
- `cargo clippy --all-targets -- -D warnings` (PyO3-internal `gil-refs`
  warnings remain tolerated — known upstream noise).
- `cargo test --all-targets` → 50/50 unit tests pass.
- `maturin develop --release` exits 0 with no warnings.
- `scripts/smoke_{inspect,fp,bindings}.py` all exit 0. The bindings
  smoke ends in `ALL OK` and a `100/100 hashes identical` line.
- `pytest` runs above stay 100 % green (67/67 today; grows with Phase 1
  acceptance surface).
- CI: `.github/workflows/rust-ci.yml` runs on `ubuntu-latest` and
  covers the kernel test + smoke + pytest chain. Local Windows is not
  CI-equivalent (POSIX-only branches in pytest); watch
  `rust-ci` for a green status before declaring a sprint done.

## Child DOX Index

- [`rust/README.md`](README.md) — build, install, and acceptance gates
  for Phase 0 (user-facing quickstart).
- [`rust/autoresearch-core/`](autoresearch-core/) — subcrate. Its own
  internal layout is documented in its `Cargo.toml`, `pyproject.toml`
  (maturin backend), and `benches/` criterion targets. No nested
  AGENTS.md yet — promote one only when the subcrate gains its own
  durable boundary or sprint cadence.

## See also

- [`docs/adr/0018-rust-native-core-phase0.md`](../../docs/adr/0018-rust-native-core-phase0.md)
  — Phase 0 decision record (cargo / PyO3 strategy, parity contract,
  carry-over list).
- `docs/sessions/2026-09-27-rust-phase0-criterion.md` — Phase 0 closure
  session log (per-call kernel numbers + PyO3 marshalling honest
  finding).
- Wiki entity `Rust-Refactoring-Strategy` (Obsidian) — strategy baseline
  + Phase 0 closure addendum.
