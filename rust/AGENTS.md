# Rust native core — `rust/`

<!-- Scope: any agent editing the Rust workspace at `rust/`.
DOX hierarchy: README.md → local Cargo / PyO3 contracts → this AGENTS.md. -->

## Purpose

Rust workspace for the CPU-bound modules of `ailocal-model-autotuning`.
Phase 0 (closed 2026-09-27) lands the three hot-path modules behind a
PyO3 binding exposed as the `autoresearch_core` Python package. Phase 1
will add a native `autoresearch-loop` binary that owns the loop, the
clock, the subprocess and the results store — see
[`docs/adr/0019-rust-native-core-ownership.md`](../docs/adr/0019-rust-native-core-ownership.md),
which supersedes the Phase 0 PyO3 strategy — plus the eventual port of
`llama_runner.py` (1 738 LOC, **T1 carry-over, Allan decision pending**).

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
  (67 tests across `test_pareto` / `test_state` / `test_fingerprint*`;
  **plus** `test_classify`, which is *not* Rust coverage — see below)
  plus 69 more in the three parity files listed under Verification.
- **The shims re-export AND re-implement (corrected 2026-09-28).** The
  zero-ruptura line above is true about *coverage* and misleading about
  *content*. `pareto.py`, `state.py` and `fingerprint.py` each define real
  Python on top of the Rust symbols: `pareto.AXES` + `class VectorLike`
  (`:31,34`), `class SearchState` in full (`state.py:24-67`),
  `def dump/load/apply/mismatch_reason/path_for/default_dir` plus
  `DEFAULT_DIR_NAME`, `SERVER_ENGINE_KEYS`, `HARNESS_ONLY_ENGINE_KEYS`
  (`fingerprint.py`). In `fingerprint.py:53-58` the docstring says the
  Python was kept **on purpose so tests can `unittest.mock.patch` it**, and
  `mismatch_reason` (`:181-191`) keeps a whole if/elif/else for that.
  Consequence: `config.write_baseline` — the regex rewriter of the
  operator's `config.py` — is reachable from the shim layer via
  `fingerprint.apply` (`:113-114`) and `SearchState.update_baseline`
  (`state.py:36-42`). Do not describe these modules as pure re-exports.
- **`classify` was never ported (corrected 2026-09-28).** The crate has no
  `classify`: `src/lib.rs:13-16` declares `error`/`fingerprint`/`pareto`/
  `state`, `src/py/mod.rs:6-8,17-30` registers three submodules, and the
  token `classify` appears zero times in the crate. `autoresearch/core/
  classify.py` is 258 lines of pure Python reaching Rust only transitively
  through the `pareto` shim. Consequently `test_classify.py` **cannot**
  gate Rust parity and is excluded from the parity count. Related: no test
  under `tests/` imports `autoresearch_core` at all — Rust coverage there
  is entirely indirect through the shims, so "67/67" measures *shim parity*,
  not direct Rust coverage.
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

- New module file → put it under `src/<area>/<topic>.rs`. The root
  `AGENTS.md` Testing Contract governs test-first here: prefer the E2E
  harness (`benchmark_search.py`, which ends in a `results.db` row) as the
  gate, and add `#[cfg(test)]` only where the `tests/AGENTS.md` closed set
  allows it — silent store/rank corruption, security holes, platform
  branches the rig never executes, and fail-closed hardware gates. A port is
  not done until the shim parity suite and the E2E Trial both pass; a unit
  test alone proves neither.
- Pure-Rust first, PyO3 second. A module earns its binding in
  `src/py/<area>.rs` only after the shim parity is green.
- New error class → use `#[pyclass(extends=PyValueError)]` or a
  similarly narrow exception subclass; never a generic `PyRuntimeError`.
- Reference upstream-first (root AGENTS.md): any new
  `ENGINE_*`/`SAMPLER_*` flag or estimator must first be checked
  against `llama.cpp/common/arg.cpp` `add_opt`. Phase 0 has no engine
  flags; Phase 1 will introduce them — keep that contract.

## Verification

- `cargo fmt --check` clean (run `cargo fmt` from `rust/autoresearch-core`).
- `cargo test` → 50/50 pass, no `unused` warnings. Keep it that way: dead
  imports are the only warning class this crate can realistically hold at
  zero.
- `cargo clippy --all-targets` is **not** clean and is not a gate. The
  remaining lints are PyO3-generated (deprecated `__pymethod_*__::SIGNATURE`
  constants, the `gil-refs` cfg name) plus doc-backtick nits in files this
  branch does not own. Read the output for *new* lints; do not gate on `-D
  warnings` until PyO3 is upgraded.
- `maturin develop --release` exits 0 with no warnings.
- `scripts/smoke_{inspect,fp,bindings}.py` all exit 0. The bindings
  smoke ends in `ALL OK` and a `100/100 hashes identical` line.
- `pytest` runs above stay 100 % green (67/67 shim-parity today, plus 69 in
  the three parity files; both numbers grow with the Phase 1 acceptance
  surface). Re-derive them from the suite rather than trusting this line.
- **E2E is the real gate.** A port is not done until
  `benchmark_search.py --validation` completes a Trial and writes a
  `results.db` row — the shim parity suite above proves the surface matches,
  but only the harness proves the decision path still admits a real model.
- CI: `.github/workflows/rust-ci.yml` runs on `ubuntu-latest` and
  covers the kernel test + smoke + pytest chain. Local Windows is not
  CI-equivalent (POSIX-only branches in pytest); watch
  `rust-ci` for a green status before declaring a sprint done.
  `validate.yml` covers the rest of the repo and also builds the wheel —
  the Python shims re-export the Rust extension, so a missing
  `autoresearch_core` fails collection for the entire suite there.

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
