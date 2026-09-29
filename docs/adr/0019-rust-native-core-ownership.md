# ADR 0019: Rust Native Core — process ownership and the anti-agent contract

**Status:** Accepted
**Date:** 2026-09-28
**Supersedes:** [ADR 0018](0018-rust-native-core-phase0.md) (strategy only; the
Phase 0 deliverable stands)
**Origin:** decisions from the wayfinder dossier
`.scratch/rust-native-refactor/` (tickets 01–18); **the plan of record is
`.scratch/rust-refactor/`** (tickets 01–13), which supersedes the dossier as
the tracker of execution.

## Context

The Python edition of `ailocal-model-autotuning` is decided as a
deprecation target. ADR 0018 recorded a **partial** refactor via PyO3, which
embeds Rust *inside* Python and leaves Python owning the loop, the clock and
the subprocess — paying marshalling on every call. That is the opposite of
what the current goal needs, so this ADR replaces the strategy.

Two goals, equally weighted. Neither is a performance threshold:

- **Move what the language allows** into a native binary. No minimum speedup
  was ever chosen. The wiki previously carried a ">3x measurable speedup"
  acceptance criterion that was **inherited from another source and never
  decided by Allan** — his answer was that any gain counts, from 1 ms to 1 h.
  That number is removed, not renegotiated.
- **Make the code hard for an AI agent to break.** This goal did not exist
  when ADR 0018 was written and appears in no prior ADR. It is what this ADR
  mostly records.

The second goal is the one with a real constraint: Allan chose *deprecate*
over *delete*, so for a window both editions exist, and during that window
the fragile path is still on disk. The contract below is honest about that.

## Decision

### 1. One native binary owns the process

`autoresearch-loop` is a **process**, sole owner of the loop, the clock, the
subprocess, preflight and `results.db`. **A single binary with subcommands**
(`run`, `validate`, `rank`, `bench`, `export`); crates remain code
boundaries, not deploy boundaries.

This is the most consequential decision in the map, because
`process_guard.py:227` uses `atexit` — which presupposes a process. In a
`cdylib`/PyO3 world that `atexit` belongs to the *interpreter*, and the whole
teardown surface changes shape. It does not: the teardown is the owner's
`Drop`.

`autoloop.py` survives as a **thin launcher** — argv and exit code, zero
logic. The E2E gate `benchmark_search.py --validation` becomes
`autoresearch-loop validate`; the gate's *contract* (complete a Trial, write
a row in `results.db`) is unchanged, only the invoker.

> **Recorded as an agent decision.** Allan declined to adjudicate one-binary
> vs multi-binary ("sei lá, como que vou saber isso?"). The agent took it on
> the reversibility criterion — splitting one binary into several later is
> cheap, merging is a rewrite — plus single-owner store. **Reverses if:** a
> cut requires `rank` or `bench` to run as independent processes during a
> Trial. Nothing in the block verdicts points that way today.

### 2. Anti-agent axis (a) — illegal state is unrepresentable

`ENGINE_*` / `SAMPLER_*` become a record of typed fields, with a closed enum
only where the value set is closed and a newtype where it is not. `CTX_SIZE`
becomes `CtxSize(NonZeroU32)` refusing `< 2048`; `KV_CACHE`/`KV_CACHE_K`/
`KV_CACHE_V` collapse into one `KvCacheSpec` whose constructor rejects the
illegal `k`-quant/`v`-other combination; `apply_pins` becomes total.

**Honest limit:** "never reduce `CTX_SIZE`" is a *temporal* property and
**no Rust type can express it** — a newtype has no memory of the previous
value. The type guarantees the floor; non-reduction is a single named
transition (`Baseline::narrowed_to`) plus the `AGENTS.md` rule plus a test.
This is weaker than "an agent that tries to reduce won't compile", and it
is the most the language delivers.

The frontier between type and runtime: **the value comes from the host, the
decision is typed.** No policy decision about a host-read integer is written
outside a type constructor.

### 3. Anti-agent axis (b) — single-owner store, atomic writes

The Rust binary is the only writer of `results.db`. `results.tsv` **stops
being a fallback** and becomes a deliberate `export` — because the fallback is
actively harmful, see the bug below. Readers (`ui/`, operator scripts) read
SQLite directly; `read_rows` no longer writes (it currently calls
`ensure_schema`, so a read can take a write lock and mutate the store).
`PRAGMA journal_mode=WAL` plus `sqlx` with compile-time-checked queries.
`recompute` becomes derivation at read time instead of a store-wide
read-modify-write after every Trial.

> **Live bug found while deciding this, in the current Python.** A failed
> *partial* `upsert_rows` triggers `try_sync_from_tsv`, which runs
> `replace_all` — `DELETE FROM trials` followed by INSERT from the TSV. If a
> Trial's DB write succeeded but its best-effort TSV append failed
> (`run.py:865-868` swallows it), the next recompute **deletes the row that
> only existed in the DB**. The heal is driven by the less authoritative
> source and runs in exactly the branches where the code had just established
> the DB as authoritative. The correct heal for a partial upsert is to retry
> the upsert. Separately, `parity_check` exists and **is never called** in
> production — its only caller is `scripts/rebuild_results_db.py:60`.

### 4. Anti-agent axis (c) — the containment gate

A fail-closed AST gate proving "a Python shim contains no logic". Prototype at
`.scratch/rust-native-refactor/prototypes/check_no_python_logic.py`,
self-test 4/4. Three rules: `shim-logic`, `shim-symbol-collision`,
`cut-surface-reappeared`. Every violation reports file, line, rule and a
**remediation path written for the agent that tripped on it**. Runs on
pre-commit, `rust-ci.yml` and `validate.yml`; needs no Rust toolchain.

> **The gate's first run against the real tree found 17 violations across all
> three shims.** The Phase 0 shims re-export *and re-implement*:
> `state.SearchState` is a full Python class wrapping the Rust one, and
> `fingerprint.py:53-58` states the Python was kept **deliberately so tests
> could `mock.patch` it**. Test-mockability was the reason logic stayed in
> Python. Consequences: `config.write_baseline` — the regex rewriter of the
> operator's `config.py`, the mechanism this whole refactor wants to remove —
> is reachable from the shim layer via `fingerprint.apply` and
> `SearchState.update_baseline`. The axis-(c) invariant is **already violated
> today**, so the debt is known and dated rather than open.

The gate also refuses to pass when it checks nothing (`gate-vacuous`) — the
prototype shipped with a wrong repo root once, found zero shims, and printed
"OK" with total confidence. A gate that looks at nothing is worse than no
gate.

### 5. The mutable Baseline becomes data, not code

`config.py` is gitignored Python that `write_baseline` rewrites **by regex
over its own source** (`config.py.example:302-315`). That is the kind of
mechanism an agent breaks without noticing, and a native binary owning a
Python source file is a permanent liability. The Baseline becomes a data
file owned by Rust (materialising the TOML ADR 0005 deferred);
`config.py` becomes a generated read-only shim. `validate_config` becomes
type invariants.

### 6. Engine: ephemeral port, `/health` readiness, Job Object teardown

- **Port: ephemeral, allocated by the harness**, using the same pattern
  upstream itself uses in router mode (`common_http_get_free_port`).
  `--port 0` is **rejected on evidence**: `llama-server` binds *before*
  loading the model (`server.cpp:462` vs `:475`) but only *reports* the port
  afterwards (`server.cpp:513`), so the harness would be blind during the
  whole load — the exact window the 300 s fail-closed timeout exists for.
  The report line is also prose that nothing parses.
- **Readiness: `/health` 503→200**, fail-closed at
  `SERVER_HEALTH_TIMEOUT_SECONDS = 300.0`. `is_ready` is a one-way latch, so
  200 means "startup finished", **not** "a model is resident" — it stays 200
  during `SLEEPING`.
- **Teardown: Job Object / `PR_SET_PDEATHSIG`**, total kernel equivalence.
  The name∩port orphan sweep **is removed** — it is what killed a live
  `llama-server` mid-Trial (`WinError 10061`, agentic score `0.0`) and, with
  ephemeral ports, can no longer enumerate targets anyway.
- **`EndpointDriftError`**, mirroring `FlagDriftError`, recorded as
  `INFRA_ERROR`, never `CODE_ERROR`. Upstream explicitly denies OpenAI
  compatibility, and the harness already reads ~13 endpoints.

### 7. The principle that decides every surface

> **Our adapter → Rust. Cloned external payload → stays original.**

`agentic_coding` (100% ours) goes to Rust outright. `agentic_runner`,
`mini_swe_agent` and `benchmark_coding` have our orchestration in Rust with
their vendored payload untouched. `sglang_runner` is a declared structural
exception: the engine *is* a Python program. The engine binaries themselves
are external and read-only by repo contract.

### 8. Honest scope

This ADR is a **decision record, not a plan of record**. It produces no port.
The deprecation window means the anti-agent objective is **partial while that
window lasts**: the gate prevents *reintroduction after a cut*, it cannot stop
an agent from writing new logic into a Python module during the window.

## Consequences

### Positive

- The teardown is owned, ordered and inspectable — not a best-effort `atexit`.
- The store gets a single writer, and the data-loss bug in §3 is closed by
  construction once the mirror stops being a fallback.
- The test-mockability argument for Python logic is retired: in Rust, tests
  construct the real type.
- No MSRV or edition change is required — Job Object and `PR_SET_PDEATHSIG`
  are available through the `windows` and `nix` crates directly, so
  `process-wrap` 10.x and `sysinfo` 0.39 are not needed and the
  `AGENTS.md` "update only with explicit Allan approval" rule is not
  triggered.

### Negative

- Phase 0 shipped ~2 581 LOC of Rust against ~12 655 LOC of Python
  production: about 17%, in the three most delicate modules. `classify`, the
  largest single surface, was **never ported at all**.
- Cutting a surface is not deleting a shim — it is porting the logic that
  lives in it first. The 17 measured violations size that work.
- Agentic axes bypass `LlamaClient.complete()` and rebuild their own payload;
  a port that reimplements only `complete()` silently loses the `tools`
  forwarding.
- Two external dependencies the map cannot control are registered, not
  solved: the upstream `CTRL_BREAK_EVENT` teardown handler (Windows), and
  the MSRV of the chosen `windows` crate version.

### Neutral

- The `claw-eval/` tree is absent from this worktree; the agentic E2E gate
  needs it present. Environment prerequisite, not an architecture blocker.
- `detect_physical_cores()` returns 16 where 8 physical is correct, because
  `wmic` no longer exists and the failure falls through to `os.cpu_count()`
  with no signal. Verified to be **display-only**: the count never reaches
  the search space, the Pareto axes, or the store. The real damage is an
  operator hand-seeding `THREADS=16` from a wrong printout.

## Connections

- Plan of record: `.scratch/rust-refactor/MAP.md` (13 execution tickets,
  each with a verifiable done-criterion). The decision dossier it grew out of
  is `.scratch/rust-native-refactor/` — kept for the terrain research
  (tickets 04, 10, 11) and the gate prototype.
- Gate prototype:
  `.scratch/rust-native-refactor/prototypes/check_no_python_logic.py`.
- Wiki entity `Rust-Refactoring-Strategy` (Obsidian) — state, updated
  2026-09-28 with the removal of the ">3x" criterion.
- Source ADRs preserved: 0005 (mutable Baseline), 0006 (Pareto nucleus),
  0010 (cross-platform zombie prevention), 0012 (basename Pareto point),
  0014 (Fingerprint bus), 0017 (rank membership quality-first).
