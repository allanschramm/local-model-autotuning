# `tests/` — Test Suite Contract

## Purpose
Closed set of high-signal unit tests guarding what E2E harness runs miss: silent `results.db`/ranking corruption, security holes, platform branches the rig never executes, and fail-closed hardware gates. Primary verification is E2E (root `AGENTS.md` Testing rules).

## Ownership
Repository developers.

## Local Contracts
- Root `AGENTS.md` **Testing** rules are binding here: no unit tests after writing code; E2E (harness → `results.db` artifact) is the sole preferred mechanism; isolation tests only after writing down all failure modes, then the code.
- A unit test belongs in this directory only under one of four exceptions:
  1. **STORE-SILENT** — a bug would silently corrupt/mis-compute what lands in `results.db` or rankings (wrong score/status/merge/fingerprint with exit 0).
  2. **SECURITY** — path traversal/injection/auth holes happy-path E2E never probes.
  3. **PLATFORM** — `fcntl`/Unix-only branches the Windows rig's E2E never executes.
  4. **FAIL-CLOSED GATE** — hardware-safety decision logic where a bug fails open and can freeze/OOM the machine before E2E can report (VRAM/host preflight, keepout, single-load, circuit breaker, process-guard kill decisions). Assert the decision, not call-args plumbing.
- Everything else (CLI/argparse wiring, flag mapping, catalog metadata, mock-tests-the-mock, loud-failure guards) is out of scope — a broken version makes the harness fail loudly, which E2E catches.
- Every test file must follow `test_*.py` naming conventions and run under `pytest`.
- Always execute tests using the project virtual environment (`.\venv\Scripts\pytest.exe` or `.\venv\Scripts\python.exe -m pytest tests/`). Do not run system-global python/pytest.
- Mock external resources (CUDA/NVML libraries, llama-server instances) to ensure tests can run in CPU-only or restricted environments.
- Always mock `ctypes.CDLL` in VRAM/memory tracking tests to avoid querying host hardware directly.

## Work Guidance
- Run `.\venv\Scripts\python.exe -m pytest tests/` locally before completing tasks. All collected tests must pass.
- Do not add tests for new options, features, or regression bugs (root Testing rule) unless they fall under one of the four exceptions above — and then follow rule 3: write down the failure modes first.
- Platform-split code (`sys.platform`, `fcntl` vs `msvcrt`) under exception 3 needs a test that forces the other OS branch; Windows pytest will not execute the Linux `fcntl` path otherwise.
- Run `node --test teach/progress.test.js` for browser progress-contract changes.

## Verification
- All tests must pass with `.\venv\Scripts\python.exe -m pytest tests/`.

## Child DOX Index
None — `tests/` is a leaf directory.

> Per-test-file pointer (not enumerated above because `tests/` stays a leaf): `tests/test_mini_swe_agent.py` exercises the pinned DM-Code-Agent loader, single-tool-call mini-swe-agent config, per-task trajectory/no-progress abort, guarded subprocess timeout, scope rules, and suite fingerprint.
