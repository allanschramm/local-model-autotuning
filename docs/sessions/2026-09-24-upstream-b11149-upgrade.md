# Session Log: 2026-09-24 — Upstream b11149 Upgrade (post-v0.5.0) + Submodule to v0.5.0

## Goal

Move the pinned upstream runtime from `upstream@b10867` (CUDA 13.3) to the latest
nightly `upstream@b11149` (CUDA 13.4, 3 commits past the `v0.5.0` tag), smoke-validate
it, and bump the `llama.cpp/` source submodule to the `v0.5.0` tag.

## Hardware

- Discrete 8 GB-class NVIDIA GPU (CUDA 8.9), Windows 11, 32 GB-class host RAM.
- Engine under test: `llama.cpp-releases/upstream/b11149` (nightly CUDA 13.4).

## Setup

1. `gh release list --repo ggml-org/llama.cpp` showed `v0.5.0` as Latest
   (2026-09-23) with nightly `b11149` published minutes later.
2. `gh release view v0.5.0 --json assets` returns only `nightly-tag.txt` —
   the versioned tag ships **no prebuilt binaries**. Its release page names
   nightly `b11146` as its build. Operator chose latest nightly `b11149`
   (CUDA 13.4 asset set) over the exact `b11146` counterpart.
3. Prior runtime `llama.cpp-releases/upstream/b10867/` kept on disk (fallback);
   `b10375` also remains.

## Commands (reproducible, repo-relative)

```powershell
gh release download b11149 --repo ggml-org/llama.cpp `
  --pattern "llama-b11149-bin-win-cuda-13.4-x64.zip" --dir <temp> --clobber
gh release download b11149 --repo ggml-org/llama.cpp `
  --pattern "cudart-llama-bin-win-cuda-13.4-x64.zip" --dir <temp> --clobber
Expand-Archive <temp>\llama-b11149-bin-win-cuda-13.4-x64.zip `
  -DestinationPath llama.cpp-releases\upstream\b11149\build-cuda\bin -Force
Expand-Archive <temp>\cudart-llama-bin-win-cuda-13.4-x64.zip `
  -DestinationPath llama.cpp-releases\upstream\b11149\build-cuda\bin -Force
Set-Location llama.cpp-releases\upstream\b11149\build-cuda\bin
.\llama-server.exe --version
.\llama-bench.exe --version
.\llama-server.exe --help | Select-String "fit-target|no-kv-offload|reasoning-effort|spec-type|cache-ram"
git -C llama.cpp fetch --tags origin
git -C llama.cpp checkout v0.5.0
git add llama.cpp
setx AUTORESEARCH_LLAMA_CPP_ROOT "<repo>\llama.cpp-releases\upstream\b11149"
```

## Findings

- `llama-server --version`: `0.5.0-dev (build 11149, commit d2e54583c)`,
  Clang 20.1.8 Windows x86_64.
- `llama-bench --version`: same build; CUDA backend loads from the release
  dir (`ggml-cuda.dll`, `ggml-rpc.dll`, CPU backend), device detected —
  CUDA 13.4 DLLs accepted by the installed driver.
- `--help` confirms the upstream-first surface: `--fit-target`,
  `--no-kv-offload`, `--cache-ram`, `--reasoning-effort`, and `--spec-type`
  now including `draft-dspark` (new in v0.5.0: Gemma4 DSpark draft backbone).
  Harness `SPEC_TYPE` values need verification before use (flags audit updated).
- `v0.5.0` tag = `7fe450e`; `b11149` = `d2e54583c` (3 commits later).
  Submodule `llama.cpp` now at `v0.5.0` (`7fe450e19`); prior pin was
  `50f068fff` (`b10679`-era).
- Tracked pointers updated: root `AGENTS.md` runtime line, `docs/llamacpp-toolset.md`
  (2 install examples), `docs/llamacpp-flags-audit.md` release line.
  Model cards / dated session logs still cite `b10867` as their measured engine —
  historical evidence, intentionally untouched.

## Errors

- First `gh release download` with two `--pattern` flags in one call fetched only
  the `llama-*` zip; the `cudart-*` zip needed a second identical call. Verbatim
  lesson: verify both zips on disk before extracting.
- First nested-`if ($?)` PowerShell one-liner failed with a brace parser error;
  rewrote as flat sequential commands.
- `2>&1` merges in PowerShell surface stderr lines as `RemoteException` noise;
  exit codes were 0 and `--version` output is authoritative.

## Decisions

- `b11149` over `b11146`: latest nightly, 3 commits past the `v0.5.0` tag, same
  `d2e54583c` the release page already points past.
- CUDA 13.4 asset set per operator choice (prior pin was CUDA 13.3); driver
  accepted it per smoke test.
- User-scope `AUTORESEARCH_LLAMA_CPP_ROOT` now `b11149`; Machine scope still
  references the old `b10375` path — left untouched (stale host default).
- `b10867` retained on disk as instant rollback; no model re-validation run —
  next Trial on the new engine re-establishes the frontier.
