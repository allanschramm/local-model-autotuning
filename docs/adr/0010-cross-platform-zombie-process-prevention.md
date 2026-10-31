# ADR 0010: Cross-Platform Zombie Process Prevention Architecture

**Date:** 2026-08-06
**Status:** Accepted

## Context & Problem Statement

Subprocesses launched during local LLM evaluations and benchmarks (`llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, `sglang`, and Python mock services) can become orphaned ("zombie" processes) if the parent Python runner exits abruptly due to unhandled exceptions, Ctrl+C, SIGKILL, or forced task termination.

When orphaned, these processes continue running in the background, holding CPU threads at ~50%+ capacity, locking VRAM allocations, and blocking TCP ports (`18080`, `28080`, `9100`–`9114`).

The solution must be cross-platform (Windows, Linux, macOS), contained strictly within Python stdlib/codebase (no external daemons or user harness requirements), and safe for user machines (avoiding collateral termination of unrelated personal LLM servers like LM Studio or Ollama).

## Decision

1. **OS-Native Process Guard (`autoresearch/core/process_guard.py`)**:
   - **Windows**: Bind spawned subprocesses to a Win32 Job Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The Windows kernel automatically kills all child processes if Python exits under any circumstance.
   - **Linux**: Use `PR_SET_PDEATHSIG` via `ctypes` (`libc.prctl(1, signal.SIGKILL)`), instructing the Linux kernel to send `SIGKILL` to child processes if the parent Python process dies.
   - **macOS / POSIX**: Spawn subprocesses in isolated process groups (`preexec_fn=os.setsid` / `process_group=0`) and execute group-wide signal termination (`os.killpg`) upon signal handlers (`SIGINT`, `SIGTERM`) or `atexit`.

2. **Pre-Flight Port & Process Sweep (`cleanup_leftover_processes`)**:
   - Before binding or launching a new server in `LlamaServerRunner.__enter__` or mock services, scan active TCP listeners on target harness ports (`18080`, `28080`, `9100`–`9114`) and check for matching process names (`llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, `sglang`).
   - Safely terminate any matching leftover processes before opening new candidate ports.

3. **Extensible Target List**:
   - Process name targets are configurable with defaults covering `"llama-server"`, `"llama-cli"`, `"llama-bench"`, `"llama-perplexity"`, and `"sglang"`.

4. **Graceful Teardown with Forced Fallback**:
   - Attempt a graceful termination (`terminate()`) with a 2.0-second timeout, allowing `llama.cpp` CUDA contexts to clean up.
   - If the process fails to exit within 2.0 seconds (e.g. frozen in host CPU loops), issue immediate forced OS kill (`kill()`).

5. **Domain Vocabulary**:
   - Add the **Process Guard** term to `CONTEXT.md`.

## Consequences

### Positive
- Zero zombie processes left behind on user machines across Windows, Linux, and macOS.
- Pre-flight checks prevent port collision errors (`EADDRINUSE`) and VRAM leaks between benchmark trials.
- Protects user's personal LLM servers (LM Studio, Ollama) by filtering strictly by harness ports and process names.

### Negative
- Requires platform-specific branches inside `process_guard.py`.

### Neutral
- Standard test suite (`pytest`) verifies process group and job object lifecycle management.

## References
- [ADR 0001: Deepen Llama Server Runner](0001-deepen-llama-server-runner.md)
- [ADR 0003: In-Process Benchmark Orchestration](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/adr/0003-in-process-benchmark-orchestration.md)
