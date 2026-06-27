# AutoResearch Agent Onboarding Guide

Welcome! This guide helps you immediately bootstrap context on the AutoResearch project.

## Core Purpose

An autonomous hill-climbing optimization system for tuning local LLM runtime flags (KV cache formats, threads, batch sizes, context sizes) to maximize coding proficiency while keeping VRAM within limits.

---

## Codebase Map

| File/Path | Role |
| :--- | :--- |
| [autoloop.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoloop.py) | Autonomous hill-climbing runner. Mutates configs and loops forever. |
| [autoresearch/core/config.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/config.py) | **Only mutable file.** Holds current baseline configs. |
| [autoresearch/core/search.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/search.py) | Hill-climbing engine ([SearchStrategy](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/search.py#L14)). Evaluates improvement. |
| [autoresearch/core/llama_runner.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/core/llama_runner.py) | Wrapper around `llama-server`. Handles port collision and VRAM telemetry. |
| [autoresearch/benchmarks/benchmark_coding.py](file:///home/shark/workspace/Nexus-System/local-model-autotuning/autoresearch/benchmarks/benchmark_coding.py) | Evaluates coding capabilities via LCB, HE+, MBPP+, and BigCodeBench. |
| [results.tsv](file:///home/shark/workspace/Nexus-System/local-model-autotuning/results.tsv) | Tab-separated database recording trial history. |

---

## Local Rules & Work Contracts

1. **Be Terse**: Respond in smart caveman style (drop articles, filler, pleasantries).
2. **Loop Rule**: If running `autoloop.py` and a crash or code error occurs, **stop immediately**. Do not edit code to fix bugs during active search unless explicitly requested.
3. **No Pushing**: Never push results or config tweaks to remote branches. Keep all benchmark runs offline.
4. **GitNexus Rules**:
   - Run `impact` analysis on any symbol before modifying it.
   - Run `detect_changes()` before committing.
5. **DOX Framework**: Read the `AGENTS.md` hierarchy path to any file before touching it.

---

## Essential Commands

- **Run setup check**:
  ```bash
  bash scripts/setup-check.sh
  ```
- **Run test suite**:
  ```bash
  pytest
  ```
- **Run autonomous optimizer**:
  ```bash
  python autoloop.py
  ```
- **Run single manual trial**:
  ```bash
  python benchmark_search.py --desc "Hypothesis details here"
  ```
