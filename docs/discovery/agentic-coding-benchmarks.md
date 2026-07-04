# Agentic Coding Benchmarks

This repo is moving from direct coding-only benchmarks toward long-horizon
agentic coding evaluation. The current scoring suite still runs direct
generation tasks: HumanEval+, MBPP+, LiveCodeBench, and BigCodeBench Hard.
Those remain useful as fast smoke tests, but they do not measure whether a
model can operate a terminal, inspect a repository, edit multiple files, and
recover from test failures.

## Current Code Hook

The catalog lives in:

```bash
python benchmark_search.py --list-agentic-benchmarks
```

This command only lists approved targets. If it prints only the header, there
are no approved agentic benchmark targets yet.

Real execution still needs an agent adapter that can:

- start a local model server with the existing llama.cpp/SGLang runner
- expose the model through an OpenAI-compatible endpoint
- let an external harness or local fixture drive a terminal or repository workspace
- collect pass/fail, elapsed time, token usage, and peak VRAM
- append results without modifying benchmark fixtures

## Candidate Direction

The next candidate to verify is `claw-eval/claw-eval`, because it is the ClawEval
project the user already runs here. Do not add it to the catalog until its local
command, runtime budget, task subset, and result parser are verified in this repo.

## Scoring Rule

During migration:

- Direct coding score remains a fast preflight.
- Agentic benchmarks become the main quality gate once at least one adapter is
  implemented.
- Use exactly 10 tasks per dataset for local model comparisons unless running a
  full-suite report.
- Never run local benchmark validation below 100k context.