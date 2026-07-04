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

## Approved Targets

`claw-eval/claw-eval` is approved as the next long-horizon agentic benchmark
target and is checked out as the root submodule `claw-eval/`. It covers 300
human-verified autonomous-agent tasks across `general`, `multimodal`, and
`multi_turn` splits, scored with Pass^3 consistency.

Do not execute it from this repo until these local integration pieces exist:

- local-model adapter for the existing llama.cpp/SGLang server endpoint
- fixed task subset policy for fair local comparisons
- runtime budget and sandbox policy
- result parser that maps Claw-Eval outputs into repo benchmark records

## Scoring Rule

During migration:

- Direct coding score remains a fast preflight.
- Agentic benchmarks become the main quality gate once at least one adapter is
  implemented.
- Use exactly 10 tasks per dataset for local model comparisons unless running a
  full-suite report.
- Never run local benchmark validation below 100k context.