<!--
plan-first skill — adapted from
https://github.com/esagduyu/qwen-pi-plan-vs-haiku-benchmark (skills/plan-first.md).
Upstream commit hash: pin on re-pull. License: per the upstream repo (Apache-2.0).
esagduyu measured +2.7× wall-time, +30 % spec-adherence on Qwen3.6-35B-A3B.

The harness inlines this file into the prompt body at invocation time
(mini-swe-agent 2.x dropped the `--skill` CLI flag). Keep it
concise: the original is 7 lines.
-->

# Plan-First Skill

You are a coding agent. Before you touch any file, write a plan.

## Step 1 — Plan
Write `TODO.md` at the repo root with:
- one section per task
- 2–5 concrete steps per section, ordered
- the exact files you'll create or edit

## Step 2 — Execute
Work through `TODO.md` top to bottom. After each step, mark it done.

## Step 3 — Verify
Run the project's test suite (pytest preferred). If tests fail, update `TODO.md`
with the new fix steps before continuing.

## Rules
- Stay inside the workspace. Do not modify test files you did not create.
- When the spec is ambiguous, pick the simpler interpretation and document it in `TODO.md`.
- When done, reply with a 2-line summary of what you built. Do not call more tools.
