# Spec — Tuning Dashboard (v1)

Read-only operator dashboard so a human can follow a coding-agent Search outside the terminal.

## Goal

Agent conducts Search (mutates Baseline, runs Trials via harness). Human opens the UI to watch model activity, Baseline, Trial history, and `llama-server` log — without start/stop controls.

## Non-goals (v1)

- Start/stop of agent, `autoloop`, or `llama-server`
- Editing Baseline / config / VRAM budget
- Agent session log tail
- Hill-climb charts, GPU live metrics
- Binding outside `127.0.0.1`
- Auth

## Canonical flows

1. Agent (or human) runs `python -m ui` after `pip install -r ui/requirements.txt`.
2. Open `http://127.0.0.1:18765`.
3. Page polls every ~2–3s for status, Baseline, last 50 trials, log tail.

## Surfaces

| Area | Source | Behaviour |
|---|---|---|
| Status | `llama_server.log` mtime/size growth (~10s) | Idle vs Em execução |
| Baseline | `.autoresearch_state.json` | MODEL + runtime flags |
| Trials | `results.tsv` | Last 50, newest first; minimal columns + diagnostic |
| Log | `autoresearch/runners/llama_server.log` | Tail of current/last Trial only |

## Stack

FastAPI + minimal HTML/JS. Deps in `ui/requirements.txt`. Code under `ui/` + `ui/AGENTS.md` (DOX). UI language: pt-BR.

## Decisions (grilling)

- Search operator = coding agent (`program.md`), not `autoloop`
- UI does not launch agents or processes
- Log = model/`llama-server` only, not agent reasoning
- Last Trial log only (harness rewrites file each Trial)
