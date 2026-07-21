# Agent shell hard-gates

**Date:** 2026-07-21 (updated: full stack applied; ACL rejected as non-portable)  
**Audience:** operators + next coding agent + anyone who clones  
**Purpose:** What is installed, what it blocks, how a **human** turns it off. Clone-and-use — no OS admin rituals.

> **Agent rule:** If the user asks to disable / remove / revert these gates, **do not silently strip them**. Teach the steps in §3, wait for explicit “do it” / “desliga” / “remove”, then apply. Prefer the human running Cursor UI toggles themselves when possible.  
> Editing **wiring** under `scripts/hooks/**`, `.cursor/hooks.json`, `.claude/settings.json`, or `.cursor/rules/harness-trials.mdc` requires explicit user unlock — hooks deny those paths.

---

## 1. Inventory (live)

| Piece | Path | Role |
|---|---|---|
| Shell policy | `scripts/hooks/block-adhoc-eval.ps1` | cwd check; python allowlist; deny `-c` / raw llama / shell rewrite of gates |
| Gate-file policy | `scripts/hooks/block-gate-tamper.ps1` | Deny Write/Edit/Delete on wiring paths |
| Cursor wiring | `.cursor/hooks.json` | `beforeShellExecution` + `preToolUse` (all tools; script no-ops if no path), `failClosed: true` |
| Cursor soft rule | `.cursor/rules/harness-trials.mdc` | Always-on Trial = `config.py` + harness CLI |
| Claude wiring | `.claude/settings.json` | `permissions.deny` on gate paths + Bash python`-c` / llama\* ; `PreToolUse` Bash\|PowerShell + Edit\|Write\|Delete |
| Contract text | `AGENTS.md` + `scripts/AGENTS.md` | DOX pointers |

**Out of scope (by design):** OS ACL / `icacls` / chmod lockdowns / enterprise managed hooks. Clone users must not need admin filesystem rituals — only in-repo Cursor/Claude project hooks.

---

## 2. What the live gate blocks / allows

**Shell — blocked:**

- `python -c` / `--command`
- Any `python`/`py` not hitting allowlist entrypoints
- Scratch scripts e.g. `python .scratch\matrix.py`
- Direct `llama-cli` / `llama-server` / `llama-bench`
- `cwd` (or `cd` to absolute path) outside `workspace_roots` / `CLAUDE_PROJECT_DIR`
- Shell rewrite of gate paths (`Set-Content`, redirects, `Remove-Item`, …)

**Shell — python allowlist:**

- `benchmark_search.py`
- `autoloop.py`
- `-m pytest` / `-m unittest`
- `scripts\*.py` (operator scripts)

**Shell — allowed without python:** e.g. `nvidia-smi`, `git status`, listing files (still subject to cwd + gate-rewrite rules).

**Edit/Write/Delete — blocked paths:**

- `.cursor/hooks.json`
- `.cursor/rules/harness-trials.mdc`
- `.claude/settings.json`
- anything under `scripts/hooks/`

**Editable:** `docs/discovery/agent-shell-hard-gates.md` (this file), `AGENTS.md`, app code, `config.py`, etc.

---

## 3. How to DISABLE everything (rollback playbook)

Use when the user wants to “voltar atrás”. Do **one layer at a time**; reload Cursor / restart Claude Code after file changes.

### 3.1 Fastest — Cursor UI only (keeps files, stops enforcement)

1. Cursor Settings → Hooks → disable project hooks / untrust workspace.
2. Optional: disable rule `harness-trials.mdc` in Rules UI.

### 3.2 Full repo rollback (remove project enforcement)

**Disable hooks in UI first** (or explicit unlock) before an agent can delete wiring — otherwise Edit/Write on these paths is denied.

From repo root, after explicit user OK:

```powershell
Remove-Item -Force .cursor/hooks.json -ErrorAction SilentlyContinue
Remove-Item -Force .cursor/rules/harness-trials.mdc -ErrorAction SilentlyContinue
Remove-Item -Force .claude/settings.json -ErrorAction SilentlyContinue
# optional:
# Remove-Item -Recurse -Force scripts/hooks -ErrorAction SilentlyContinue
```

Then strip AGENTS.md / scripts/AGENTS.md hard-gate bullets (Edit tool).  
Reload Window + restart Claude Code.  
Smoke: `python -c "print(1)"` via agent should no longer be project-denied.

### 3.3 Git rollback

```powershell
git log --oneline -- .cursor/hooks.json .claude/settings.json scripts/hooks .cursor/rules/harness-trials.mdc
# revert or checkout prior revision of those paths
```

### 3.4 Claude permissions / yolo

- Removing `.claude/settings.json` drops `permissions.deny` + hooks together (current file shape).
- Yolo / `bypassPermissions` does **not** replace unwiring hooks. To run without gates, use §3.1–3.2.

### 3.5 OS ACL / enterprise

**Do not use for this repo.** Not supported. See §7.

---

## 4. Script for the next agent — “teach me to disable”

1. Confirm soft (UI) vs hard (delete files) vs git revert.  
2. Point to **this doc §3**.  
3. List inventory §1.  
4. Wait for explicit go-ahead.  
5. If wiring is still protected by hooks, tell the user to disable project hooks in Cursor UI **first**, then delete.  
6. Reload + verify.  
7. Offer to strip AGENTS.md bullets in the same change set.

---

## 5. Threat model (short)

In-repo hooks = strong friction, not a vault. Residual: user disables hooks; obfuscation; tools that omit `path` in payload. Clone-and-use wins over per-machine lockdowns.

| Control | Cursor | Claude Code |
|---|---|---|
| Block shell | `beforeShellExecution` deny | `PreToolUse` exit 2 |
| Block file edit | `preToolUse` deny | `permissions.deny` + PreToolUse |
| Soft ask | Flaky vs IDE allowlist | permissions ask |
| Sandbox OS | separate | not native Windows |

---

## 6. Yolo vs human

- Yolo skips approval prompts; it does not reliably skip exit-2 / deny hooks.  
- Wiring paths are denied to Edit/Write/Shell rewrite → agents **teach** §3; humans (or UI-disabled hooks) apply rollback.  
- **No OS ACL / icacls.** Anyone who clones gets the same in-repo hooks with zero machine setup.

---

## 7. What clone users get (portable)

On clone + open in Cursor (trusted workspace) or Claude Code:

1. Project hooks load from git.  
2. Soft rule `harness-trials.mdc` applies.  
3. No Windows/Linux permission commands required.

If hooks do not fire: trust the workspace / enable project hooks in Cursor Settings → Hooks; restart Claude Code. Non-devs only need that UI trust step.

**Explicitly not part of the product:** `icacls`, chmod lockdowns, enterprise `/etc/cursor` hooks, or any per-machine admin ritual.

---

## 8. Sources / Verification

- Cursor Hooks: https://cursor.com/docs/hooks — 2026-07-21  
- Claude Code Hooks / Settings: https://docs.anthropic.com/en/docs/claude-code/hooks , https://docs.anthropic.com/en/docs/claude-code/settings — 2026-07-21  
- Smoke (2026-07-21): deny `python -c`, scratch `.py`, llama-cli, foreign cwd, Set-Content gate; allow `benchmark_search.py`, `-m pytest`, `nvidia-smi`; deny Write `.cursor/hooks.json`; allow Write `README.md`.

---

## Open questions

- Whether every Cursor edit tool sends `path` / `file_path` in `preToolUse` (script allows when path missing — residual gap).
