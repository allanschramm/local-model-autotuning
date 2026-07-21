# Block agent Write/Edit/Delete on hard-gate files (human-only).
# Cursor preToolUse (Write|Edit|Delete) + Claude Code PreToolUse (Edit|Write|Delete).
# Deny = JSON + exit 2 (Cursor + Claude). Allow = JSON allow + exit 0.
#
# Disable / rollback: docs/discovery/agent-shell-hard-gates.md §3

$ErrorActionPreference = 'Stop'

$denyMsg = @'
BLOCKED: gate / hard-gate docs are human-maintained.

Protected paths (human unlock required to edit):
  .cursor/hooks.json
  .cursor/rules/harness-trials.mdc
  .claude/settings.json
  scripts/hooks/**

Playbook: docs/discovery/agent-shell-hard-gates.md section 3
(agent may teach / update that doc; must not edit wiring without explicit unlock).
'@

try {
    $raw = [Console]::In.ReadToEnd()
    if ([string]::IsNullOrWhiteSpace($raw)) {
        Write-Output '{"permission":"allow"}'
        exit 0
    }
    $payload = $raw | ConvertFrom-Json
} catch {
    Write-Output (@{ permission = 'deny'; user_message = $denyMsg; agent_message = "Gate-tamper hook JSON parse failed. $denyMsg" } | ConvertTo-Json -Compress)
    [Console]::Error.WriteLine($denyMsg)
    exit 2
}

$path = ''
if ($payload.file_path) { $path = [string]$payload.file_path }
if (-not $path -and $null -ne $payload.tool_input) {
    $ti = $payload.tool_input
    if ($ti -is [string]) {
        try { $ti = $ti | ConvertFrom-Json } catch { $ti = $null }
    }
    if ($ti) {
        if ($ti.file_path) { $path = [string]$ti.file_path }
        elseif ($ti.path) { $path = [string]$ti.path }
        elseif ($ti.target_notebook) { $path = [string]$ti.target_notebook }
    }
}

function Emit-Deny([string]$Message) {
    Write-Output (@{ permission = 'deny'; user_message = $Message; agent_message = $Message } | ConvertTo-Json -Compress)
    [Console]::Error.WriteLine($Message)
    exit 2
}

function Emit-Allow {
    Write-Output '{"permission":"allow"}'
    exit 0
}

if ([string]::IsNullOrWhiteSpace($path)) { Emit-Allow }

$norm = ($path -replace '\\', '/').ToLowerInvariant()

$protectedEnds = @(
    '/.cursor/hooks.json',
    '.cursor/hooks.json',
    '/.cursor/rules/harness-trials.mdc',
    '.cursor/rules/harness-trials.mdc',
    '/.claude/settings.json',
    '.claude/settings.json'
)

foreach ($suf in $protectedEnds) {
    if ($norm.EndsWith($suf)) { Emit-Deny $denyMsg }
}

if ($norm -match '(^|/)scripts/hooks(/|$)') { Emit-Deny $denyMsg }

Emit-Allow
