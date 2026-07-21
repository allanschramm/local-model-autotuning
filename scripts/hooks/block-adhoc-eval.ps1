# Agent hard-gate: shell policy for Trials / harness.
# Cursor beforeShellExecution + Claude Code PreToolUse (Bash|PowerShell).
#
# Policy:
#   1) cwd must stay under workspace_roots (when provided)
#   2) deny python -c / --command
#   3) deny raw llama-cli|server|bench
#   4) any python/py invoke must hit allowlisted entrypoints
#   5) deny shell rewrites of gate files
#   6) deny cd to absolute paths outside workspace_roots
#
# Disable / rollback: docs/discovery/agent-shell-hard-gates.md §3

$ErrorActionPreference = 'Stop'

$denyTrial = @'
BLOCKED by repo hard-gate (scripts/hooks/block-adhoc-eval.ps1).

Trial workflow:
  1. Edit autoresearch/core/config.py (one knob)
  2. Run: venv\Scripts\python.exe benchmark_search.py --desc "..." --no-agentic-quick --no-agentic-full --no-coding
     or:  venv\Scripts\python.exe autoloop.py --mode tps ...

Python allowlist only: benchmark_search.py | autoloop.py | -m pytest | -m unittest | scripts\*.py
Forbidden: Baseline knobs on benchmark_search.py, python -c, scratch .py Trials, raw llama-cli|server|bench, rewriting gate files via Shell.
'@

$denyCwd = @'
BLOCKED: shell cwd (or cd target) is outside the workspace.
Stay inside the repo root. See docs/discovery/agent-shell-hard-gates.md
'@

$denyGateShell = @'
BLOCKED: Shell must not create/overwrite/delete gate files.
Gate paths are human-maintained. Rollback playbook: docs/discovery/agent-shell-hard-gates.md section 3
'@

try {
    [Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $raw = [Console]::In.ReadToEnd()
    if ($null -ne $raw) {
        $raw = $raw.Trim([char]0, "`r", "`n", " ", "`t")
    }
    if ([string]::IsNullOrWhiteSpace($raw)) {
        Write-Output '{"permission":"allow"}'
        exit 0
    }
    $payload = $raw | ConvertFrom-Json
} catch {
    Write-Output (@{ permission = 'deny'; user_message = $denyTrial; agent_message = "Hook JSON parse failed: $_. $denyTrial" } | ConvertTo-Json -Compress)
    exit 2
}

$cmd = ''
if ($null -ne $payload.tool_input -and $payload.tool_input.command) {
    $cmd = [string]$payload.tool_input.command
} elseif ($payload.command) {
    $cmd = [string]$payload.command
}

$cwd = ''
if ($payload.cwd) { $cwd = [string]$payload.cwd }
elseif ($null -ne $payload.tool_input -and $payload.tool_input.working_directory) {
    $cwd = [string]$payload.tool_input.working_directory
}

$isClaude = ($null -ne $payload.tool_input) -or ($payload.hook_event_name -eq 'PreToolUse')

function Emit-Deny {
    param([string]$Message)
    if ($isClaude) {
        [Console]::Error.WriteLine($Message)
        exit 2
    }
    Write-Output (@{
        permission = 'deny'
        user_message = $Message
        agent_message = $Message
    } | ConvertTo-Json -Compress)
    exit 0
}

function Emit-Allow {
    if ($isClaude) { exit 0 }
    Write-Output '{"permission":"allow"}'
    exit 0
}

function Normalize-PathFlex([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return '' }
    try {
        return [System.IO.Path]::GetFullPath($p).TrimEnd('\', '/').ToLowerInvariant()
    } catch {
        return ($p -replace '/', '\').TrimEnd('\').ToLowerInvariant()
    }
}

function Get-WorkspaceRoots {
    $roots = New-Object System.Collections.Generic.List[string]
    if ($payload.workspace_roots) {
        foreach ($r in @($payload.workspace_roots)) {
            $n = Normalize-PathFlex ([string]$r)
            if ($n) { [void]$roots.Add($n) }
        }
    }
    if ($env:CLAUDE_PROJECT_DIR) {
        $n = Normalize-PathFlex $env:CLAUDE_PROJECT_DIR
        if ($n -and -not $roots.Contains($n)) { [void]$roots.Add($n) }
    }
    return $roots
}

function Test-UnderRoots([string]$path, $roots) {
    if (-not $path) { return $true }
    if ($roots.Count -eq 0) { return $true }
    $np = Normalize-PathFlex $path
    foreach ($r in $roots) {
        if ($np -eq $r -or $np.StartsWith($r + '\') -or $np.StartsWith($r + '/')) {
            return $true
        }
    }
    return $false
}

$gatePathRegex = '(?i)(?:^|[\s''"\\/])(?:\.cursor[/\\]hooks\.json|\.cursor[/\\]rules[/\\]harness-trials\.mdc|\.claude[/\\]settings\.json|scripts[/\\]hooks(?:[/\\]|\s|$))\b'
$baselineCliFlags = @(
    '--model', '--kv', '--kv-k', '--cache-type-k', '-ctk',
    '--kv-v', '--cache-type-v', '-ctv', '--max-tokens', '--ctx-size', '-c',
    '--threads', '-t', '--threads-batch', '--n-cpu-moe', '-ncmoe',
    '--ngl', '--n-gpu-layers', '-ngl', '--parallel', '--context-tokens',
    '--batch-size', '-b', '--ubatch-size', '-ub', '--flash-attn', '-fa',
    '--spec-type', '--spec-draft-n-max', '--spec-draft-model', '--no-mmap',
    '--jinja', '--reasoning-budget', '--reasoning-budget-message', '--reasoning',
    '--cont-batching', '--temp', '--top-p', '--min-p', '--top-k',
    '--repeat-penalty', '--presence-penalty', '--frequency-penalty',
    '--coding-task-limit', '--lcb-task-limit', '--bigcode-task-limit',
    '--bench-tts-threshold', '--grid', '--grid-kvs', '--grid-kvs-k',
    '--grid-kvs-v', '--grid-max-tokens', '--grid-threads',
    '--grid-threads-batch', '--grid-batch-sizes', '--grid-ubatch-sizes',
    '--grid-spec-draft-n-max'
)
$baselineCliFlagRegex = '(?i)(?:^|\s)(?:' + (($baselineCliFlags | ForEach-Object { [regex]::Escape($_) }) -join '|') + ')(?=\s|=|$)'

# Python entrypoint allowlist (after interpreter token)
$pythonAllowed = @(
    '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\b(?:\s+\S+)*\s+.*\bbenchmark_search\.py\b',
    '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\b(?:\s+\S+)*\s+.*\bautoloop\.py\b',
    '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\s+-m\s+pytest\b',
    '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\s+-m\s+unittest\b',
    '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\b(?:\s+\S+)*\s+.*\bscripts[/\\][\w.-]+\.py\b'
)

if ([string]::IsNullOrWhiteSpace($cmd)) { Emit-Allow }

$roots = Get-WorkspaceRoots
if ($cwd -and -not (Test-UnderRoots $cwd $roots)) {
    Emit-Deny $denyCwd
}

$n = ($cmd -replace '\s+', ' ').Trim()

# cd / Set-Location to absolute path outside roots
$cdMatches = [regex]::Matches($n, '(?i)(?:^|[;&|]\s*)(?:cd|Set-Location|chdir)\s+(?:-LiteralPath\s+|-Path\s+)?[''"]?([A-Za-z]:[\\/][^''";&|]+|/[^''";&|\s]+)')
foreach ($m in $cdMatches) {
    $target = $m.Groups[1].Value.Trim()
    if ($target -and $roots.Count -gt 0 -and -not (Test-UnderRoots $target $roots)) {
        Emit-Deny $denyCwd
    }
}

# Shell tamper of gate files
if ($n -match '(?i)(?:Out-File|Set-Content|Add-Content|New-Item|Copy-Item|Move-Item|Remove-Item|\bdel\b|\berase\b|\brm\b|\brmdir\b)') {
    if ($n -match $gatePathRegex) {
        Emit-Deny $denyGateShell
    }
}
if ($n -match '(?i)(?:>|>>).{0,120}(?:\.cursor[/\\]hooks\.json|\.cursor[/\\]rules[/\\]harness-trials\.mdc|\.claude[/\\]settings\.json|scripts[/\\]hooks)') {
    Emit-Deny $denyGateShell
}

# Inline python
if ($n -match '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\s+(?:-c|--command)\b') {
    Emit-Deny $denyTrial
}

# Baseline configuration must come from autoresearch/core/config.py.
if ($n -match '(?i)\bbenchmark_search\.py\b' -and $n -match $baselineCliFlagRegex) {
    Emit-Deny $denyTrial
}

# Raw llama binaries
if ($n -match '(?i)(?:^|[\s;&|\\/''"])llama-(?:cli|server|bench)(?:\.exe)?\b') {
    Emit-Deny $denyTrial
}

# Any python/py → must match allowlist
if ($n -match '(?i)\b(?:python(?:3)?|py)(?:\.exe)?\b') {
    $ok = $false
    foreach ($pat in $pythonAllowed) {
        if ($n -match $pat) { $ok = $true; break }
    }
    if (-not $ok) {
        Emit-Deny $denyTrial
    }
}

Emit-Allow
