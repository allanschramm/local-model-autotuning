# Live E2E check: running model-up server against a recorded coding Trial.
param([Parameter(Mandatory=$true)][string]$TrialArtifact, [string]$BaseUrl = 'http://127.0.0.1:18080')
$ErrorActionPreference = 'Stop'
$trial = Get-Content $TrialArtifact -Raw | ConvertFrom-Json
$props = Invoke-RestMethod "$BaseUrl/props"
$failures = @()
foreach ($key in @('temperature','top_p','top_k','min_p','repeat_penalty','presence_penalty')) {
    $expected = $trial.generation_params.$key
    $actual = $props.default_generation_settings.params.$key
    if ($null -eq $actual -or [Math]::Abs($actual - $expected) -gt 0.00001) {
        $failures += "$key expected=$expected actual=$actual"
    }
}
$actualModel = [IO.Path]::GetFileName($props.model_path)
if ($actualModel -ne $trial.model) { $failures += "Model mismatch: $actualModel" }
@{
    model = $actualModel
    context = $props.default_generation_settings.n_ctx
    expected_sampler = $trial.generation_params
    actual_sampler = $props.default_generation_settings.params
    failures = $failures
} | ConvertTo-Json -Depth 10 | Set-Content "$TrialArtifact.launcher-verification.json"
if ($failures.Count) { throw ($failures -join '; ') }
'PASS: live server model and sampler match Trial'
