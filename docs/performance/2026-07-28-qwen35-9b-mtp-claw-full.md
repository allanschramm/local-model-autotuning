# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-28-qwen35-9b-mtp-claw-full.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
# attempt 1 — limit 7900 → VRAM kill T004 @ 7906
# attempt 2 — limit 8000 → OK
```
