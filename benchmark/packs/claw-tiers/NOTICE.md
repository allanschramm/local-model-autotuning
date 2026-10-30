# Claw tier snapshot

Claw-Eval (2026), MIT; see upstream/LICENSE.
Source: https://github.com/claw-eval/claw-eval
Pinned checkout includes local upstream integration. Selected task/fixture bytes are unchanged.
The live-web server is maintained as ../../claw_web.py with SSRF protection, bounded extraction and generic errors; adapter_sources records its original hash.
Adaptation: Pi tools, per-task ASGI process, legacy deterministic completion scorer and separate safety diagnostics/veto.
This is a single-trial legacy-selected slice, not the official Pass^3 or full upstream grader.
