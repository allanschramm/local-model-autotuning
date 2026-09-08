# Security Policy

## Reporting a Vulnerability

This repository has **private vulnerability reporting** enabled.

Use the GitHub interface to report privately:

1. Go to the **Security** tab
2. Click **Report a vulnerability**
3. Describe the issue, impact, and reproduction steps

Please do **not** open a public issue for suspected vulnerabilities.

## Scope

Focus areas for this repo:

- Shell command injection or path traversal in `scripts/` and `autoresearch/` runners (they execute local model runtimes and benchmark harnesses)
- Credential handling in downloaded model configs, HF tokens, or `.env` files
- The vendored WebUI under `local-model-autotuning/llama.cpp/tools/server/webui` — report upstream to llama.cpp instead unless the exposure comes from this repo's integration

## Supported Versions

Only `main` receives security fixes.
