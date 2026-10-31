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

- Shell command injection or path traversal in `scripts/` and `autoresearch/` runners (they execute local model runtimes)
- Credential handling in downloaded model configs, HF tokens, or `.env` files
- The vendored WebUI under `llama.cpp/tools/server/webui` — report upstream to llama.cpp instead unless the exposure comes from this repo's integration

## Supported Versions

Security fixes land on `main`. Release tags are immutable snapshots; use the latest release containing the relevant fix.

## Upstream conversion dependencies

The llama.cpp source submodule is reference-only; tuning uses a compatible prebuilt runtime. Install this repository's root requirements.txt, not llama.cpp/requirements.txt. The tuning installation does not require Transformers or execute its remote custom-generation Python code.

On 2026-10-31, Dependabot reported the upstream conversion manifest's Transformers dependency under [GHSA-x9r9-c232-4q39](https://github.com/advisories/GHSA-x9r9-c232-4q39): remote generation code can be cached before trust consent. This upstream alert remains open; passing this repository's CodeQL checks does not resolve it. The clean-install E2E asserts that Transformers is absent from the tuning environment. If separately using upstream conversion tools, review their dependencies and this advisory in an isolated environment; they are outside this release's validated workflow.
