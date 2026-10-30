# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-24-lcb-patch-gambiarra.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

* LFM first attempt hung (no subprocess timeout) — killed; timeout fix; re-ran OK.
* Ornith coding still requires **ctx 32k** on the operator host (65k VRAM-kill during coding).
