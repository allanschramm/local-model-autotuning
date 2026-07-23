# 07 — OpenVINO Guide + Benchmark Runner

**What to build:** A user-facing guide at `docs/discovery/openvino-guide.md` covering Intel OpenVINO GenAI for CPU/iGPU inference. Sections: OpenVINO model export from HuggingFace via `optimum-cli`, INT8/INT4 weight quantization, runtime configuration, and benchmark workflow. Create `scripts/bench_openvino.py` as a standalone benchmark runner that measures prefill and decode TPS using OpenVINO GenAI on Intel CPUs. Update `docs/discovery/AGENTS.md` Child DOX Index and `scripts/AGENTS.md` with the new entries.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `docs/discovery/openvino-guide.md` created with export, quant, and benchmark sections
- [ ] `scripts/bench_openvino.py` created as standalone benchmark runner
- [ ] Script gracefully handles missing `openvino_genai` dependency with clear error message
- [ ] `docs/discovery/AGENTS.md` Child DOX Index updated
- [ ] `scripts/AGENTS.md` updated with new script entry
