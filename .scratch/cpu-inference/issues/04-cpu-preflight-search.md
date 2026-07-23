# 04 — Autonomous CPU Preflight & Search Space

**What to build:** Make `autoloop.py` fully autonomous on CPU-only hardware. On startup, if `N_GPU_LAYERS == -1` (Auto), call `detect_hardware_capabilities()` and permanently rewrite config to `0` (no GPU) or `99` (GPU found) via `write_baseline()`. Add `NUMA` to `SEARCH_SPACE`. When `N_GPU_LAYERS == 0` is locked, remove GPU-only params (speculative decoding entries) from the active search space so the hill-climbing loop only explores CPU-relevant knobs (threads, NUMA, batch sizes, KV cache). The loop runs end-to-end on a CPU-only machine with zero human intervention.

**Blocked by:** 01 — Config Surface, 02 — Hardware Detection Helper, 03 — Fix ngl=99 + NUMA ServerIntent

**Status:** ready-for-agent

- [ ] `autoloop.py` startup detects `N_GPU_LAYERS == -1` and auto-seeds via `write_baseline()`
- [ ] `NUMA` added to `SEARCH_SPACE` with candidates `[None, 'distribute', 'isolate']`
- [ ] When `N_GPU_LAYERS == 0`, speculative decoding params excluded from active search space
- [ ] Unit tests in `test_autoloop.py` verify auto-seeding flow (mocked hardware detection)
- [ ] Unit tests verify CPU-mode search space excludes spec params
- [ ] Full test suite passes
