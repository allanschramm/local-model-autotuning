# 02 — Hardware Detection Helper

**What to build:** Implement `detect_hardware_capabilities() -> dict` in `llama_runner.py` as the single source of truth for GPU/CPU hardware probing. Returns `{"has_gpu": bool, "physical_cores": int, "ram_mb": float}`. Uses existing `_is_gpu_working()` for GPU detection, `os.cpu_count()` with a logical-to-physical heuristic for core count (no `psutil` dependency), and platform-specific RAM detection (`wmic` on Windows, `/proc/meminfo` on Linux). Export from `autoresearch/core/__init__.py`. Tests mock GPU probes and verify correct output for CPU-only vs GPU hosts.

**Blocked by:** 01 — Config Surface: `N_GPU_LAYERS` + `NUMA`

**Status:** ready-for-agent

- [ ] `detect_hardware_capabilities()` implemented in `llama_runner.py`
- [ ] Returns correct `has_gpu`, `physical_cores`, `ram_mb` on current host
- [ ] No new dependency added (uses `os.cpu_count()` + platform commands)
- [ ] Exported from `autoresearch/core/__init__.py`
- [ ] Unit tests in `test_llama_runner.py` mock GPU probes and verify dict shape
- [ ] Full test suite passes
