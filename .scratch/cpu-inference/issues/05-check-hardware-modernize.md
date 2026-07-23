# 05 — Modernize `check_hardware.py`

**What to build:** Replace the inline GPU/RAM detection logic in `check_hardware.py` with a call to the centralized `detect_hardware_capabilities()` from `llama_runner.py`. Print physical vs logical CPU core counts. Add SIMD instruction set hints (AVX2, AVX-512 detection via `cpuid` or platform commands where available). Recommend `-ngl 0`, `-t <physical_cores>`, and `--numa` flags for CPU-only systems. Keep the existing human-friendly output format.

**Blocked by:** 02 — Hardware Detection Helper

**Status:** ready-for-agent

- [ ] `check_hardware.py` imports and calls `detect_hardware_capabilities()`
- [ ] Duplicate GPU/RAM probe code removed
- [ ] Physical vs logical core count displayed
- [ ] SIMD detection hints printed (best-effort, no crash on unsupported platforms)
- [ ] CPU-only systems get `-ngl 0` and `-t` recommendations
- [ ] Script runs successfully: `.\venv\Scripts\python.exe scripts\check_hardware.py`
- [ ] Full test suite passes
