# 01 — Config Surface: `N_GPU_LAYERS` + `NUMA`

**What to build:** Add `N_GPU_LAYERS` (default `-1` = Auto) and `NUMA` (default `None`) to `ENGINE_DEFAULTS` in both `config.py.example` and `config.py`. Update `validate_config()` to accept `N_GPU_LAYERS` values of `-1` (auto), `0` (CPU-only), and positive integers (GPU layers). Accept `NUMA` values of `None`, `'distribute'`, and `'isolate'`. Add `N_GPU_LAYERS` to `CORE_PASSTHROUGH` in `autoloop.py`. Tests confirm all valid and invalid combinations are handled correctly.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `N_GPU_LAYERS` and `NUMA` added to `ENGINE_DEFAULTS` in `config.py.example`
- [ ] `validate_config()` accepts `-1`, `0`, and positive integers for `N_GPU_LAYERS`
- [ ] `validate_config()` accepts `None`, `'distribute'`, `'isolate'` for `NUMA`
- [ ] `validate_config()` rejects invalid `NUMA` strings
- [ ] `N_GPU_LAYERS` added to `CORE_PASSTHROUGH` in `autoloop.py`
- [ ] Unit tests in `test_config_parsing.py` cover all new validation paths
- [ ] Full test suite passes (`.\venv\Scripts\pytest.exe`)
