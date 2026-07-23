# 03 — Fix Hardcoded `ngl=99` + Add NUMA to ServerIntent

**What to build:** Eliminate all hardcoded `ngl=99` sites that would silently override a CPU-only config. Three critical fixes: (1) `autoloop.py` `_defaults` dict must read `N_GPU_LAYERS` from config instead of hardcoding `99`, (2) `autoloop.py` `update_model_alias()` must read from `new_cfg['N_GPU_LAYERS']` instead of hardcoding `--n-gpu-layers 99`, (3) `ServerIntent.from_config()` must prefer `n_gpu_layers` from the normalized config dict. Add a `numa` field to the `ServerIntent` dataclass and emit `--numa <mode>` in `_build_cmd()` when set. Tests verify that `-ngl 0` and `--numa distribute` appear in the built command when configured.

**Blocked by:** 01 — Config Surface: `N_GPU_LAYERS` + `NUMA`

**Status:** ready-for-agent

- [ ] `autoloop.py` `_defaults["ngl"]` reads from `config.DEFAULTS.get("N_GPU_LAYERS", 99)`
- [ ] `autoloop.py` `update_model_alias()` reads `N_GPU_LAYERS` from `new_cfg`
- [ ] `ServerIntent.from_config()` reads `n_gpu_layers` from normalized config
- [ ] `numa` field added to `ServerIntent` dataclass
- [ ] `_build_cmd()` emits `--numa <mode>` when `intent.numa` is set
- [ ] Unit tests in `test_llama_runner.py` verify `-ngl 0` and `--numa` in built commands
- [ ] Unit tests in `test_autoloop.py` verify alias update uses config `N_GPU_LAYERS`
- [ ] Full test suite passes
