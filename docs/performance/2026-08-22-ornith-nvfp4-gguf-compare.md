# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-22-ornith-nvfp4-gguf-compare.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```python
ENGINE_DEFAULTS = {
 'MODEL': 'Ornith-1.5-35B-Q4_K_M.gguf',
 'CTX_SIZE': 65536,  # Night floor; Day also test 32768
 'N_GPU_LAYERS': -1,
 'KV_CACHE_K': 'q4_0',
 'KV_CACHE_V': 'q4_0',
 'BATCH_SIZE': 512,
 'UBATCH_SIZE': 128,
 'FLASH_ATTN': 'on',
 'N_CPU_MOE': None,  # auto block_count (40 layers)
 'VRAM_LIMIT_MB': 7900,
 'TPS_FLOOR': 20.0,
}
SAMPLER_DEFAULTS = {
 # from Ornith-1.5 card + thinking: temp 0.6 / top_p 0.95 / top_k 20 for agentic; temp 1.0 for bench repro
 'TEMP': 0.6, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.05,
 'REPEAT_PENALTY': 1.0,
}
# + REASONING_PRESERVE=True via /props supports_preserve_reasoning (see 2026-08-19 log)
```

```bash
# WSL2, from repo root
source venv-vllm/bin/activate
VLLM_NVFP4_GEMM_BACKEND=marlin \
python -m vllm.entrypoints.openai.api_server \
  --model models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 \
  --served-model-name Ornith-1.5-35B-A3B-NVFP4 \
  --dtype float16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --trust-remote-code
```

Alternative SGLang (separate env, untouched):
```bash
source venv-sglang/bin/activate
python -m sglang.launch_server --model-path models/ornith-ai/Ornith-1.5-35B-A3B-NVFP4 --quantization modelopt_fp4 --context-length 65536 --mem-fraction-static 0.90 --trust-remote-code
## Errors / notes
- First `hf download` timed out at 300s (23.5G); resumed with 1800s timeout, completed. Dry-run now 0 files.
- `venv` (Windows) has no `vllm`/`sglang` — expected; WSL `venv-sglang` is SGLang-only per docs/sessions/2026-08-18-issue-59-sglang.md; `venv-vllm` created empty, **not installed yet** (no pip run per your GPU-in-use request — scaffold only).
- No Baseline edit, no server spawn per user request.
