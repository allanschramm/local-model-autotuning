# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-06-19-mtp-baseline.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

1. **MTP-GGUF embeds MTP tensors** (`qwen35moe.nextn_predict_layers = 1`).
2. **File swap alone ≈ 2× TPS** without `--spec-type mtp`. Likely (a) inactive MTP tensors help load path, or (b) different quant profile vs base — not isolated.
3. **Autoloop auto-detect misses** filenames without "MTP". Fix: rename or set `spec_type: "mtp"`.
4. **Qwen3.6 thinking** can consume tokens with empty visible content — use `enable_thinking: false` or larger `max_tokens`.
5. **turboquant flag**: `--spec-type mtp --spec-draft-n-max 2` (not `draft-mtp`).

- Model path: `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (repo-relative).
- turboquant MTP flag: `--spec-type mtp` (not `draft-mtp`).
- TPS method: 5 runs × 100 tokens, mean; discard warmup if needed.
- Ask before downloads / long GPU runs; "prepare the command" ≠ "execute".

1. Do not equate external-draft speculation with MTP.
2. Do not download large GGUFs or start inference without explicit OK.
3. Validate MTP filename / `spec_type` before assuming auto-detect.
4. "No need to confirm" in repo context ≠ permission to run resource-heavy commands.

```bash
# start llama-server detached with MTP-GGUF + baseline flags
# wait for ready
# run 5× chat completions (100 tokens), average tok/s
# stop server
```

- Add `--spec-type mtp --spec-draft-n-max 2`.
- Expected: +1.15-1.25× on MoE → ~26-28 tok/s.
- Optional later: `--no-mmap`, `--n-cpu-moe` sweep, `--mlock` last (OOM risk).
