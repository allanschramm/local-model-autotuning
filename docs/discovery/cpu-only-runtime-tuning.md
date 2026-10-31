# CPU-Only Runtime Tuning: Bandwidth Ceiling and IQ-Preserving Knobs

> Scope: tuning a llama.cpp **CPU-only** rig (`-ngl 0`, no discrete GPU, unified RAM).
> Companion to [`cpu-inference-guide.md`](./cpu-inference-guide.md) (build flags / NUMA / allocators)
> and [`good-enough-tuning.md`](./good-enough-tuning.md) (the speed-search path).
> Findings measured on a hybrid 12-thread CPU (2 P-cores + 8 E-cores), dual-channel DDR4-3200, ~16 GB unified RAM.

## 1. The ceiling is memory bandwidth

On a CPU-only rig, **decode (token generation) throughput is memory-bandwidth-bound**:

```
decode t/s ≈ effective_bandwidth_GBs ÷ model_bytes_per_token
```

- Measure effective bandwidth as `decode_t/s × GGUF_size` — it stays roughly constant across model sizes (~23–24 GB/s on dual-channel DDR4-3200).
- So decode t/s scales inversely with model size (measured, Q4-class quants): 1.2B → ~33 t/s, 1.5B → ~23 t/s, 2.6B → ~17 t/s, 3B → ~12 t/s, 4B → ~7 t/s.

Corollary: **you cannot software-tune past the bandwidth ceiling.** Threads, batch, KV type, and allocators move *prefill* (and, on hybrid CPUs, a few % of decode) — they never double decode. An integrated GPU on such a rig shares the same DRAM, so offloading does not lift the decode ceiling either.

## 2. What changes IQ — and what does not

The most common misconception: batch size affects output quality. It does not.

| Knob | Changes IQ/PPL? | What it actually controls |
|---|---|---|
| `--threads` / `--threads-batch` | **no** | parallelism only |
| `--batch-size` / `--ubatch-size` | **no** | prefill throughput + memory |
| mmap / mlock / NUMA | **no** | memory layout |
| flash-attn | no (rounding noise only) | attention speed / memory |
| **KV cache quant** (`--cache-type-k/v`) | **slightly** (long ctx) | quantizes K/V states |
| **model quant** (Q4_0 vs Q3 vs Q8) | **yes** | quantizes weights |
| sampler (temp / top_p / top_k) | changes *behavior*, not IQ per se | decode distribution |

- Batch/ubatch only change **prefill** (how fast the prompt is ingested) — never the output. A coding task scores the same at `-b 32` or `-b 2048`.
- Batch-size floating-point tiling differences are ~1e-6 rounding noise — far below any measurable PPL/IQ shift.

## 3. Hybrid P+E thread split

On a hybrid CPU (performance + efficiency cores), decode and prefill want **different** thread counts:

- **Decode is bandwidth-bound** → fewer threads win (less contention).
- **Prefill is compute-bound** → more threads win.

Measured (2.6B Q4_0, dual-channel DDR4):

| threads (decode) | decode t/s |
|---|---|
| 4 | 14.1 |
| 8 | **17.0** |
| 12 | 15.0 |

Separate the two: `--threads 8 --threads-batch 12`. `--threads` sets generation threads; `--threads-batch` (`-tb` / `-tbd`) sets prompt-processing threads. Prefill measured 49.5 t/s at `--threads-batch 12` vs 48.3 at 8.

## 4. Sweeping batch size in one command

`llama-bench` takes comma-separated values and reports prefill (`pp`) and decode (`tg`) per combination:

```bash
llama-bench -m <model.gguf> -p 1024 -n 64 -t 8 \
  -b 512,1024,2048 -ub 128,256,512 \
  -ngl 0 -ctk q4_0 -ctv q4_0 -fa on -r 1 --no-warmup
```

- `-r 1` = one repetition; `--no-warmup` keeps a sweep short.
- Note: llama-bench has **no `-c` flag** — context = prompt (`-p`) + gen (`-n`) tokens.

Finding on this class of rig: **larger batches hurt prefill** — 512/128 won (~48 t/s) while 1024+ dropped to ~43. Decode stayed flat (~16.5 t/s) regardless, exactly as the bandwidth ceiling predicts.

## 5. IQ-preserving TPS hill-climb (PPL guard)

The repo's speed search already enforces a quality ceiling — see [`good-enough-tuning.md`](./good-enough-tuning.md):

- Each TPS-mode Trial runs **bench** (`llama-cli`) + **perplexity** (`llama-perplexity` over `data/perplexity_val.txt`).
- Guard: reject any config with `PPL > baseline × 1.01` (1% ceiling).
- Trials are minutes, not Claw hours.

**PPL vs KL:** the repo gates on PPL; KL is the same information — `KL(new‖base) ≈ ln(PPL_new / PPL_base)`, so a 1% PPL ceiling ≈ 0.01 nat KL. This is the same drift check people use when quantizing a model.

Use [bounded tuning](../../autoresearch/TUNING.md) with a finite Trial budget. Its PPL constraint prevents accepting throughput gains that exceed the fixed degradation limit; preserve the configured context.

## 6. OpenVINO on CPU-only (negative result)

On a CPU-only rig, OpenVINO GenAI did **not** beat llama.cpp on the same model:

| Engine | Model (same 1.5B, 4-bit) | decode t/s |
|---|---|---|
| llama.cpp CPU | Qwen2.5-1.5B Q4_K_M | 22.6 |
| OpenVINO CPU | Qwen2.5-1.5B int4 | 18.6 |

Also observed:

- The integrated GPU was **not exposed** to OpenVINO's GPU plugin (missing Level Zero/OpenCL driver stack) — and even if it were, it shares the same DRAM, so the decode ceiling is unchanged.
- A pre-converted `gemma-4-E4B` IR was rejected by `LLMPipeline` (5-input graph; GenAI expects a 3–4 input decoder signature).
- Pre-converted IR (`OpenVINO/<model>-*-ov`) loads without conversion, but only for standard architectures — non-standard families (e.g. Liquid `lfm2`) are not covered.

## Open questions

- **TBD:** whether `--no-mmap` / `mlock` moves decode measurably on small models (~±1–2% expected).
- **TBD:** prefill throughput for genuinely long prompts (32k+ tokens) — measured ~49 t/s at 1k tokens, but long-context prefill scales with the attention cost over the growing KV.
