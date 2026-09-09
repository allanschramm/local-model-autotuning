# llama.cpp Flags Audit — Harness vs Upstream

**Source of truth:** `llama.cpp/common/arg.cpp:1424` `get_common_arg_defs()` (`~310 add_opt`), `common/fit.h:14`, `common/fit.cpp:178`, `tools/*/README.md`. **Do not add a new `ENGINE_*/SAMPLER_*` or harness gate without checking this doc + `arg.cpp` first.** Harness must be a thin `config -> llama.cpp arg` mapper; upstream already handles VRAM/KV/spec/threading/rope/sampling.

## Mandatory pre-implementation checklist

Before any flag or feature:

1. `grep -n "add_opt.*\"--<flag>\"" llama.cpp/common/arg.cpp` — if `~310` already has it, **forward it** in `llama_runner.py:895 _build_cmd` instead of re-implementing.
2. `grep -n "fit\|host" llama.cpp/common/fit.h common/fit.cpp` — fit is **VRAM-only** (`fit.h:14 "assumes system memory is unlimited"` via `ggml_backend_dev_memory`); host RAM needs our gate.
3. Check `docs/llamacpp-toolset.md` + `tools/fit-params` + `llama-bench` README for existing tooling.
4. Update this audit if you add a forwarding.

---

## Currently forwarded (~27/310) — `llama_runner.py:895` + `evaluation.py:268`

`--model -c/--ctx-size -b/--batch-size -ub/--ubatch-size -t/--threads --threads-batch --parallel -ngl/--n-gpu-layers --numa --cache-type-k/v --flash-attn --no-mmap --mlock --jinja --reasoning/--reasoning-effort/--reasoning-budget/message/--reasoning-preserve --cont-batching --cache-reuse --spec-type --spec-draft-n-max --spec-draft-type-k/v --spec-draft-model --n-cpu-moe` (+ bench `-p -n -r -o -fa -ctk/ctv -t -ngl`)

Everything else is untapped.

---

## Full matrix — Duplicate (delegate) vs Value-add (keep)

### Safety / Memory — upstream lacks host/WDDM awareness

| Upstream `arg.cpp` | Harness | Verdict |
|---|---|---|
| `--fit on/off` `:2813` `--fit-target MiB` `:2842` (default 1024) `--fit-ctx N` `:2866` `--fit-print` `:2827` `common_fit_params:fit.cpp:178` | `estimate_vram_mb:424` + `preflight_vram:488` static + `VRAM_LIMIT_MB 7900` `AGENTS.md:12` | **Delegate for exploration, keep reject for Trials.** `fit` mutates `c/ngl/tensor_split/tensor_buft_overrides` to meet `fit-target` via `llama_get_memory_breakdown`+`dev_memory`. Our reject is deterministic; `fit=on 256` (ornith snippet) would silently shrink `c=65536` but still `mmap 21.7GB` -> host OOM (fit ignores host per `fit.h:14`). Use `tools/fit-params` + `--fit-print` for manual sizing only. |
| `--load-mode none/mmap` `--mmap/--no-mmap` `--mlock` `--direct-io` `:2402` `--defrag-thold` | `NO_MMAP bool` `config:47` + `--no-mmap` `:941` `--mlock` `:943` | **Fix forwarding.** `load-mode` superset; `mmap+mlock` conflation locks 21.7GB on 16GB unified -> OOM. Wire `load-mode` passthrough, keep default `mmap`+`mlock False`. |
| `--warmup/--no-warmup` | Not wired. `THERMAL_WAIT True` `config:73` only | **Wire `--no-warmup`** for 8GB tight (golden rule: empty warmup OOM). Pair `THERMAL_WAIT` with `--no-warmup`. |
| ``--cache-ram`` (default ``8192 MiB``) ``--kv-unified``/``--no-kv-unified`` ``--no-kv-offload`` ``--repack``/``--no-repack`` ``--op-offload``/``--no-op-offload`` ``:1704,2210`` | `CACHE_REUSE 256` only, no `cache-ram`/`kv-unified`/`repack` | **Wire** — [prefix-cache-reuse](../discovery/prefix-cache-reuse.md) covers the three-layer chain (``--cache-prompt`` default on, ``--cache-ram`` default 8192 MiB, ``--cache-reuse`` harness override 256). Also: ``kv-unified`` saves ~512MB @65k, ``no-kv-offload`` keeps KV on CPU for MoE (0.7GB q4_0 vs 2.6GB f16), ``repack`` +3% tg (see `cpu-inference-guide.md:6`). **Measured caveat (2026-08-28, b10549):** the server **disables** `--cache-reuse` on the default unified-KV context (`cache_reuse is not supported by this context, it will be disabled`) — the harness forwards a no-op there. |
| `GPU: --tensor-split --split-mode layer/row --main-gpu --device --override-tensor --override-tensor-draft` | Only `--n-cpu-moe` | **Defer** — single-GPU today; `override-tensor "blk.40.*=CPU"` explicit alternative to `fit` regex `LAYER_FRACTION_MOE:434`. **Measured (2026-08-28, 35B):** manual `-ot` regex ≡ `--n-cpu-moe 41` within noise at ub 512/6144 — both compile to `LLM_FFN_EXPS_REGEX` instances (`common.h:1130/1142`); defer stands. **Flag semantics:** `--cpu-moe`/`-cmoe` `:2781` pushes the bare regex once (experts of ALL blocks → CPU) ≡ `-ncmoe <block_count>`; `-ncmoe N` `:2788` instantiates `blk\.i\.` + regex for the first N blocks only (`common.h:1142-1149`) — same tensor set, different block scope. Draft-model variants exist upstream: `--spec-draft-cpu-moe`/`--spec-draft-n-cpu-moe` `:4121-4137` (not wired). |
| ``--cache-prompt``/``--no-cache-prompt`` `:3536` | default on (`common.h:611`) | **Do not disable** — kills multi-turn agent KV prefix reuse. See [prefix-cache-reuse](../discovery/prefix-cache-reuse.md) §2. |

**Keep (no upstream):** `preflight_host_memory:662` `full GGUF+KV+draft` vs `RAM-max(6144,0.2*RAM)` unified / `max(4096,0.15*RAM)` discrete fail-closed `hardware.py:506`; `physical-512 keepout` + `SHARED 2048` kill `llama_runner.py:305,310` + `GGML_CUDA_NO_PINNED=1`; `free-at-start - headroom` clamp issue #10; `thermal` wait; `TPS_FLOOR/REPS`.

### Performance

| Flag | Current | Opportunity |
|---|---|---|
| `--cache-type-k/v f16/q8_0/q4_0/turbo2/3/4` `:1735` + `VRAM_QUANT_FACTORS:294` | `q4_0` only | `turbo3` `76t/s` vs `73t/s` gemma `results.tsv:94` measured; wire `turbo*` (tqp fork only, upstream lacks) |
| `--swa-full --ctx-checkpoints/--swa-checkpoints --checkpoint-min-step` `:1678` | Never | `swa-full` +15% long ctx gemma SWA (`min(ctx,window)` already fixed) |
| `--threads --threads-batch --threads-batch-draft --cpu-mask/--cpu-range/--cpu-strict --poll/--poll-batch --prio/--poll-batch-draft --threads-http` `:1439` | `THREADS 8` + `THREADS_BATCH 8` only | Pin `cpu-mask 0xFF` P-cores + `poll 50->100` + `threads-batch` tuning +5% tg on 5800X |
| `--numa distribute/isolate` | `NUMA None` wired | Already wired - use `isolate` for CPU-only `N_GPU_LAYERS 0` |
| `--spec-type mtp/draft-mtp/draft/dflash/ngram` `--spec-draft-n-max/min --spec-draft-p-min/--p-split --spec-draft-backend-sampling` `--spec-ngram-*` `--lookup-cache-static/dynamic` `:1613,2201` | `draft-mtp` auto MTP only (`b10549`) | `ngram` lookup `+8-20%` no draft VRAM; `p-min 0.8 p-split 0.9` `+10%` acceptance; wire `spec-ngram` for non-MTP |
| `--flash-attn on/off/auto` `:1735` | Hard `on` `validate:118` | `auto` lets llama.cpp disable on SWA mismatch - don't force `on` |

### Quality / Sampling / Context

| Flag | Current | Opportunity |
|---|---|---|
| `--rope-scaling linear/yarn --rope-freq-base/scale --yarn-* --grp-attn-n/w` | Never | `yarn-ext-factor 1.5` extends `65k->98k` for T053 `124k` overflow without new `CTX` |
| `--samplers --sampling-seq --temp/--top-p/--top-k/--min-p/--repeat-penalty/--presence/--frequency --dry-* --xtc-* --top-nsigma/--typical-p --dynatemp-* --mirostat` `:2004` | `TEMP/TOP_P/K/MIN_P/REPEAT/PRESENCE` 7 keys `config:81` | `dry 1.1` fixes code loops, `xtc 0.1` diversity, `top-nsigma` - free quality |
| `--reasoning-effort LEVEL` (arg.cpp:3717) joined `--reasoning-preserve` `:952` and `reasoning-budget/message` | Wired str (2026-08-29, `REASONING_EFFORT`, template-gated — only templates that read `reasoning_effort` react, e.g. Qwen3.8-27B; no-op on qwen35 family) | Keep `None=omit`; `True` agentic `0.8667` ornith costs 30% ctx |
| `--override-kv --override-tensor --lora --control-vector --grammar/--json-schema` | Never | `override-kv tokenizer.add_bos=false` parse fix; `lora` without retrain |
| `--cont-batching --cache-prompt --slots/--parallel --ctx-checkpoints` | `CONT_BATCHING False` `parallel 1` | Multi-slot `parallel 15` for throughput but single-slot deterministic for Trials |

---

## Prioritization

1. **Safety first:** wire `--load-mode`, `--no-warmup`, `--fit-print` (explore). Never set `mlock True` default on 21.7GB file.
2. **Perf (no new code):** `--cache-type turbo/repack/kv-unified/no-kv-offload/cache-ram` > `--spec-ngram` > `--poll/cpu-mask/prio` > `--swa-full` > `--flash-attn auto`
3. **Quality (free):** `dry/xtc/top-nsigma` > `yarn/grp-attn` for ctx > `override-kv/lora` as needed

Do not add `ENGINE_*` that duplicates `add_opt` - forward the `llama.cpp` flag and log it in this doc.

## References

- `llama.cpp/common/arg.cpp:1424` (~310 `add_opt`), `common/fit.h:14`, `common/fit.cpp:15,178,291`
- `autoresearch/core/config.py.example:29` `ENGINE_DEFAULTS`, `autoresearch/core/llama_runner.py:895 _build_cmd`, `autoresearch/core/hardware.py:506`, `autoresearch/core/llama_runner.py:662`
- `docs/llamacpp-toolset.md` (binaries), `docs/discovery/cpu-inference-guide.md:6` (`repack`), `docs/models/ornith-1.5-35b.md:12` (MTP), `results.tsv:94` (turbo)
- Release `b10867` (nightly CUDA 13.3) embeds MTP `nextn_predict_layers` and `spark2_5`
