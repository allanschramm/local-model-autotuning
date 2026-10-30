# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-28-ornith-35b-ubatch-ot-ab.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Validate a community testimonial (community operator, 8 GB-class AMD rig, PCIe 3.0 host,
Vulkan/ROCm) on this rig:

1. Manual `-ot` regex tensor mapping (built by an LLM reading the GGUF headers) was
   claimed to lift prefill 170-200 → 260 t/s and decode 12-15 → ~17 t/s over
   `--n-cpu-moe`.
2. `--ubatch-size 6144` was claimed as the processing (prefill) lever.
3. "Even with spill to shared RAM it does not crash TPS" (WDDM spill tolerance).

Upstream code review predicted (1) is mechanically null: `--n-cpu-moe N` instantiates
`blk\.i\.` + `LLM_FFN_EXPS_REGEX` per block (`common/common.h:1142-1149`), `--cpu-moe`
applies the same regex globally (`common/arg.cpp:2782-2787`), and the testimonial's
remaining pins (`attn`/`shexp`/`ssm`/`ffn_gate_inp`/`token_embd`/`output` → GPU) are
no-ops under full offload. The A/B tested that prediction. Arms F* additionally test
the codacus expert-cache profile × ubatch interaction on the daily 35B config.

- `discrete_gpu` 8 GB-class; Baseline `VRAM_LIMIT_MB` 8000, keepout 512 → effective
  7676 MB; Windows; unified memory 32 GB-class. Desktop VRAM baseline drifted
  900 → ~1500 MB during the session (operator apps), eating cache-config margin.
- Engine arms A1-A3/B1/B2: upstream `b10549`. Arms F1-F3e: codacus `perf-0ac3d9b`.

- Model: `Ornith-1.5-35B-Q4_K_M.gguf` (20.2 GiB file). ctx 131072, KV q4_0/q4_0,
  flash-attn on, threads 6/6, cont-batching, ngl 99, `N_CPU_MOE` 41 explicit, mmap
  default.
- Probe (harness): `ServerIntent.from_config(config.load_config(), Path('models'), **OVR)`
  inside `LlamaServerRunner` — Baseline `config.py` untouched, every arm fully
  overridden. POST `/v1/chat/completions`, 4142-token prompt (fixed ~90-word paragraph
  × 40 + per-rep leading nonce so the prefix cache never matches), `max_tokens 256`,
  temp 0.6; 2 reps per arm; parse `print_timing` lines; peak VRAM from the runner
  sampler (process-scoped). Prompt-size note: 4142 tokens ⇒ at ub 6144 the whole
  prompt is ONE micro-batch (no chunking); the measured ubatch gain includes
  chunk-overhead amortization.

```
AUTORESEARCH_LLAMA_CPP_ROOT=<engine-root> ./venv/Scripts/python.exe -c "
from pathlib import Path
from autoresearch.core import config
from autoresearch.core.llama_runner import ServerIntent, LlamaServerRunner
OVR = dict(MODEL='ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf',
           CTX_SIZE=131072, N_CPU_MOE=41,
           BATCH_SIZE=B, UBATCH_SIZE=UB, THREADS=6, THREADS_BATCH=6,
           MOE_CACHE_PROFILE='models/traces/ornith-1.5-35b-merged.csv',  # F arms
           MOE_CACHE_SLOTS=<S>)                                          # F arms
intent, _ = ServerIntent.from_config(config.load_config(), Path('models'), **OVR)
with LlamaServerRunner(intent) as runner:
    ... 2x POST /v1/chat/completions (distinct leading nonce per rep) ...
    print(runner.peak_vram_mb, runner.peak_shared_mb)
"
```

B arms (re-run contract-compliant after an initial violation, see Errors): two TEMP
alias configs (`ot-ab-ub512`, `ot-ab-ub6144`) with the testimonial regex verbatim —
`--override-tensor ._exps..=CPU,.attn.=CUDA0,._shexp..=CUDA0,.ssm.=CUDA0,.ffn_gate_inp.=CUDA0,token_embd.=CUDA0,output.=CUDA0`
— plus engine/block/KV/batch flags matched to A arms, served via `scripts/model_up.py`
on `127.0.0.1:18081`; both aliases deleted after measurement.

Expert-cache trace regeneration (lost `models/traces/`, see Findings 9): the
`docs/discovery/low-vram-optimizations.md` workflow — `MOE_TRACE_OUT=<csv>
llama-moe-trace -m <model>.gguf -ngl 99 -ncmoe 41 -fa 1 -c 4096 -n 512 -p "<prompt>"`
for one code-ish and one chatty prompt (512 gen tokens each, ~28 s per run on the
pinned fork build), concatenated (headerless) into
`models/traces/ornith-1.5-35b-merged.csv` (49 360 rows).


| Arm | Engine | Placement / cache | ubatch | pp t/s (rep1/rep2) | tg t/s (rep1/rep2) | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | b10549 | `--n-cpu-moe 41` | 512 | 238.6 / 305.7 | 32.5 / 33.4 | 4318 MB |
| A2 | b10549 | `--n-cpu-moe 41` | 2048 | 445.4 / 716.9 | 32.4 / 33.6 | 4786 MB |
| A3 | b10549 | `--n-cpu-moe 41` | 6144 | 588.7 / 1172.9 | 32.7 / 32.9 | 6306 MB |
| B1 | b10549 | `-ot` regex | 512 | 232.2 / 300.3 | 32.1 / 32.3 | 4984 MB (device-wide, client sampler) |
| B2 | b10549 | `-ot` regex | 6144 | 570.8 / 1171.0 | 32.3 / 32.8 | 6549 MB (device-wide, client sampler) |
| F1 | perf-0ac3d9b | cache 48 slots — **failed to activate** (missing profile CSV) | 512 | 231.9 / 296.1 | 30.3 / 31.2 | 4725 MB |
| F1c | perf-0ac3d9b | cache 32 slots (regenerated CSV) | 512 | 202.5 / 237.3 | 34.3 / 35.7 | 6836 MB |
| F2 | perf-0ac3d9b | cache 32 slots | 6144 | — **VRAM_LIMIT_EXCEEDED** (7857 > 7676) at load | — | — |
| F3e | perf-0ac3d9b | cache 32 slots | 2048 | 341.3 / 534.9 | 34.9 / 35.1 | 6862 MB |


1. **`-ot` ≡ `--n-cpu-moe 41`: null result, as predicted** (contract-compliant B
   re-run). ≤ 2 % deltas at both ubatch points. The testimonial's placement gain did
   not replicate; upstream mechanics say it cannot — both flags are the same
   `LLM_FFN_EXPS_REGEX` override.
2. **ubatch is THE prefill lever on CPU-offloaded MoE (upstream):** warm pp 305.7 →
   716.9 → 1172.9 t/s (+283 %) while **tg stays flat (~33 t/s) across every arm** —
   decode is CPU-RAM-bandwidth bound on expert streaming; neither placement nor
   ubatch moves it.
3. ub 6144 VRAM cost (upstream, no cache): +1988 MB vs ub 512 (4318 → 6306 MB) —
   fits keepout with ~1.4 GB margin at ctx 131k / KV q4_0. Shared stayed ~76 MB
   (no spill).
4. **Cache × ubatch interaction (codacus fork):** cache 32 slots + ub 2048 fits
   (6862 MB, only +26 MB vs ub 512 — the fork's compute buffer grows far less than
   upstream's) and keeps the cache decode gain (35.1 vs 35.7 @ub512) while doubling
   warm prefill (534.9 vs 237.3). Cache + ub 6144 does NOT fit at 131k (7857).
5. Cross-read: plain upstream + ub 2048 (716.9 pp / 33.6 tg / 4786 MB) beats
   cache-32 + ub 2048 (534.9 / 35.1 / 6862 MB) on prefill and VRAM at −1.5 t/s tg.
   Cache is a decode-context tool; for prefill-heavy jobs plain upstream with a big
   ubatch is the better spend of the same VRAM.
6. **Daily alias degradation found:** `ornith-1.5-35b` (codacus fork, cache-only 48
   slots) referenced `models/traces/ornith-1.5-35b-merged.csv`, which was MISSING —
   the fork logs `cannot open profile ... expert cache disabled` and serves as plain
   fork control (~31 t/s instead of 36.9). `models/traces/` had been wiped by a
   `models/` cleanup sweep. Traces regenerated this session (code-ish + chatty
   prompts, 512 gen tokens each); profile is a READ input — the fork does NOT
   auto-create it.
8. Community decode 17 t/s vs ~33 here: consistent with its PCIe 3.0 host and
   mixed-size DIMM config, not tensor placement. The testimonial's own final config
   (`--cpu-moe`) is flag-for-flag the auto `--n-cpu-moe 41` this repo already runs.
9. Side findings: b10549 warns `tensor overrides to CPU are used with mmap enabled -
   consider using --load-mode none` (load-speed hint; untested). `--cache-reuse 256`
   is disabled by the server on the default unified-KV context (`cache_reuse is not
   supported by this context, it will be disabled`) — the harness forwards a no-op.
   Every `blk.40.*` tensor except the 4 `nextn` ones is unused/ignored at load —
   effective MoE blocks are 0-39; `block_count` 41 still covers everything.
10. **KV mixed K/V fit probe (Qwen3.8-4B-Distill, dense GDN hybrid, b10549, 4142-tok
    prompt, warm rep2):** q4_0/q4_0 @131k = pp 2764.7 / tg 71.2 / 5498 MB; q8_0/q8_0
    @131k = pp 2767.5 / tg 71.6 / 6527 MB — **whole-cache KV quantization costs zero
    speed** on this model. K q8_0 + V q4_0 MIXED = pp ~152-157 / tg ~30 at BOTH 131k
    and 65k (18× pp cliff, 2.4× tg cliff) with ZERO VRAM saving (5462/4641 MB — the
    GDN hybrid's KV is small) — a kernel-path fallback, not a memory tradeoff.
    Ornith-1.5-35B @65k reference point (cache-32 + ub2048): pp 555.4 / tg 36.9 /
    5991 MB.

- RAM preflight refusals (fail-closed, working as designed): first A1 launch
  (`free=21315 < need=22244`) — a killed eval kernel still held a ~20 GB GGUF
  memmap; reclaimed after. F3 series (`free≈21800-22214 < 22244`) — repeated 20 GB
  mmaps left standby/modified pages and the operator's desktop apps (browser,
  Discord) grew during the session. Note: `RAM_PREFLIGHT_MARGIN_MB` overrides must
  be patched into `config.DEFAULTS` in-process — `ServerIntent.from_config(**OVR)`
  does NOT reach the margin, which is read from module `DEFAULTS` at runner start.
- **Rule violation then correction (`use-harness-not-raw-llama`):** B arms were first
  launched as direct `llama-server.exe` supervised processes (no incident, peaks
  under keepout); per operator decision they were RE-RUN contract-compliant through
  TEMP alias configs + `model-up` (values above are the sanctioned re-run) and the
  aliases were deleted afterwards.

- ubatch 2048 is the keepout-compliant prefill upgrade on BOTH engines: upstream
  (no cache) 716.9 pp @4786 MB; codacus cache-32 534.9 pp @6862 MB with the decode
  gain kept. ub 6144 is VRAM-feasible only WITHOUT the expert cache.
- Cache trace CSV restored at `models/traces/ornith-1.5-35b-merged.csv` (regenerated
  with fresh prompts — coverage may differ slightly from the lost original).
- `-ot` placement recipes: not adopted. `N_CPU_MOE=None` (auto → GGUF block_count)
  stays; manual-regex placement adds nothing the stock flag lacks.
- Operator re-pointed the `ornith-1.5-35b` alias (2026-08-28, decode-first per
  operator rule: cache tg 35.1-35.7 > upstream 33.6): cache 32 slots + ubatch 2048,
  pp 534.9 @6862 MB. Upstream + ub 2048 (716.9 pp / 33.6 tg @4786 MB) stays
  documented as the prefill-first alternative.
- Qwen3.6-35B-A3B-MTP trace CSV regenerated (2026-08-28, operator go-ahead after
  WSL2 shutdown): `llama-moe-trace` x2 prompts via a venv-python driver that ran the
  repo's own `circuit_breaker.preflight_ram` (512 MB margin) + `GGML_CUDA_NO_PINNED=1`
  before spawning the fork profiler — the `model-up` path cannot launch a non-server
  binary, so the preflight-guarded driver is the compliant shape. Merged 49 360 rows.
  Repair verified by serving the alias via `model-up` (NO `cannot open profile`
  warning at startup) + one smoke request: tg 36.6 t/s @256 tokens (above fork
  control 28.5-31, in the MTP+cache band) — smoke evidence, not a measured vector.
  The earlier same-session ornith trace runs were raw supervised binaries (recorded
  as a rule deviation; this driver pattern supersedes it).
- Mixed K/V KV fingerprints: not adopted. On the GDN-hybrid arch the K q8_0/V q4_0
  combo hits a kernel cliff on b10549 — whole-cache q8_0 is the only fast precision

  bought with +1029 MB VRAM @131k. An IQ Trial on the mixed config is NOT justified
  — the fit probe rejected it on speed alone.

- Long-prompt (≥ 32k tokens) pp at ub 2048/6144: does the single-micro-batch
  advantage persist or taper once chunking returns?
- Regenerated trace quality: is 32-slot coverage from two 512-token prompts as good
  as the lost merged trace at 48 slots?

- [`2026-08-26-ornith-mtp-cache-ladder.md`](../sessions/2026-08-26-ornith-mtp-cache-ladder.md) —
  the 4-profile MTP × cache ladder this probe complements (35B control 31.0/232.1,
  cache-only 36.9/204.2 on the fork).
- [`docs/discovery/low-vram-optimizations.md`](../discovery/low-vram-optimizations.md)
  — expert-profile-cache workflow (trace capture + slot serving) used to regenerate
  the lost CSV.
- [`docs/llamacpp-flags-audit.md`](../llamacpp-flags-audit.md) — GPU row updated with
  the `-ot` ≡ `--n-cpu-moe` measurement; cache-reuse row carries the no-op caveat.
- Model cards: [`../models/ornith-1.5-35b.md`](../models/ornith-1.5-35b.md)
- Model cards: [`../models/qwen3.8-4b-distill.md`](../models/qwen3.8-4b-distill.md) — KV-quant rows + the mixed-K/V kernel cliff.
