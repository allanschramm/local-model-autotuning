# Prefix-Cache / KV-Cache Reuse (Multi-Turn Sessions)

## 1. TL;DR

`llama-server` re-uses the KV cache along **three independent layers** — A) an
in-slot prefix match inside the active `llama_context`, B) a host-RAM
hot-swap pool that parks idle slots' KV state in regular RAM and re-loads it
on demand, and C) a block-shift ("shift-and-extend") reuse that walks the
prefix in `N`-token chunks when a partial overlap exists. The default chain
on upstream master is **all three on by default at their stated
defaults**: `cache_prompt = true`, `cache_idle_slots = true` (requires
`--cache-ram`), `n_cache_reuse = 0` (disabled — the **harness** in this repo
overrides layer C to `256` via `scripts/serve-config.py:166-169`). The
default-on chain is the vLLM `--enable-prefix-caching` / SGLang
`RadixAttention` analogue; it is what makes Hermes / Claw / coding-10
multi-turn agent sessions fast on the 2nd turn and beyond.

---

## 2. Three layers of cache reuse

The three flags are independent and stack. None of them on its own is "the"
prefix cache.

| Layer | Flag(s) | What it does | Upstream default | Where defined |
|---|---|---|---|---|
| **A** | `--cache-prompt` / `--no-cache-prompt`, `-sps N` | In-slot prefix match — when a new request tokenises to a leading prefix already cached inside an active slot's KV, only the unseen suffix is evaluated. | `cache_prompt = true`; `-sps = 0.5` | `common/arg.cpp:3536-3540`, `common/common.h:611` |
| **B** | `--cache-ram MiB`, `--cache-idle-slots` / `--no-cache-idle-slots` | Host-RAM hot-swap — when a slot finishes, its KV state is **offloaded to regular RAM** (up to `--cache-ram` MiB) and re-loaded on the next matching request, even if the slot is currently busy. | `--cache-ram = 8192` (8 GiB); `cache_idle_slots = true` | `common/arg.cpp:1716-1725`, `common/common.h:615`, [PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391) |
| **C** | `--cache-reuse N` | Block-shift reuse — when neither A nor B can match the whole prefix, walk the prefix in `N`-token chunks and reuse whatever aligns (KV-shifting the cached portion to the start of the new context). | `n_cache_reuse = 0` (**disabled** upstream) | `common/arg.cpp:3541-3551`, `common/common.h:610`, [Tutorial — discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606) |

The default-on chain is: **A always on, B on for any deployment that sets
`--cache-ram` (which is the upstream default value, so effectively on
out-of-the-box), C off**. The harness in this repo re-enables C with
`--cache-reuse 256` because it is the cheap defence against partial-prefix
drift in long agent sessions.

---

## 3. What layer A actually does

Layer A is the simplest. The server tokenises the incoming prompt, finds the
longest prefix that already lives inside a slot's KV cache, evaluates only
the unseen suffix, and reports `progress = n_tokens / n_prompt_tokens` in the
log line.

`-sps N` controls the **slot-prompt-similarity threshold** for picking a
slot. `-sps 0.5` (the upstream default — see
[Tutorial — discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606))
means "match a slot if at least 50% of the prompt context overlaps". `-sps
0.0` disables automatic slot selection entirely (callers must then pin
`id_slot` per request). Above ~0.7 the server prefers the best-overlap slot
even when a free slot exists.

**Predictability caveat.** `tools/server/README.md` warns: "*Because
(depending on the backend) the logits are **not** guaranteed to be bit-for-bit
identical for different batch sizes (prompt processing vs. token generation)
enabling this option can cause nondeterministic results.*" Treat any
exact-match assertion against `cache_prompt = true` output as
non-deterministic; compare the score distribution, not the bit-exact token
sequence.

---

## 4. What layer B actually does

Layer B is the ggerganov "host-memory prompt caching" feature, introduced
in [PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391) (merged
2025-10). From the PR description:

> *Initial version of automatic memory offloading to host memory using an
> extended logic for minimizing the prompt reprocessing. The host-memory
> prompt cache acts as "extra slots" with which we can calculate prefix
> similarity and decide to hot-swap them into the `llama_context` if it
> would reduce the processing. The cache is stored in regular RAM.*

Two limits cap the pool (PR #16391): **max bytes** (`--cache-ram`,
default `8192` MiB = 8 GiB) and **max tokens** (default `== -c
--context-size`). `-cram -1` disables the byte cap; `-cram 0` disables the
feature entirely. `--cache-idle-slots` (default on, requires `--cache-ram`)
governs whether idle slots are saved to the pool when a new task arrives.

Production-scale confirmation from a maintainer-noted comment by **AesSedai
2025-10-12** ([PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391))
on a dual-3090 + 768 GB DDR5 rig running R1 0528 / GLM-4.6 with
`--cache-ram 65536`:

> *This PR has made me a very happy dev and has saved me probably hours of
> prompt processing already. […] So this capability to offload the already
> processed prompt in a sort of pause / resume fashion means I don't have
> to compromise with eg, `--parallel 2` to get an "agentic slot" any longer
> and I can swap between tasks in cline / roo code as well without
> thrashing the cache in its entirety now on long contexts.*

The canonical log line for a layer-B hit (sampled from AesSedai's log on
2025-10-13 in [PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391)):

```
srv  get_availabl: updating prompt cache
srv  prompt_save:  - saving prompt with length 7745, total state size = 2783.450 MiB
srv        load:  - looking for better prompt, base f_keep = 0.001, sim = 0.001
srv      update:  - cache token limit reached, removing oldest entry (size = 4769.420 MiB)
srv      update:  - cache state: 10 prompts, 18965.563 MiB (limits: 65536.000 MiB, 65536 tokens)
srv      update:    - prompt 0x46352d50:   13469 tokens, checkpoints:  0,  4840.578 MiB
srv      update:    - prompt 0x4eef33d0:   14177 tokens, checkpoints:  0,  5095.024 MiB
```

`f_keep` is the *fraction of the new prompt that the existing cached KV
already covers*. `sim` is the *similarity score* between cached tokens and
the new prefix. When `f_keep` is high and `sim` is high, the cache pool
hot-swaps that prompt back into a slot before prefill starts, so only
`1 - f_keep` of the prompt is actually evaluated.

The 8-GiB default (`8192` MiB) is documented in `common/arg.cpp:1716-1725`
and again in `tools/server/README.md`'s `-cram, --cache-ram` row, both
pointing at [PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391).

---

## 5. What layer C actually does

Layer C is `n_cache_reuse` / `--cache-reuse N` — **block-shift reuse**. When
a new request tokenises to a prompt whose prefix **partially** overlaps a
cached slot, the engine shifts the cached KV blocks to the start of the new
context window and processes only the trailing `N`-token chunk that didn't
fit.

**The upstream default is `0` (disabled)**, confirmed in two places:
`common/common.h:610` (`int32_t n_cache_reuse = 0; // min chunk size to
reuse from the cache via KV shifting`) and `common/arg.cpp:3541-3551`
(`(default: %d)` formats `params.n_cache_reuse`, which is `0`). This is the
factual correction to the earlier sketch that claimed the default was `256`.

The harness in this repo wires `--cache-reuse 256` via
`scripts/serve-config.py:166-169` (overrides the `CACHE_REUSE` key). The
`256` is a token floor: anything below 256 tokens of aligned prefix is
considered too small to be worth the hash-lookup + KV-shift overhead. The
trade-off is small `N` = more aggressive reuse (more wall-time saved when
the prefix matches) but more hash work; large `N` = fewer matches but
cheaper lookups.

The maintainer view on whether layer C is still useful after
[PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391) was asked
explicitly by ddh0 2025-10-12. ggerganov 2025-10-12 answered:

> *No, `cache_reuse` is still useful. For example you still need it for the
> advanced FIM used in https://github.com/ggml-org/llama.vim.*

So layer C remains the right tool for **fill-in-the-middle** workloads
(`llama.vim` and any FIM-style code completion) where the prefix is short
and matches drift across completions.

---

## 6. How to read the log to confirm a hit

The canonical log line for a **layer-A hit** comes from the smahs
[Tutorial — discussion #13606](https://github.com/ggml-org/llama.cpp/discussions/13606)
2025-05-17 (edited by ggerganov), second-request trace:

```
slot update_slots: id  0 | task 31 | new prompt, n_ctx_slot = 512, n_keep = 0, n_prompt_tokens = 43
slot update_slots: id  0 | task 31 | need to evaluate at least 1 token to generate logits, n_past = 43, n_prompt_tokens = 43
slot update_slots: id  0 | task 31 | kv cache rm [42, end)
slot update_slots: id  0 | task 31 | prompt processing progress, n_past = 43, n_tokens = 1, progress = 0.023256
slot update_slots: id  0 | task 31 | prompt done, n_past = 43, n_tokens = 1
```

| Field | Meaning |
|---|---|
| `n_prompt_tokens` | Total token count of the new request's prompt. |
| `n_past` | Tokens already in the slot's KV cache (matched prefix length). |
| `n_tokens` | Tokens actually evaluated during prefill (the *unseen* suffix). |
| `progress` | `n_tokens / n_past` ≈ `(1 - prefix_fraction)`. |

Reading the table: `progress ≈ 1.0` means full prefill (cold). `progress
≪ 1.0` means most of the prompt was reused. The exact ratio for the
canonical line is `1 / 43 ≈ 0.023256` — 42 of 43 tokens were matched, only
1 evaluated.

For **layer-B** the confirming log is the `cache state: N prompts, X MiB
(limits: …)` line plus the `f_keep`/`sim`/`n_tokens` triplet on the
`load:` line — see §4. For **layer-C** the confirming log is the same
`prompt processing progress` line; layer C is what makes `progress` small

on a **partial** prefix overlap that layer A would have rejected.

> **Cleaner signal: per-request JSON observables.** For programmatic A/B the
> JSON response carries the same numbers without scraping the server log.
> From the upstream server schema ([`tools/server/README.md` L1390-1430](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#L1390)):
> `timings.cache_n` is the number of prompt tokens reused from cache,
> `timings.prompt_n` is the number of prompt tokens actually processed (the
> *unseen* suffix), `timings.prompt_ms` is the total prefill wall time, and
> `usage.prompt_tokens_details.cached_tokens` is the standard OpenAI-style
> cached token count. The invariant on every chat-completion response is:
> **`prompt_n + cache_n + predicted_n = total prompt-context tokens evaluated`**.

---

## 7. SWA models

Sliding-Window Attention (SWA) models (Mistral, Phi-3-mini, command-r,
etc.) have a finite attention span: tokens beyond the sliding window
contribute nothing to the output. For these models, the maintainer's
documented rule is: **set `--swa-full` if you intend to evaluate prompts
whose length exceeds the sliding window**. Without `--swa-full`, the
prefix-reuse machinery may return a cached prefix that the model can no
longer attend to, which silently degrades quality rather than failing
loudly.

If you cache a prefix under SWA and then serve a request longer than the
window, layer A/B may hand back a stale prefix the model can't fully use.
Rule: when running SWA on this rig, set `--swa-full` and re-measure; if
quality regresses, check whether the cached prefix length exceeds the
model's sliding window.

---

## 8. Multimodal caveat (mtmd)

[PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391) is explicit on
this:

> *Note: mtmd workarounds are starting to cause some headaches. For example
> `server_tokens` is not copyable which complicates the cache logic and
> makes the prompt caching feature incompatible with mtmd.*

Practical consequence: **do not run `--cache-ram` (and probably don't run
`--cache-prompt`) on a server that is also loading an `-mm`/`-mmproj`
multimodal projector**. Either the image side will reject the request, or
the cache will silently fail to hit. For multimodal workloads on this rig,
disable layer B (and accept that layer A only hits while a slot is alive).

---

## 9. vLLM and SGLang parity

The closest analogue is vLLM's Automatic Prefix Caching (APC) — see
[vLLM Prefix Caching design doc](https://docs.vllm.ai/en/v0.17.1/design/prefix_caching/).
APC uses **block-level hashes** of contiguous token blocks, parent-linked
into a tree, with the same goal as llama.cpp's layer A/B/C. Two
differences worth noting:

- **Granularity.** SGLang's [RadixAttention](https://lmsys.org/blog/2024-01-17-sglang/)
  ([paper](https://arxiv.org/abs/2312.07104)) goes finer than block-level:
  it maintains a token-level trie and can match at token granularity.
  llama.cpp's cache is block-level (token chunks, default `n_cache_reuse`
  grain) — see [sglang-inference-engine.md §2.1](./sglang-inference-engine.md).
- **Cache isolation.** vLLM supports a per-request `cache_salt` that is
  injected into the first block's hash so that requests with different
  salts cannot reuse each other's blocks — see
  [vLLM Prefix Caching — Cache Isolation](https://docs.vllm.ai/en/v0.17.1/design/prefix_caching/).
  **llama.cpp has no `cache_salt` equivalent.** Anyone able to send a
  request to your `llama-server` can intentionally or accidentally prime
  the host-RAM cache for another tenant's prompts. This is a documented
  shared-trust-group assumption.

---

## 10. Operator rules for this rig

1. **Verify-by-log, not by assumption.** Confirm a prefix-cache hit is
   actually happening by reading the `prompt processing progress,
   n_past=…, n_tokens=…, progress=…` line. `progress < 0.05` on a
   shared-prefix turn is the smoking gun.
2. **Don't disable without a documented reason.** `--no-cache-prompt` or
   `-cram 0` measurably destroys multi-turn agent wall-time. If you must
   disable for a measurement, name the reason in the commit message.
3. **Two-curl A/B recipe** (when validating the win on a new baseline):

   ```bash
   # Turn 1: cold prefill
   curl -s -X POST http://127.0.0.1:18765/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg s "$(cat long_system_prompt.txt)" \
            '{messages:[{role:"system",content:$s},{role:"user",content:"turn-1"}],cache_prompt:true}')" \
        > /dev/null
   # Turn 2: warm prefill (same prefix, different user content)
   curl -s -X POST http://127.0.0.1:18765/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg s "$(cat long_system_prompt.txt)" \
            '{messages:[{role:"system",content:$s},{role:"user",content:"turn-2"}],cache_prompt:true}')" \
        > /dev/null
   ```

   Read the server log between the two curls. A real layer-A/B hit drops
   `progress` from ≈ 1.0 to < 0.1 on turn 2.

4. **Layer-C A/B.** Compare `CACHE_REUSE=0` (rely on layers A+B only) vs
   `CACHE_REUSE=256` (the harness default) on a long agent session with
   slightly drifting prefixes (different system-prompt timestamp each
   turn). If `CACHE_REUSE=0` is within ±5% wall-time, layer C isn't
   earning its overhead for that workload — note that finding and keep
   the harness default until a workload explicitly benefits.
5. **JSON-observable A/B (wire-side).** For programmatic checks that don't
   scrape the server log, `timings.cache_n` is the ground-truth hit count:

   ```bash
   curl -s -X POST http://127.0.0.1:18765/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '…same-prefix-as-turn-1…' \
        | jq '.timings.cache_n'
   ```

   Compare two consecutive turns; assert `cache_n_2 > 0` for a real
   layer-A/B hit (see §6). **Falsifiable one-shot:** with the same system
   prefix on both turns, the 2nd response must report
   `cache_n >= N` where `N` = system-prompt tokens (minus the new user
   message tokens). Hit ⇒ `cache_n >= N`; miss ⇒ `cache_n == 0`.

---
--cache-ram ≈ N_prefixes × N_tokens × bytes_per_token
```

with **`bytes_per_token ≈ 16 KiB` for Qwen q4_0** at typical head sizes
(estimate; verify per model — MoE and SWA both raise this). Concrete
budget for a 4-prefix pool of 8 k tokens each at 16 KiB/token:

```
4 × 8192 × 16 KiB = 512 MiB → --cache-ram 512 is comfortable.
```

The upstream default of `8192` MiB gives headroom for ~64 such prefixes
at 8 k tokens each. MoE models (GLM-4.6, R1 0528) take more — AesSedai's
PR-16391 production setup uses `--cache-ram 65536` precisely because
GLM-4.6 KV at 65 k context is multiple GiB per prompt. See
[advanced-inference-optimizations.md §3](./advanced-inference-optimizations.md)
for KV-budget background and the 8 GiB cap discussion.

---

## Open questions

- **`-sps` predictability.** Layer A's log line reports `progress` but
  not the slot chosen. For multi-slot rigs, reproducing a hit requires
  pinning `id_slot` — but `-sps` behaviour at > 0.7 is not documented in
  upstream. TBD: empirical sweep on this rig.
- **SWA audit.** No runbook yet for which SWA models on this rig require
  `--swa-full` under layer-B. Need a per-model check.
- **Multi-tenant isolation.** No `cache_salt` analogue (vLLM has it;
  llama.cpp doesn't — see §9). For a future shared-trust deployment this
  is a known gap.
- **`CACHE_REUSE=0` vs `256` A/B.** No measured comparison on the rig
  yet — §10 step 4 will produce it.

---

## References

Primary sources cited above (accessed 2026-08-23):

- `common/arg.cpp` master, lines 1716-1725, 3536-3551 — `common/arg.cpp` master on GitHub.
- `common/common.h` master, lines 609-615 — `common/common.h` master on GitHub.
- `tools/server/README.md` master — `tools/server/README.md` master on GitHub.
- [PR #16391 — server: host-memory prompt caching](https://github.com/ggml-org/llama.cpp/pull/16391) (ggerganov, merged 2025-10) — authoritative source for `--cache-ram` semantics and the mtmd incompatibility; AesSedai 2025-10-12 production comment and 2025-10-13 log dump.
- [Discussion #13606 — Tutorial: KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606) (smahs 2025-05-17, edited by ggerganov) — canonical source for `-sps 0.5` default and the `prompt processing progress, n_past=43, n_tokens=1, progress=0.023256` log line.
- [vLLM Prefix Caching design doc v0.17.1](https://docs.vllm.ai/en/v0.17.1/design/prefix_caching/) — block-hash APC + `cache_salt`.
- [SGLang RadixAttention blog post (2024-01-17)](https://lmsys.org/blog/2024-01-17-sglang/) and [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) — token-level trie prefix cache.

Cross-link targets in this repo:

- [`docs/llamacpp-flags-audit.md`](../llamacpp-flags-audit.md) — `--cache-reuse`, `--cache-prompt`, `--cache-ram` audit rows.
- [`docs/discovery/advanced-inference-optimizations.md §3`](./advanced-inference-optimizations.md) — KV budget context.
- [`docs/discovery/sglang-inference-engine.md §2.1`](./sglang-inference-engine.md) — RadixAttention overview.
- [`docs/sessions/2026-08-02-research-gap-closure.md §KV sizing`](../sessions/2026-08-02-research-gap-closure.md) — local rig KV sizing.
