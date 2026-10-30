# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-25-qwen38-4b-distill-kv-ab.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```powershell
# one per variant; only KV_CACHE / KV_CACHE_K / KV_CACHE_V and CTX_SIZE edited in config.py Baseline
$env:AUTORESEARCH_SKIP_FREE_CLAMP = "1"
# q4_0 / q8_0 / f16 @65536 and q8_0 @131072 — engine b10549 (default AUTORESEARCH_LLAMA_CPP_ROOT)
# turbo3 @65536 — fork engine
$env:AUTORESEARCH_LLAMA_CPP_ROOT = "<repo>\llama.cpp-releases\turboquant\tqp-v0.3.0"
```


| KV | trial |
| ---- | ------- |
| q4_0 | `621d0719` |
| q8_0 | `5a70ee91` |
| f16 | `55aa739e` |
| turbo3 | `38fc1cec` |


- q8_0: historical throughput was 0.5 t/s below q4_0 at 65k context.
- f16: historical peak VRAM was 1.3 GB above q4_0.
- turbo3 (fork engine): ties q8_0 agentic, lowest coding + TPS of the set; VRAM ≈ q4_0 — turbo compression buys nothing on this arch at 65k (GDN KV already ≈¼ dense). Row is **not engine-isolated** (type exists only in the fork; `binary_version` = `turboquant@tqp-v0.3.0`)


| KV | trial |
| ---- | ------- |
| q4_0 @131072 (ref) | `6069530a` |
| q8_0 @131072 | `54e49297` |


Ran clean under the 7676 monitor ceiling (est 7601 passed with SKIP_FREE_CLAMP).

## Preserved historical runtime tables

| KV | bench tg | trial |
| ---- | ---------- | ------- |
| q4_0 | 73.4 | `621d0719` |
| q8_0 | 72.9 | `5a70ee91` |
| f16 | **74.4** | `55aa739e` |
| turbo3 | 71.8 | `38fc1cec` |

| KV | bench tg | trial |
| ---- | ---------- | ------- |
| q4_0 @131072 (ref) | 74.9 | `6069530a` |
| q8_0 @131072 | 73.8 | `54e49297` |
