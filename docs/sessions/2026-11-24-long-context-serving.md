# Local model serving at 100K, November 24–27, 2026

The tests ran on Windows with an 8 GiB-class CUDA device. SenseNova and Inkling used the [previously validated GPU-only CUDA build](2026-11-22-sensenova-loop-cuda.md) and [seven-block expert placement](2026-11-23-inkling-expert-placement.md), respectively. Bonsai used [its PrismML prebuilt and hybrid-attention preflight](../models/ternary-bonsai-2.5-27b.md). All three used K/V `q4_0`.

| Model and quant | Requested target | Observed slot | Input → output | Scope |
| --- | ---: | ---: | ---: | --- |
| SenseNova Loop 9B `Q4_K_M` | 100,000 | 100,096 | 98,000 → 32 | One monitored native generation, 17.86 decode TPS |
| Inkling 2 Mini `UD-Q4_K_M` | 100,000 | 100,096 | 4,096 → 64 | One monitored native generation, 19.11 decode TPS |
| Ternary Bonsai 2.5 27B `PTQ1_0` | 100,000 | 100,096 | 32 → 512, three samples | Bounded Trial, 37.90 median decode TPS, PPL 7.3213 |

The two serving checks completed without truncation or a watchdog trip. They peaked at about 8,076 and 8,102 MiB of dedicated VRAM, respectively, with 78 MiB shared in each run. These are run-specific monitor readings. The small margin makes background GPU load material. For SenseNova, the earlier estimate was about 530 MiB of additional KV from 65,536 to 100,000 tokens. The near-full smoke tested that the prompt could be processed and decoded. Retrieval is still unmeasured for all three models, and near-full prefill is still unmeasured for Inkling and Bonsai.

## Read the fit estimate and the slot separately

- SenseNova's first bounded plan was rejected by VRAM preflight (`9ed4092a`). The separate serving check used the same GPU-only placement and monitored the actual native request. Its single decode measurement has a different prompt and sample count from the 65,536-token Trial.
- Inkling kept expert blocks 6–12 on GPU at 100K, with `N_CPU_MOE=17` and batch/ubatch 512/16. The accepted serving check used the Unsloth prebuilt. A separate preflight rejection (`f5b76234`) used an experimental runtime and belongs to that runtime's results.
- Bonsai's first bounded plan was rejected by a generic KV estimate (`6102e654`). The pinned binary's fit tool estimated 7,427 MiB including weights, context and buffers. The corrected estimator counts context-scaled KV for only the 16 full-attention blocks out of 64. The follow-up bounded Trial (`2bec6b4a`) confirmed 100,096 in the actual slot. Its fixed-text PPL used 2,048 tokens.

For a serving check, use one slot, keep RAM and GPU watchdogs active, compare the requested context to `/props` and `/slots`, and record the actual input and output counts, truncation and dedicated/shared GPU peaks. A rounded allocation can differ from the requested client limit. The SenseNova and Inkling timed requests are serving evidence. In SQLite, their earlier preflight rejections still have their original outcomes. The local evidence filenames are `100k-smoke.json` and `accepted-serving-smoke.json` in the respective serving artifact folders. The SQLite run prefixes identify the bounded plans.

## OpenCode2 and launcher checks

The tested OpenCode2 2.4.1 installation loaded two OpenAI-compatible providers and verified model listing and a text response through each at 65,536. The first configuration used distinct ports. The final configuration reused the launcher port with one active server at a time. Stop the previous server, start the selected model, and check the API model ID before using it in the client.

The provider used the OpenAI-compatible package and a loopback `/v1` base URL. Provider IDs and API model IDs are distinct fields. Switching both configurations to 100,000 aligned the client `limit.context` with the tested server capacity. Reloading confirmed both models were listed. A SenseNova text response was also verified through the client at the new limit. Inkling's 100K generation was checked directly via the native server. Tool calls and actual compaction behavior at the ceiling stayed outside the tests. [Model configuration](https://opencode.ai/docs/models/) · [Compaction configuration](https://opencode.ai/docs/config/#compaction)

Bonsai's launcher command was checked against the compatible runtime after its validated Fingerprint was applied. A separate launcher generation was outside that check. The alias cleanup reconciled registrations against existing GGUF files. Private provider settings, Fingerprints, alias inventory and raw logs remain local.
