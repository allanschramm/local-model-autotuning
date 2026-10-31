# Qwen3.6-35B DFlash vs MTP vs no-spec (Q3, CTX 65k)

## Goal

Measure whether `draft-dflash` raises TPS on `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` under the harness MoE split. Compare to no-spec and embedded `draft-mtp`.

## Hardware

- `discrete_gpu`; 8 GB-class; Baseline `VRAM_LIMIT_MB=7900`
- OS family: Windows
- Runtime: llama.cpp CUDA (harness `llama-cli` TPS + monitored `llama-server`)

## Setup

- Basename: `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` (`qwen35moe`, `block_count=41`, `nextn_predict_layers=1`)
- DFlash sidecar: `dflash-Qwen3.6-35B-A3B-Q8_0.gguf` (`dflash.block_size=16`, `target_layers=[2,7,12,17,23,28,33,38]`)
- Shared knobs: `CTX_SIZE=65536`, KV `q4_0/q4_0`, batch/ubatch `512/128`, threads `8`, `N_CPU_MOE=40`, `NO_MMAP=True`, `FLASH_ATTN=on`
- TPS = harness `llama-cli` `Generation:` t/s (`-c` capped at 4096)

## Commands

```powershell
# Baseline ENGINE only in autoresearch/core/config.py, then:
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "…"
```

## Findings

`results.tsv` ground truth. Validation smokes (incomplete Objective Vectors).

| Trial | Spec | `SPEC_DRAFT_N_MAX` | Bench tg | Peak VRAM |
|---|---|---:|---:|---:|
| `512c52ee-5f0c-423e-b348-28ddd9ed59b3` | none | 0 | **24.6** | 4.7 GB |
| `6fe4189f-7329-4c44-87b0-6325ed173b9b` | `draft-mtp` | 1 | **29.5** | 5.7 GB |
| `54e972a6-c0ec-4687-ba60-56a941125e3e` | `draft-dflash` + Q8 draft | 15 | **12.5** | 6.6 GB |
| `c409f082-8061-41d9-96da-015a1edb0504` | `draft-dflash` + Q8 draft | 15 | **9.5** | — (rejected: `TPS_FLOOR`) |

Prior Q4 @ 32k (`b10286`): DFlash 17.5 vs no-spec 27.2 vs MTP n=2 27.5 — [2026-08-07-qwen36-35b-dflash-tps.md](../performance/2026-08-07-qwen36-35b-dflash-tps.md).

**Verdict:** DFlash cannot raise TPS on this 8 GB-class host with this GGUF. It needs the 35B target fully on GPU. `--n-cpu-moe 40` makes DFlash slower. Embedded MTP can raise TPS. DFlash extracts eight target-layer hiddens every decode (`llama_set_embeddings_layer_inp`); that tax plus MoE CPU verify loses to no-spec.

Raw-cli WDDM shared-memory readings were a measurement artifact. Harness DFlash peak was 6.6 GB dedicated, no shared spill, no freeze.

## Errors

- First DFlash smoke 9.5 t/s rejected at `TPS_FLOOR=10`.
- Agent file-gate blocked Baseline edits to gitignored `config.py` and blocked Delete of the DFlash GGUF.

## Decisions

- Do not seed `draft-dflash` as a TPS Baseline on 8 GB-class MoE+`n-cpu-moe`.
- Prefer `draft-mtp` (embedded `nextn`) or no-spec.
- Keep TSV rows. Operator deletes leftover `dflash-*` GGUF locally (gate blocked the agent).

## Related

- Model card: [qwen3.6-35b-a3b.md](../models/qwen3.6-35b-a3b.md)
- Spec formats: [speculative-decoding-formats.md](../discovery/speculative-decoding-formats.md)
