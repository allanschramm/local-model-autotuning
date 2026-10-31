# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-08-23-new-models-qwen38-distill.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

```powershell
# Hardware gate (do first — operator-only, not run this session)
.\venv\Scripts\python.exe scripts\check_hardware.py
# Candidate planning (candidate list only)
# uvx whichllm@latest --profile coding
# llmfit plan "Qwen3.8-4B"

# Download — the one NEW fit (run only after hardware confirm)
hf download empero-ai/Qwen3.8-4B-Distill-GGUF Qwen3.8-4B-Q4_K_M.gguf --local-dir models/empero-ai/Qwen3.8-4B-Distill-GGUF
# optional A/B same dir: Qwen3.8-4B-Q5_K_M.gguf (3.161 GB), Q6_K (3.563 GB), Q8_0 (4.611 GB)

# Validation — one model at a time, no autoloop
# then on same Fingerprint:
```


| Repo (HF) | Arch / ctx | GGUF file (hf --dry-run 2026-08-23) |
| --- | --- | --- |
| **`empero-ai/Qwen3.8-4B-Distill-GGUF`** | Distill of Qwen3.8-2.4T-A95B into Qwen3.5-4B arch, Gated DeltaNet 8/32 full-attn, 262K-equivalent | **2.8 GB** Q4_K_M, 3.2 GB Q5, 3.6 GB Q6, 4.6 GB Q8 (verified) |
| `deepgrove/maple-preview-GGUF` | 20B-A1B MoE, 24L 256×8, 3:1 SWA-512:GA, MIT | **5.0 GB** TQ1_0-Q4_K / **5.9 GB** TQ2_0-Q4_K (+ F16 5.4/6.3G heads) — **hf-verified** |
| `HuggingFaceTB/SmolLM3-3B` (+ `unsloth/SmolLM3-3B-GGUF`) | dense 3B, 64K train → **128K YARN** | **1.9 GB** Q4_K_M / 1.8 GB Q4_0 (verified) |
| `ai9stars/G9v3-3B` | dense 3B, 131K | **no GGUF** in `ai9stars/G9v3-3B` (safetensors 5.4+0.6 GB only — hf-verified) |
| `amd/Instella-MoE-16B-A3B-Think` | 16B/2.8B active, 27L Gated MLA, 64K max | **no GGUF** (6×5.4G safetensors — hf-verified) |
| `microsoft/Fara1.5-9B` | Qwen3.5-9B CUA fine-tune (vision→click/type), 262K | **no GGUF** (4×5.9-6.0G safetensors — hf-verified) |
| `Qwen/Qwen3.8-27B` (+ GGUF forks) | dense 27B, 262K, Apache-2.0 | **16.8 GB** Q4_K_M + mmproj (C Chungulus) |

**Global context 2026-08-23 (web + hf):** `state-of-open-models-summer-2026` confirms Qwen dominance but weekly new-model roundup is only MiniMax-H3 + Qwen3.8-27B. **hf-verified now:** `maple-preview` is NOT 11-13 GB — actual 5.0/5.9 GB TQ quants fit 8GB class (ternary, not plain Q4). `LiquidAI/LFM2.5-DSpark` (2026-08-20) also NEW — drafters ~300M for `LFM2.5-1.2B/2.6B/8B-A1B`, 3.18× H100 / 2.87× M4 Max, identical greedy output — **engine adjacency, not a NEW base model**.
