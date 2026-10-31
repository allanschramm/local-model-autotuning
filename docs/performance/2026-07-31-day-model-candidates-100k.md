# Historical runtime evidence

Source: [archived original](https://github.com/allanschramm/local-model-autotuning/blob/ea43a71f48571811ec1ba25df4ab71ed272d34f1/docs/sessions/2026-07-31-day-model-candidates-100k.md). This is a performance-only extract; no new measurement was performed. Historical CLI smoke throughput is not proof of full-context `tuning-runtime-v1` performance. Estimates and open questions remain unvalidated; only entries explicitly labelled measured describe measurements. Use current tuning instructions for new runs.

Find new GGUF candidates for the operator host that can plausibly run at least 100k effective context and compete as DAY models. No model was downloaded or benchmarked in this session.

- discrete 8 GB-class NVIDIA, about 32 GB host RAM, `VRAM_LIMIT_MB=7900`.
- Dense models may not spill layers into shared memory. MoE experts may use `N_CPU_MOE`.
- User floor: `CTX_SIZE >= 100000`; target the repo default `131072` where the model supports it.


| GGUF basename | TPS |
| :--- | ---: |
| `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 43.6 |
| `ornith-1.0-9b-Q4_K_M.gguf` | 42.5 |
| `Qwen3.5-4B-MTP-Q4_K_M.gguf` | 87.0 |


- Official sources: [model/GGUF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF), [official files](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/tree/main).
- Official GGUF: `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`, 2.84 GB.
- Architecture: 3.97B-parameter Mamba2/Transformer hybrid, primarily Mamba-2 and MLP with only four attention layers; compressed from Nemotron Nano 9B v2.
- Context: up to 262k; NVIDIA reports RULER at 128k for the Q4_K_M.
- License: NVIDIA Nemotron Open Model License; official card says commercial use is allowed.
- Recommended sampling: reasoning `TEMP=1.0`, `TOP_P=0.95`; tool calling `TEMP=0.6`, `TOP_P=0.95`. `TOP_K`, `MIN_P`, and penalties are not published: leave them explicitly TBD rather than inventing values.
- DAY rationale: strongest 100k fit. The 2.84 GB weights and only four attention layers leave substantially more KV headroom than a full-attention 4B/8B model. Official positioning covers agentic edge use and coding languages. Quality and llama.cpp throughput remain unmeasured locally.

- Official sources: [model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny), [official GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF).
- Official GGUF choices: Q4_K_S 4.00 GB, Q4_0 3.96 GB, Q4_K_M 4.23 GB, Q5_K_M 4.95 GB, Q6_K 5.71 GB, Q8_0 7.39 GB. Start candidate: Q4_K_M; Q4_K_S is the fallback if preflight lacks headroom.
- Architecture: decoder-only hybrid MoE, 7B total / 1B active, four attention plus 36 Mamba-2 layers, 64 experts / six active, GQA.
- Context: 128k.
- License: Apache-2.0.
- Recommended sampling: not published on the official card; all sampler fields are TBD and must be resolved before a Trial.
- DAY rationale: 1B active parameters suggest high generation speed, while four attention layers make 100k KV plausible. Under repo rules the first Baseline uses `N_CPU_MOE=None` (full expert CPU offload resolved by GGUF metadata); actual CPU-offload TPS may erase the advantage.

- Official sources: [model card](https://huggingface.co/ibm-granite/granite-4.1-3b), [official GGUF](https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF).
- Official GGUF choices: Q4_K_S 2.00 GB, Q4_K_M 2.10 GB, Q5_K_M 2.44 GB, Q6_K 2.80 GB, Q8_0 3.62 GB. Start candidate: Q4_K_M.
- Architecture: dense decoder-only Transformer, 3B parameters, 40 layers, hidden size 2560, 40 attention heads / eight KV heads, head size 64, GQA/RoPE/SwiGLU.
- Context: 131072.
- License: Apache-2.0.
- Recommended sampling: not published on the official card; sampler fields remain TBD.
- DAY rationale: official card reports ; ; BFCL v3 60.80 and explicit tool-use support. Its small weights and half-size attention heads make 131k with quantized KV plausible on 8 GB, but unlike the two hybrids it pays KV for all 40 layers.


| Model | Decision | Reason |
| :--- | :--- | :--- |
| Granite 4.1 8B Q4_K_M (5.35 GB) | reject | Forty full-attention layers, eight KV heads and 128-wide heads leave no realistic 100k KV + runtime headroom on 8 GB under the dense no-spill rule. |
| Qwen3 4B Q4_K_M (2.50 GB) | defer | Official 131k support, but 36 full-attention layers with eight 128-wide KV heads make 100k very tight before runtime overhead; Granite 3B offers a safer full-attention control. |
| Qwen3.5 4B | not new / reject for DAY | It already has a complete local 131k vector: 87.0 TPS; ; below DAY admission. |
| Gemma 3 4B QAT Q4_0 | defer | 128k-capable and 3.16 GB, but official evidence is weaker for tool use and agentic coding than the three shortlisted candidates. |


1. Test Nemotron 3 Nano 4B Q4_K_M first.
2. Test Granite 4.0 H Tiny Q4_K_M second; its DAY case depends on measured CPU-expert-offload TPS.
3. Use Granite 4.1 3B Q4_K_M as the dense quality/control trial.

- Minimum context is 100k, so the current 32k DAY pick is not a valid comparison target for this search.
- Prefer hybrid layouts with few attention layers; they buy KV headroom without dense shared-memory spill.
- No downloads or Trials were authorized by this research request.
