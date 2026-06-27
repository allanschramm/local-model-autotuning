# Agent Experiment Memory

## 1. Experiment Log (Best per model)

| Commit | Model | KV | Ctx | VRAM (GB) | TPS | Score | Status |
|---|---|---|---|---|---|---|---|
| `bab91e9` | Qwen3.5-9B-MTP | q4_0 | 32768 | 6.9 | 157.3 | 0.772067 | Keep |
| `0c8e5fc` | gemma4-v2 | turbo4 | 65536 | 8.0 | 29.5 | 0.312922 | Keep |
| `a06b9f9` | Ornith-1.0-9B | q4_0 | 131072 | 7.4 | 52.2 | 0.580000 | Keep |
| `d5ddbd1` | Ornith-1.0-35B | q4_0 | 131072 | 6.3 | 27.9 | 0.555000 | Keep |

## 2. Working Configs

*   **Qwen3.5-9B-MTP**: `kv=q4_0`, `ctx=32768`, `threads=12` → **0.7721** (coding=0.8167, retrieval=0.4150)
*   **gemma4-v2**: `kv=turbo4 K+V`, `ctx=65536`, `threads=8`, `temp=0.4`, `batch=128/ubatch=64` → **0.3129** (coding=0.3333, retrieval=0.1499)
*   **Ornith-1.0-9B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4` → **0.5800** (coding=0.5800: lcb=0.40, he=0.80, mbpp=0.90, bigcode=0.10)
*   **Ornith-1.0-35B**: `kv=q4_0`, `ctx=131072`, `threads=8`, `temp=0.4`, `n-cpu-moe=36` (VITRIOL) → **0.5550** (coding=0.5550: lcb=0.40, he=0.80, mbpp=0.80, bigcode=0.10)

## 3. Blocked Configurations (Fails / Regressions)

*   **gemma4-v2 temp=0.2 + batch=512/256 + repeat_penalty=1.05**: score caiu para **0.1648**. MBPP timeout na task 1/3. Não usar.
*   **gemma4-v2 ctx<65536 com Nexus**: padding de 50K tokens quebra o contexto. Nexus exige ctx>=65536.
*   **Low Throughput**: `g4-opt-it` com `q4_0` em 16k → 24.3 TPS (abaixo do threshold 30.0). Score zerado. Precisa turbo3/turbo4 ou mais threads.
*   **VRAM Limits**: modelos maiores que 9B com `f16` KV em 64k+ estouram 7.9 GB. Usar turbo4.

## 4. Working Hypotheses

*   **Turbo4 no Gemma4 12B dense**: VRAM fica flat em ~7.9GB até 128K ctx graças ao TurboQuant agressivo. Custo do KV cache é amortizado na medição por `nvidia-smi` a 10Hz.
*   **MTP ausente no Gemma4**: modelo não tem `nextn_predict_layers`. Speculative decoding não ajuda aqui.
*   **Threads**: 8 threads no Gemma4 12B batem melhor que 12 (teste anterior com 12 threads não foi registrado porque ainda não há baseline confirmed).
*   **Batch sizing**: `batch=128, ubatch=64` é mais estável no Gemma4 do que `512/256` sob trial budget curto.
*   **Identação nos Modelos de Raciocínio (Ornith/Qwen 3.5)**: Blocos `<think>` geram rascunhos identados em 8+ espaços (dentro de markdown lists). Se o output final for truncado, o fallback parser do harness traz a identação extra, quebrando a compilação (`IndentationError`) ao concatenar a assinatura. `textwrap.dedent()` e re-identação com exatos 4 espaços corrigem o parser e dobram a taxa de acerto (de 0.40 para 0.80 no HE+ do Ornith 9B).
*   **VITRIOL MoE Streaming em 35B**: Mantendo todos os 256 routed experts no CPU/RAM via `--n-cpu-moe 40` permite carregar modelos de 35B em GPUs de 8 GB (RTX 4060) mantendo throughput aceitável (23.6 TPS) com pico de apenas 4.1 GB de VRAM.
