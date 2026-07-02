# Session 2026-06-19 (parte 4) — whichllm plan + speed usable

## Origem
Allan rodou 2 `whichllm plan` + 1 `whichllm --speed usable`. Outputs verbatim abaixo.

## Output 3: `whichllm plan "qwen 3.6 35b"`

```
user@DESKTOP:~$ whichllm plan "qwen 3.6 35b"

Found 10 matches, using: nvidia/Qwen3.6-35B-A3B-NVFP4

╭───────────────────────────────────────────────────── Model Info ─────────────────────────────────────────────────────╮
│ Model:  nvidia/Qwen3.6-35B-A3B-NVFP4                                                                                 │
│ Params: 18.7B (3.0B active) | Arch: qwen3_5moe | Context: unknown                                                    │
│ License: apache-2.0                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

     VRAM Required (context: 4096)

┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Quant    ┃       VRAM ┃ Quality Loss ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Q2_K     │     6.8 GB │         -25% │
│ Q3_K_M   │     9.0 GB │          -8% │
│ Q4_K_M ★ │    11.2 GB │          -5% │
│ Q5_K_M   │    13.3 GB │          -3% │
│ Q6_K     │    15.5 GB │          -2% │
│ Q8_0     │    19.9 GB │          -1% │
│ F16      │    36.2 GB │           0% │
└──────────┴────────────┴──────────────┘

      GPU Compatibility (Q4_K_M, 11.2 GB required)

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ GPU            ┃     VRAM ┃     Fit      ┃ Est. Speed ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ RTX 4060       │     8 GB │  ~ Partial   │ 39.9 tok/s │
│ RTX 3060       │    12 GB │  ✓ Full GPU  │      117.3 │
│ RTX 4070       │    12 GB │  ✓ Full GPU  │      164.3 │
│ RTX 4080       │    16 GB │  ✓ Full GPU  │      233.6 │
│ RTX 4090       │    24 GB │  ✓ Full GPU  │      267.9 │
│ RX 7900 XTX    │    24 GB │  ✓ Full GPU  │      267.9 │
│ RTX 5090       │    32 GB │  ✓ Full GPU  │      375.1 │
│ A100 40GB      │    40 GB │  ✓ Full GPU  │      325.5 │
│ L40S           │    48 GB │  ✓ Full GPU  │      267.9 │
│ A100 80GB      │    80 GB │  ✓ Full GPU  │      426.8 │
│ H100           │    80 GB │  ✓ Full GPU  │      701.3 │
│ H200           │   141 GB │  ✓ Full GPU  │     1004.8 │
└────────────────┴──────────┴──────────────┴────────────┘

  ★ Minimum GPU for full offload: RTX 3060 (12 GB) at Q4_K_M
```

## Output 4: `whichllm plan 'qwen3.5 9b'`

```
user@DESKTOP:~$ whichllm plan 'qwen3.5 9b'

Found 6 matches, using: LuffyTheFox/Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF

╭───────────────────────────────────────────────────── Model Info ─────────────────────────────────────────────────────╮
│ Model:  LuffyTheFox/Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF                                             │
│ Params: 9.0B | Arch: qwen3_5 | Context: 262144                                                                       │
│ License: apache-2.0                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

     VRAM Required (context: 4096)

┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Quant    ┃       VRAM ┃ Quality Loss ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Q2_K     │     4.4 GB │         -25% │
│ Q3_K_M   │     5.4 GB │          -8% │
│ Q4_K_M ★ │     6.5 GB │          -5% │
│ Q5_K_M   │     7.5 GB │          -3% │
│ Q6_K     │     8.5 GB │          -2% │
│ Q8_0     │    10.6 GB │          -1% │
│ F16      │    18.4 GB │           0% │
└──────────┴────────────┴──────────────┘

       GPU Compatibility (Q4_K_M, 6.5 GB required)

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ GPU            ┃     VRAM ┃     Fit      ┃ Est. Speed ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ RTX 4060       │     8 GB │  ✓ Full GPU  │ 29.7 tok/s │
│ RTX 3060       │    12 GB │  ✓ Full GPU  │ 39.3 tok/s │
│ RTX 4070       │    12 GB │  ✓ Full GPU  │ 55.0 tok/s │
│ RTX 4080       │    16 GB │  ✓ Full GPU  │ 78.3 tok/s │
│ RTX 4090       │    24 GB │  ✓ Full GPU  │      110.1 │
│ RX 7900 XTX    │    24 GB │  ✓ Full GPU  │      104.8 │
│ RTX 5090       │    32 GB │  ✓ Full GPU  │      195.7 │
│ A100 40GB      │    40 GB │  ✓ Full GPU  │      169.8 │
│ L40S           │    48 GB │  ✓ Full GPU  │ 94.4 tok/s │
│ A100 80GB      │    80 GB │  ✓ Full GPU  │      222.7 │
│ H100           │    80 GB │  ✓ Full GPU  │      365.8 │
│ H200           │   141 GB │  ✓ Full GPU  │      524.2 │
└────────────────┴──────────┴──────────────┴────────────┘

  ★ Minimum GPU for full offload: RTX 4060 (8 GB) at Q4_K_M
```

## Output 5: `whichllm --speed usable`

Idêntico ao Output 2 do parte 3 (15:36 BRT). Lista com 10 modelos, top Gemma-4-26B-A4B-it Q3_K_M (66.3).

## Validação empírica vs whichllm estimate

### Qwen3.6-35B-A3B Q4_K_M no RTX 4060 (8GB)
- **whichllm**: 39.9 tok/s (Partial offload)
- **Empírico Allan (nossa medição)**: 22.5 tok/s com MTP-GGUF file swap
- **Gap**: ~57% do que whichllm prevê (chute generoso, como esperado)

### Qwen3.5-9B Q4_K_M no RTX 4060 (8GB)
- **whichllm**: 29.7 tok/s (Full GPU)
- **Empírico Allan (results.tsv)**: **134-272 TPS** em múltiplos runs com autoloop
- **Gap**: **4-9x MAIS RÁPIDO** do que whichllm prevê

Por que o gap gigante no Qwen3.5-9B:
- whichllm é low-confidence, chute baseado em heurísticas genéricas
- Nosso autoloop já otimizou config específica pra esse modelo (`q4_0` KV, ctx=32768, threads=8, etc)
- MTP-GGUF disponível — speedup real vs base GGUF que whichllm provavelmente assume
- whichllm não conhece nossa infra (turboquant build, MTP, cache quantization ótima)

## Observações

1. **whichllm plan matched um modelo que NÃO é o oficial Alibaba**:
   - `nvidia/Qwen3.6-35B-A3B-NVFP4` (vendor variant, quant NVFP4 — formato proprietário NVIDIA)
   - **NVFP4 NÃO é suportado pelo nosso llama.cpp turboquant build** (pelo que sei). Vendor-forked, não testado.
   - Pra rodar Qwen3.6-35B-A3B oficialmente, manter o `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (UD-Q4_K_M, Apache 2.0, suportado).

2. **whichllm plan matched um community fine-tune**:
   - `LuffyTheFox/Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF` — Claude-distilled, censurado removido. Não é o Qwen3.5-9B base oficial.
   - O **Qwen3.5-9B-MTP-Q4_K_M** que nosso autoloop usa é o oficial Alibaba com MTP, **não tem nada a ver** com essa distill.

3. **VRAM budget real do whichllm**: 7.5 GB usable (8 GB - 512 MB headroom). Allan pode ter mais VRAM disponível ajustando o headroom, mas que vai contra recomendação deles.

4. **Qwen3.5-9B Q4_K_M = único modelo que cabe Full GPU** no RTX 4060 com 7.5 GB budget (6.5 GB needed). É nossa referência de velocidade máxima possível nessa máquina.

5. **Qwen3.6-35B-A3B Q4_K_M precisa 11.2 GB** — sempre vai precisar CPU offload no RTX 4060. Aceitar o Partial.

## Cross-ref com nossos dados

| Modelo | whichllm est. (4060) | Empírico Allan | Notas |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 39.9 (Partial) | 22.5 (MTP-GGUF) | Gap -57%, esperado |
| Qwen3.5-9B Q4_K_M | 29.7 (Full GPU) | **134-272** (autoloop) | Gap **+450-820%** |

## Pendências atualizadas
- [ ] Confirmar que `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` é suportado (já validado: `nextn_predict_layers=1` no log)
- [ ] Investigar NVFP4 (vendor NVIDIA) vs GGUF (open llama.cpp) — qual realmente suportado pelo nosso build
- [ ] Criar alias `qwen3.5-9b-mtp` quando Allan for testar manualmente (já tem best config 134-272 TPS histórico)
