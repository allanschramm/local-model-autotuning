# Session 2026-06-19 (parte 3) — whichllm outputs + coding benchmarks

## Origem
Allan descobriu a ferramenta `uvx whichllm@latest` e rodou duas vezes no mesmo dia com resultados diferentes. Salvos verbatim abaixo.

## Output 1 (15:31 BRT) — primeiro snapshot

```
shark@DESKTOP-2CDK6UR:~$ uvx whichllm@latest

╭─────────────────────────────────────────────────── Hardware Info ────────────────────────────────────────────────────╮
│ GPU 0: NVIDIA GeForce RTX 4060 — 8.0 GB (budget 7.5 GB) (CC 8.9, CUDA 13.3) — BW: 272 GB/s                           │
│ CPU: AMD Ryzen 7 5800X 8-Core Processor — 8 cores (AVX2)                                                             │
│ RAM: 23.5 GB                                                                                                         │
│ Disk free: 823.2 GB                                                                                                  │
│ OS: linux                                                                                                            │
│ VRAM headroom: 512 MB reserved per GPU                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

                                     Recommended Models
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃   # ┃ Model                       ┃ Quant  ┃   VRAM   ┃        Speed ┃ Published  ┃ Score ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│   1 │ google/gemma-4-26B-A4B-it   │ Q3_K_M │ Partial  │ 36.8 tok/s ? │ 2026-03-11 │  66.3 │
│   2 │ Qwen/Qwen3-8B               │ Q5_K_M │ Full GPU │ 25.1 tok/s ~ │ 2025-04-27 │  65.2 │
│   3 │ Qwen/Qwen3-30B-A3B          │ Q4_K_M │ Partial  │ 39.9 tok/s ? │ 2025-04-27 │  64.7 │
│   4 │ openai/gpt-oss-20b          │ Q3_K_M │ Partial  │ 38.9 tok/s ? │ 2025-08-04 │  63.7 │
│   5 │ Qwen/Qwen3-4B-Thinking-2507 │  Q6_K  │ Full GPU │ 41.6 tok/s ~ │ 2025-08-05 │  63.0 │
│   6 │ Qwen/Qwen3-14B              │ Q3_K_M │ Partial  │  9.5 tok/s ? │ 2025-04-27 │  62.2 │
│   7 │ microsoft/phi-4             │ Q4_K_M │ Partial  │  8.2 tok/s ? │ 2024-12-11 │  61.5 │
│   8 │ zai-org/GLM-4.7-Flash       │ Q3_K_M │ Partial  │ 11.7 tok/s ? │ 2026-01-19 │  59.6 │
│   9 │ google/gemma-3-12b-it       │ Q3_K_M │ Full GPU │ 25.5 tok/s ~ │ 2025-03-01 │  58.6 │
│  10 │ Qwen/Qwen3.6-27B            │ Q4_K_M │ Partial  │  4.3 tok/s ? │ 2026-04-21 │  58.0 │
└─────┴─────────────────────────────┴────────┴──────────┴──────────────┴────────────┴───────┘

  Speed:  ~ = estimated tok/s range,  ? = low-confidence/backend-sensitive tok/s
  Top pick confidence: Low (direct benchmark, gap +1.2, partial offload, low-confidence speed)
  Benchmark reference: 2026-05 curated snapshot; live AA / LiveBench / Aider merged when reachable.
  Note: Top candidates are very close (#1 vs #2: 1.2 pts).
  Speed caution: Low-confidence speed estimates in top ranks: #1, #3
  Warning #1 gemma-4-26B-A4B-it: ~39% of layers will be offloaded to CPU RAM
  Warning #3 Qwen3-30B-A3B: ~57% of layers will be offloaded to CPU RAM
```

## Output 2 (15:36 BRT) — segundo snapshot, ~5min depois

```
shark@DESKTOP-2CDK6UR:~$ uvx whichllm@latest

╭─────────────────────────────────────────────────── Hardware Info ────────────────────────────────────────────────────╮
│ GPU 0: NVIDIA GeForce RTX 4060 — 8.0 GB (budget 7.5 GB) (CC 8.9, CUDA 13.3) — BW: 272 GB/s                           │
│ CPU: AMD Ryzen 7 5800X 8-Core Processor — 8 cores (AVX2)                                                             │
│ RAM: 23.5 GB                                                                                                         │
│ Disk free: 823.2 GB                                                                                                  │
│ OS: linux                                                                                                            │
│ VRAM headroom: 512 MB reserved per GPU                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

                                     Recommended Models
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃   # ┃ Model                                    ┃ Quant  ┃   VRAM   ┃        Speed ┃ Published  ┃ Score ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│   1 │ google/gemma-4-26B-A4B-it                │ Q3_K_M │ Partial  │ 36.8 tok/s ? │ 2026-03-11 │  66.3 │
│   2 │ Qwen/Qwen3-8B                            │ Q5_K_M │ Full GPU │ 25.1 tok/s ~ │ 2025-04-27 │  65.2 │
│   3 │ Qwen/Qwen3-30B-A3B                       │ Q4_K_M │ Partial  │ 39.9 tok/s ? │ 2025-04-27 │  64.7 │
│   4 │ openai/gpt-oss-20b                       │ Q3_K_M │ Partial  │ 38.9 tok/s ? │ 2025-08-04 │  63.7 │
│   5 │ Qwen/Qwen3-4B-Thinking-2507              │  Q6_K  │ Full GPU │ 41.6 tok/s ~ │ 2025-08-05 │  63.0 │
│   6 │ zai-org/GLM-4.7-Flash                    │ Q3_K_M │ Partial  │ 11.7 tok/s ? │ 2026-01-19 │  59.6 │
│   7 │ google/gemma-3-12b-it                    │ Q3_K_M │ Full GPU │ 25.5 tok/s ~ │ 2025-03-01 │  58.6 │
│   8 │ Qwen/Qwen3-4B-Instruct-2507              │  Q6_K  │ Full GPU │ 41.6 tok/s ~ │ 2025-08-05 │  57.7 │
│   9 │ deepseek-ai/DeepSeek-R1-Distill-Llama-8B │ Q5_K_M │ Full GPU │ 25.6 tok/s ~ │ 2025-01-20 │  55.1 │
│  10 │ Qwen/Qwen3-4B                            │  Q6_K  │ Full GPU │ 41.6 tok/s ~ │ 2025-04-27 │  54.5 │
└─────┴──────────────────────────────────────────┴────────┴──────────┴──────────────┴────────────┴───────┘

  Speed:  ~ = estimated tok/s range,  ? = low-confidence/backend-sensitive tok/s
  Top pick confidence: Low (direct benchmark, gap +1.2, partial offload, low-confidence speed)
  Benchmark reference: 2026-05 curated snapshot; live AA / LiveBench / Aider merged when reachable.
  Note: Top candidates are very close (#1 vs #2: 1.2 pts).
  Speed caution: Low-confidence speed estimates in top ranks: #1, #3
  Warning #1 gemma-4-26B-A4B-it: ~39% of layers will be offloaded to CPU RAM
  Warning #3 Qwen3-30B-A3B: ~57% of layers will be offloaded to CPU RAM
```

## Diff entre os dois snapshots (~5min de diferença)

**Mantidos** (mesma posição ou score):
- #1 gemma-4-26B-A4B-it (66.3)
- #2 Qwen3-8B (65.2)
- #3 Qwen3-30B-A3B (64.7)
- #4 openai/gpt-oss-20b (63.7)
- #5 Qwen3-4B-Thinking-2507 (63.0)

**Movidos**:
- GLM-4.7-Flash: #8 → #6 (sobe 2)
- gemma-3-12b-it: #9 → #7 (sobe 2)

**Removidos**:
- microsoft/phi-4
- Qwen3-14B
- Qwen3.6-27B

**Adicionados**:
- Qwen3-4B-Instruct-2507 (#8, score 57.7)
- deepseek-ai/DeepSeek-R1-Distill-Llama-8B (#9, score 55.1)
- Qwen3-4B (#10, score 54.5)

## Observação crítica

whichllm **NÃO é determinístico entre runs** — a saída muda em ~5min. Provavelmente o backend tem snapshot flutuante ou score varia. Significa:
- Ranking do whichllm **NÃO é referência confiável** de qualidade.
- "Score" da whichllm não corresponde a benchmarks reais de coding (veja comparação abaixo).
- O número `? tok/s` é low-confidence — sem chance de bater 36.8 Gemma-4 ou 39.9 Qwen3-30B-A3B com VITRIOL no hardware de Allan.

## Comparação com coding benchmarks reais (não whichllm)

Cruzei os modelos da lista com dados de Artificial Analysis, SWE-bench leaderboard, Aider polyglot, Kaitchup Substack, Reddit, Qwen oficial.

| Rank coding real | Modelo | SWE-bench Verified | LiveCodeBench / Aider | Outros | Fonte |
|---|---|---|---|---|---|
| 1 | **Qwen3.6-27B** (dense) | **77.2** | strong | SWE-Pro 53.5, Terminal-Bench 59.3, SkillsBench 48.2 | Qwen oficial Apr 24 |
| 2 | **Qwen3.6-35B-A3B** (MoE, nosso atual) | 73.4 | — | Coding Agent 73.4 | Qwen oficial model card |
| 3 | **Qwen3-30B-A3B** | — | beats gpt-oss-20b no Aider (anecdotal) | r/LocalLLaMA |
| 4 | **gpt-oss-20b** | — | **98.3%** (Paterson 38-tasks), ~80-88% Aider | ianlpaterson + aider.chat |
| 5 | GLM-4.7-Flash (high reasoning) | 72.8 | — | SWE-bench leaderboard |
| 6 | Qwen3-8B | — | — | denso Full GPU baseline |
| 7 | phi-4 | — | — | GPQA/MATH > GPT-4o (dados 2024, desatualizado) |
| 8 | Gemma-3-12B-it | — | 80.6% Paterson | older gen |
| **9** | **Gemma-4-26B-A4B-it** | **17.4** | ruim | Reddit: "5x worse than Qwen3.6-27B on SWE-bench" |
| 10 | Qwen3-4B family | — | — | small, Full GPU |

## Conclusões

1. **whichllm ranking ≠ coding ranking.** Gemma-4-26B-A4B é #1 no whichllm mas é a pior em coding (17.4% SWE-bench). Inversão completa de prioridade pra uso coding.

2. **Qwen3.6-27B** (whichllm #10) é o melhor coding open-source viável no hardware de Allan. Dense, Apache 2.0, multimodal. Tem versão MTP-GGUF.

3. **Qwen3.6-35B-A3B** (nosso atual, NÃO está no whichllm) é #2 em coding. Mesma geração Qwen3.6 mas MoE.

4. **gpt-oss-20b** (whichllm #4) é forte em coding (Paterson 98.3%) mas comunidade reporta fraco em agentic.

5. **whichllm não cobre Qwen3.6-35B-A3B** nem Gemma 4 31B nem vários outros modelos relevantes. Base de dados incompleta.

## Próximo passo candidato (ainda NÃO executado)

Baixar `Qwen3.6-27B-MTP-GGUF` (variante MTP-GGUF do Qwen3.6-27B, mesma arquitetura que já validamos). Razões:
- Mesma arquitetura do nosso 35B-A3B mas **dense** (mais simples, sem overhead MoE)
- Coding best dos open models viáveis (77.2% SWE-bench)
- MTP-GGUF deve dar speedup similar (1.15-1.25× pra MTP)
- Estimativa otimista: 30-40 tok/s com Q4_K_M + MTP + q4_0 KV

## Pendência
- [ ] Adicionar `whichllm` aos tools conhecidos (já documentado aqui).
- [ ] Criar alias placeholder `qwen3.6-27b-mtp` quando Allan for baixar o GGUF.
- [ ] Comparar resultados empíricos locais (TPS medido) com as projeções whichllm `? tok/s`.
