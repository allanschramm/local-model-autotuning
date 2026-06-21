# Session 2026-06-19 (parte 5) — whichllm source code deep dive

## Origem
Allan pediu pra usar `gh` CLI pra extrair info do repo `Andyyyy64/whichllm` e cruzar com benchmarks de coding. **Análise READ-ONLY** do código fonte via GitHub API. Resultado salvo aqui.

## Repo metadata
- **Autor**: Andyyyy64 (= Allan Schramm — MESMO Allan)
- **Stars**: 4993 | **Forks**: 270
- **License**: MIT
- **Linguagem**: Python 100%
- **Created**: 2026-03-04 | **Updated**: 2026-06-19 (mesmo dia!)
- **Tags**: ai, cli, llm, local-llm, gguf, gpu, huggingface, inference, ollama, vram, apple-silicon, benchmarks

## Estrutura
```
src/whichllm/
├── cli.py                   (47.5 KB) — main CLI
├── constants.py             (1.3 KB)
├── data/
│   ├── framework.py         — quant framework table
│   ├── gpu.py               (7.9 KB)
│   ├── lineage.py           (5.4 KB) — generation half-order
│   └── quantization.py      (3.0 KB)
├── engine/
│   ├── compatibility.py     (9.4 KB) — VRAM/fit
│   ├── performance.py       (10.4 KB) — tok/s estimation
│   ├── quantization.py      (2.7 KB)
│   ├── ranker.py            (32.0 KB) — scoring & ranking
│   ├── types.py
│   └── vram.py              (3.0 KB)
├── hardware/                — GPU/CPU/RAM detection
├── models/
│   ├── benchmark.py         (22.2 KB) — fetch & merge benchmark sources
│   ├── benchmark_sources/
│   │   ├── aa_index.py      (15.5 KB) — Artificial Analysis
│   │   ├── aider.py         (5.0 KB) — Aider polyglot YAML
│   │   ├── chatbot_arena.py (4.5 KB) — HF dataset ELO
│   │   ├── livebench.py     (4.1 KB) — inlined CSV snapshot
│   │   ├── open_llm_leaderboard.py (3.2 KB) — parquet fetch
│   │   └── vision.py        (3.3 KB) — curated VLM snapshot
│   ├── fetcher.py           (35.2 KB) — model metadata fetch
│   └── grouper.py
docs/
├── how-it-works.md          (7.0 KB)
├── scoring.md               (6.6 KB)
└── ...
tests/                       — 27 test files
```

## Fontes de benchmark (5 ativas + 1 vision)

| Fonte | Status | Refresh | URL | Cap |
|---|---|---|---|---|
| **AA Index** | Current (live) | ~weekly | `artificialanalysis.ai` via `__NEXT_DATA__` JSON | sem cap explícito |
| **Aider polyglot** | Current (live) | quando muda YAML | `raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml` | sem cap |
| **Chatbot Arena ELO** | **FROZEN 2025-07-17** | nunca | HF dataset `mathewhe/chatbot-arena-elo` | **82 normalized** |
| **LiveBench** | Current (snapshot) | monthly refresh via `scripts/import_livebench_csv.py` | `livebench.ai/table_YYYY_MM_DD.csv` | sem cap |
| **Open LLM Leaderboard v2** | **ARCHIVED 2025-06** | nunca | HF parquet | **78 normalized** |
| **Vision (VLM)** | Curated snapshot | manual refresh | MMMU-Pro / MMBench blend | — |

### Merge logic (`benchmark.py`)

```python
# Frozen tier (older sources): OLLB v2, Arena ELO
# Current tier (live sources): LiveBench, Aider, AA Index
# Per-model merge: take MAX across all sources that have data
# Coverage: model gets current score if AA/LiveBench/Aider have it
#            else falls back to frozen, demoted by lineage recency
```

### Lineage recency demotion

Models sem cobertura current (só OLLB/Arena frozen) recebem multiplicador baseado na geração:
- Newest da família = 1.0 (sem demotion)
- Cada geração mais antiga = -12%
- Floor = 0.55 (55% do score frozen)

**Lineage map (genealogia half-order)**:
```python
"qwen":  [(qwen3.6, 7), (qwen3.5, 6), (qwen3-next, 6),
          (qwen3-coder-next, 6), (qwen3-omni, 5), (qwen3, 5),
          (qwq, 4), (qwen2.5, 3), (qwen2, 2), (qwen1, 1)]
"gemma": [(gemma-4, 4), (gemma-3, 3), (gemma-2, 2), (gemma, 1)]
"llama": [(llama-4.5, 5), (llama-4, 4), (llama-3.x, 3), ...]
```

## Scoring (ranker.py)

### Source weights
```python
"direct":       0.62  # independent leaderboard hit exato
"base_model":   0.55  # cardData.base_model pointer
"variant":      0.50  # suffix-stripped derivative
"line_interp":  0.40  # size-aware interpolation within family
"self_reported": 0.30  # uploader's own evalResults
"none":         0.00
```

### Quant quality penalty
```yaml
Q2_K:   -25%
Q3_K_M: -8%
Q4_K_M: -5%
Q5_K_M: -3%
Q6_K:   -2%
Q8_0:   -1%
F16:    0%
```

### Partial offload factor (CRÍTICO — explica ranking)
```python
# ratio = % de layers offloaded to CPU
if ratio >= 0.75:   factor = 0.42  # quase CPU-only
elif ratio >= 0.60: factor = 0.52
elif ratio >= 0.40: factor = 0.62
elif ratio >= 0.25: factor = 0.76
else:               factor = 0.86  # quase Full GPU

# MoE adjustment: se active set cabe na GPU, fator melhora
if model.is_moe and active_set_fits_in_gpu:
    factor = max(factor, 0.66 to 0.88)
```

### Generation bonus/penalty
```python
MODEL_GENERATION_BONUS_MAX = 10.0  # newest da família
MODEL_GENERATION_PENALTY_MAX = 6.0  # oldest
```

## Speed estimation (performance.py)

### Quant efficiency (fração do bandwidth-bound teórico)
```python
"NVFP4": 0.56  # 4-bit microscaling floats, FP4 tensor cores
"Q4_K_M": 0.55
"MXFP4": 0.55
"Q4_K_S": 0.55
"Q4_0":   0.53
"Q5_K_M": 0.52
"Q6_K":   0.50
"Q3_K_M": 0.50
"Q8_0":   0.45
"F16":    0.40
"BF16":   0.40
```

### Backend factor
```python
"nvidia": 1.00
"amd":    0.78
"apple":  0.82  # Metal kernel atrás na dequantization
"intel":  0.65
```

### MoE floor (per-token read ratio)
- Floor = 0.05-0.25 baseado em bandwidth
- ~5% em 256 GB/s (alto bandwidth), até 25% legacy
- **Explica a sub-estimativa do Qwen3.5-9B no nosso rig**

### Speed confidence (intervalo de incerteza)
```python
"low":    (0.35, 2.00)   # partial offload, no bandwidth, etc
"medium": (0.60, 1.60)
"high":   (0.85, 1.20)
```

## Por que whichllm ranK FLUTUA entre runs

1. **LiveBench inlined snapshot** — atualizado mensalmente; se Allan rodar em 2 dias diferentes, scores podem mudar ±2
2. **AA Index live fetch** — `__NEXT_DATA__` JSON pode mudar semanalmente
3. **Max-merge per model**: `current.get(k, 0.0) < v` → se 2 fontes empatam em 67.3 mas arredondamento varia, ranking muda
4. **Cache TTL 24h** — `benchmark.json` local cache; segunda run dentro de 24h usa cache idêntico (a flutuação que vimos é entre runs com mais de 24h OU com cache limpo)

Mas Allan rodou 2x em 5min com resultados diferentes. Provavelmente o `cache.json` ainda não existia no primeiro run (criou), e no segundo usou o cache. Mas o conteúdo é diferente porque AA Index fetched live em runs separados.

## Por que Gemma-4-26B-A4B é #1 no whichllm (mas coding ruim)

1. **Generation bonus**: gemma-4 = max (+10)
2. **AA Index direct hit**: 0.62 weight
3. **Partial offload factor with MoE adjustment**: active 4B cabe em 7.5 GB VRAM → factor max 0.88
4. **Score final**: provavelmente ~0.62 × 79 (AA) × 0.88 (MoE bonus) × 1.10 (gen bonus) ≈ ~62-70

Mas **coding benchmark (SWE-bench Verified 17.4%)** é medido em outro lugar:
- AA Index é INTELLIGENCE INDEX geral (10 hard benchmarks, code é um deles)
- MoE 4B-active vs SWE-bench multi-step agentic = ruim por design
- Aider polyglot (per-source 0.50) ajuda pra coding mas só single-turn
- **NENHUM dos sources usados mede multi-step agentic SWE-bench score diretamente**

**Conclusão**: whichllm mede inteligência geral + velocidade, NÃO coding agentic. Pra coding agentic (que é o que nosso autoloop testa via Coding benchmark), Gemma-4-26B é ruim despite alto score whichllm.

## Por que LuffyTheFox community distil ganhou pra Qwen3.5-9B

- LuffyTheFox/Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF
- Provavelmente tem **direct Aider polyglot hit** (community fine-tunes costumam entrar em leaderboards)
- `direct` source weight = 0.62 vs `line_interp` = 0.40 do base Qwen3.5-9B
- Luffy ganha mesmo sendo distil não-oficial

**Para Allan**: se quiser testar Qwen3.5-9B base oficial, precisa forçar `model_id` exato ou filtrar por `org=Qwen`.

## Comparação direta: whichllm prediction vs nosso empírico

| Modelo | whichllm est (4060) | Empírico Allan | Fator | Diagnóstico |
|---|---|---|---|---|
| Qwen3.6-35B-A3B Q4_K_M | 39.9 (Partial) | 22.5 (MTP-GGUF file swap) | **0.56×** | whichllm generoso, MoE offload penaliza mais que previsto |
| Qwen3.5-9B Q4_K_M | 29.7 (Full GPU) | 134-272 TPS (autoloop) | **4.5-9.2×** | whichllm **muito subestima** — MoE floor muito alto, mas nossa config otimizada é >10× mais rápida |

**Conclusão global**: whichllm **subestima systematicamente Full GPU** quando há otimização específica (MTP, KV cache q4_0, batch tuning). E **superestima Partial offload** porque não modela CPU RAM bandwidth direito.

## Como USAR whichllm pra Allan

**Confiável pra**:
- Listar modelos candidatos viáveis pro hardware
- Listar variantes de quant disponíveis
- Cross-ref VRAM requirement vs seu budget
- Identificar geração (qwen3.6 vs 3.5 etc)

**NÃO confiável pra**:
- Score de qualidade absoluto (intelligence index ≠ coding agentic score)
- Predição de tok/s (subestima Full GPU, superestima Partial)
- Ranking stability (flutua entre runs)

**Uso correto**: `whichllm plan "<modelo>"` pra ver VRAM/quant options, depois validar empiricamente.

## Próximos passos sugeridos (não executados)

1. **Adicionar whichllm como tool de discovery** no local-model-autotuning pipeline
2. **Cruzar whichllm ranking com EvalPlus HumanEval+ scores** (que é o que nosso autoloop testa) pra ver qual modelo melhor em coding agentic local
3. **Re-rodar `whichllm --gpu-only` (Full GPU only)**: o Allan tem 7.5 GB budget → só Qwen3.5-9B Q4_K_M cabe Full GPU → candidato ideal pra autoloop rodar sem gargalo CPU
4. **Documentar gap whichllm vs real**: contribuir upstream um PR que adiciona MTP como quant efficiency boost? (sem marca — Allan decide)
