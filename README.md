# local-model-autoresearch

Otimizador autônomo (hill-climbing) de flags de runtime de LLMs locais via `llama.cpp`. Baseado em [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

**O que faz:** Encontra a config de runtime mais rápida e precisa pro seu modelo GGUF local, testando milhares de combinações de flags automaticamente.

**O que NÃO faz:** Re-quantizar modelos. Só ajusta como o modelo é servido (KV cache, batching, threads, MTP).

---

## Quickstart (via Agente)

Abra seu agente de coding (Claude Code, Codex, Pi Agent, OpenCode) e cole:

> *"Descubra o melhor modelo pra **coding** que cabe no meu PC, baixe e comece o auto-tuning."*

O agente vai:
1. Detectar seu hardware (GPU/VRAM/RAM)
2. Rodar `whichllm` pra listar candidatos
3. Cruzar com SWE-bench / Aider / LiveCodeBench
4. Plotar Pareto frontier (tok/s vs qualidade)
5. Editar `autoresearch/core/config.py` com o melhor modelo
6. Rodar `python3 autoloop.py --vram-limit-mb=<budget>` overnight

**Resultado de manhã:** `results.tsv` com todos os trials + `config.py` na melhor config encontrada.

---

## Pré-requisitos

Instale **antes** de pedir pro agente:

| Dep | Comando | Por quê |
|---|---|---|
| Python 3.11+ | `sudo apt install python3.11 python3.11-venv` | runtime do autoloop |
| CUDA Toolkit | `nvidia-smi` + driver NVIDIA | llama.cpp precisa de `-DGGML_CUDA=ON` |
| build-essential + cmake >= 3.14 | `sudo apt install build-essential cmake` | compilar llama.cpp |
| uvx (ou uv) | `pip install uv` | rodar `uvx whichllm@latest` |
| huggingface_hub[cli] | `pip install huggingface_hub[cli]` | baixar GGUFs |

Depois clone e compile o `llama.cpp` (ver [seção Build](#build-do-llamacpp-com-cuda) abaixo).

### Verificar se tá tudo pronto

```bash
bash scripts/setup-check.sh
```

Output verde = pronto pro autoloop.

---

## Como Funciona

### O Loop

1. Lê a config atual melhor de `autoresearch/core/config.py`
2. Roda todos os benchmarks habilitados (Coding: HE+, MBPP+, LCB, BigCodeBench)
3. Calcula Val Score (acurácia ponderada + TPS floor)
4. Muta um param -> gera config Neighbor
5. Avalia Neighbor -> keep se melhorou (ou Pareto tie-break)
6. Se local maxima -> random restart
7. Loop pra sempre até Ctrl+C

### Contrato de Edição

| Arquivo | O quê | Agente/loop pode editar? |
|---|---|---|
| `autoresearch/core/config.py` | Config de runtime | **Sim** (só constantes) |
| `benchmark_search.py` | CLI runner | **Não** |
| `autoresearch/benchmarks/*` | Lógica de avaliação | **Não** |
| `results.tsv` | Métricas dos trials | **Só append** |

### Val Score

Métrica escalar única pra decisões de keep/discard:
- Com Coding: `80% Coding + 10% Nexus + 10% Claw`
- Sem Coding: `60% Nexus + 40% Claw`

TPS Floor = 20 tok/s. Abixo disso -> score zerado.

### Segurança

- Checagem de VRAM antes de subir o servidor
- Flash attention sempre ligado
- Todas as falhas logadas como `FAIL` no results.tsv, loop continua
- Nunca faz push pro remote

---

## Build do llama.cpp com CUDA

Clone dentro da pasta raiz do repo pra detecção automática:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build-cuda --config Release -j
```

Se clonou em outro lugar, exporte o path:

```bash
export AUTORESEARCH_LLAMA_CPP_ROOT="/caminho/pra/llama.cpp"
```

### Forks (TurboQuant / MTP)

Pra modos avançados de KV cache (`turbo2`, `turbo3`, `turbo4`, SPEC MTP):

- **TurboQuant**: `https://github.com/TheTom/llama-cpp-turboquant`
- **MTP & TurboQuant**: `https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant`

Clone como `llama.cpp` na raiz do repo. Comandos de build idênticos.

---

## Depois do Tuning: Subir o Modelo

```bash
# Mostra o comando (sem iniciar)
python3 scripts/serve-config.py print-cmd

# Sobe o llama-server detached
python3 scripts/serve-config.py serve

# Checa status
python3 scripts/serve-config.py status

# Para
python3 scripts/serve-config.py stop
```

Pluga no seu agente:

```
base_url: http://127.0.0.1:18080/v1
model:    <nome-do-modelo-do-config>
```

---

## Modo Manual

Se preferir fazer na mão:

1. Leia `program.md` pra regras
2. Edite `autoresearch/core/config.py` com uma hipótese
3. Rode `python3 benchmark_search.py --desc "sua hipótese"`
4. Cheque `results.tsv` pelos resultados
5. Keep se o Val Score melhorou, reverte caso contrário

---

## Profiles Suportados

| Profile | Benchmarks | Modelos Exemplo |
|---|---|---|
| **Coding** (default) | SWE-bench, Aider, HE+ | Qwen3.6-27B, Qwen3.6-35B-A3B |
| **Writing** | MMLU-Pro, Chatbot Arena | Qwen3-14B, Gemma3-12B |
| **Vision** | MMMU-Pro, MMBench | Qwen3-VL, Gemma-4-26B-A4B |

Troque em `autoresearch/core/config.py`:

```python
INCLUDE_CODING = True
INCLUDE_NEXUS = False
INCLUDE_CLAW = False
```

---

## Documentação pra Agentes

Agentes trabalhando neste repo, leiam nesta ordem:

1. `AGENTS.md` (root) — DOX hierarchy, work contracts
2. `program.md` — Regras do protocolo Search
3. `GOLDEN-RULES.md` — Flags de performance, segurança, validação
4. `CONTEXT.md` — Terminologia e definições
5. `docs/discovery/discover-models.md` — Workflow de seleção de modelo
6. `docs/discovery/whichllm-reference.md` — Referência CLI
7. `docs/discovery/quantization-cascade.md` — Seleção de formato de quant
8. `docs/llamacpp-toolset.md` — Referência dos binários do llama.cpp

---

## Licença

MIT
