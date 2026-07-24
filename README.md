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
2. Rodar `whichllm` ou `llmfit` pra listar candidatos e estimar o footprint de VRAM
3. Cruzar com SWE-bench / Aider / LiveCodeBench
4. Plotar Pareto frontier (tok/s vs qualidade)
5. Semear e editar o Baseline (`cp autoresearch/core/config.py.example autoresearch/core/config.py`, depois definir `MODEL`)
6. Rodar `python3 autoloop.py --vram-limit-mb=<budget>` overnight

**Resultado de manhã:** `results.tsv` com todos os Trials + `config.py` local com a melhor configuração (visited em `.autoresearch_state.json`).

---

## Pré-requisitos

Instale **antes** de pedir pro agente:

| Dep | Comando | Por quê |
|---|---|---|
| Python 3.11+ | `sudo apt install python3.11 python3.11-venv` | runtime do autoloop |
| CUDA Toolkit | `nvidia-smi` + driver NVIDIA | llama.cpp precisa de `-DGGML_CUDA=ON` |
| build-essential + cmake >= 3.14 | `sudo apt install build-essential cmake` | compilar llama.cpp |
| uvx / whichllm | `pip install uv` | rodar `uvx whichllm@latest` |
| llmfit | `cargo install llmfit` (ou `scoop install llmfit`) | dimensionamento de hardware e CLI/TUI Rust |
| huggingface_hub[cli] | `pip install huggingface_hub[cli]` | baixar GGUFs |

Depois clone e compile o `llama.cpp` (ver [seção Build](#build-do-llamacpp-com-cuda) abaixo).

### Baseline local (`config.py`)

O Baseline mutável **não vem no git** (fica só na sua máquina). Depois do clone:

```bash
cp autoresearch/core/config.py.example autoresearch/core/config.py
```

Edite `MODEL` (basename do GGUF em `models/`) e os knobs ENGINE/SAMPLER. O autoloop reescreve esse arquivo a cada keep — **não faça commit** dele.

### Verificar se tá tudo pronto

```bash
bash scripts/setup-check.sh
```

Output verde = pronto pro autoloop.

---

## 🎓 Curso Prático & Utilitários para Alunos

Este repositório inclui uma jornada publicada de 6 aulas em HTML (Módulo 0 + Semana 1). A Semana 2 permanece visível como currículo em construção:

* **Portal Interativo:** Abra [teach/index.html](teach/index.html) no navegador para acompanhar as 6 aulas, práticas e quizzes publicados.
* **Diagnóstico de Hardware:** `.\venv\Scripts\python.exe scripts\check_hardware.py` (orienta modelo GGUF, GPU e contexto sem estimar TPS).
* **Validação de Servidor e TPS:** `.\venv\Scripts\python.exe scripts\verify_setup.py --port 18080` (testa o servidor local e mede a velocidade real em tokens/s).

---

## Como Funciona

### O Loop

1. Lê o Baseline atual de `autoresearch/core/config.py`
2. Valida throughput e roda o smoke test Claw-Eval quick
3. Roda Claw-Eval full e calcula o Val Score agentic (+ TPS floor)
4. Muta um param -> gera config Neighbor
5. Avalia Neighbor -> keep se melhorou (ou Pareto tie-break)
6. Se local maxima -> random restart
7. Loop pra sempre até Ctrl+C

### Contrato de Edição

| Arquivo | O quê | Agente/loop pode editar? |
|---|---|---|
| `autoresearch/core/config.py` | Baseline local (gitignored; seed = `.example`) | **Sim** (via autoloop / edição manual) |
| `autoresearch/core/config.py.example` | Template versionado do Baseline | **Não** (só pra atualizar defaults genéricos) |
| `.autoresearch_state.json` | Visited memory (local) | **Sim** (só visited) |
| `autoresearch/benchmarks/bench_config.py` | Quais benches rodam | **Não** (só com permissão explícita) |
| `benchmark_search.py` | CLI runner | **Não** |
| `autoresearch/benchmarks/*` | Lógica de avaliação | **Não** |
| `program.md` | Protocolo do Search | **Não** |
| `results.tsv` | Métricas dos trials | **Só append** |

### Val Score

O Claw-Eval full é a métrica canônica para decisões de keep/discard. HE+, MBPP+, LCB e BigCodeBench são preflight opcional e, quando ativados, sempre usam 10 tarefas por dataset.

TPS Floor = `TPS_FLOOR` no Baseline (`config.py`, default 20 tok/s). Abaixo disso -> score zerado. MoE grande em 8GB: baixe o floor (ex.: 15).

Ranking local atual (8 GB): [docs/discovery/claw-eval-leaderboard.md](docs/discovery/claw-eval-leaderboard.md) — líder claw-full: Laguna-XS **0.6667** (2026-07-24).

### Preflight coding

HE+/MBPP+/LCB/BigCode ficam como preflight rápido opcional, não como medida final de agente:

```bash
python benchmark_search.py --list-agentic-benchmarks
```

### Segurança

- Checagem de VRAM antes de subir o servidor
- Flash attention sempre ligado
- Todas as falhas logadas como `FAIL` no results.tsv, loop continua
- Nunca faz push pro remote

---

## Build do llama.cpp (CUDA ou CPU)

Clone dentro da pasta raiz do repo pra detecção automática:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Opção A: Build com aceleração CUDA (GPU NVIDIA)
cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build-cuda --config Release -j

# Opção B: Build apenas CPU (sem necessidade de GPU)
cmake -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_SERVER=ON

cmake --build build-cpu --config Release -j
```

Se clonou em outro lugar, exporte o path:

```bash
export AUTORESEARCH_LLAMA_CPP_ROOT="/caminho/pra/llama.cpp"
```

### Windows nativo

Depois da migracao do WSL2, o harness tambem resolve builds nativos do Windows (tanto em `build-cuda` quanto em `build-cpu`):

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = "D:\Dev\Nexus-System\local-model-autotuning\llama.cpp"
python benchmark_search.py --validation --desc "validar modelo no Windows"
python scripts\serve-config.py print-cmd
python scripts\serve-config.py serve
```

O resolver procura `llama-server.exe` e `llama-bench.exe` em `build-cuda\bin`, `build-cpu\bin`, `build\bin` (e subpastas `Release`/`Debug`) e no `PATH`. O diretorio `models\` deve apontar para modelos locais do Windows, nao para paths `/mnt/...` ou WSL.

Runtime canônico: submodule `llama.cpp/` (upstream). Forks externos (TurboQuant/MTP) não ficam no repo — se precisar, clone à parte e aponte `AUTORESEARCH_LLAMA_CPP_ROOT`.

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
2. Se ainda não tiver: `cp autoresearch/core/config.py.example autoresearch/core/config.py`
3. Ajuste o Baseline em `autoresearch/core/config.py` (`MODEL` = basename do GGUF)
4. Rode `python3 benchmark_search.py --desc "sua hipótese"` (sem flag soup)
5. Cheque `results.tsv` pelos resultados
6. Keep se o Val Score melhorou, reverte o `config.py` caso contrário

---

## Profiles Suportados

| Profile | Benchmarks | Modelos Exemplo |
|---|---|---|
| **Agentic Coding** (default) | Claw-Eval full (Val Score) + quick smoke | Modelos locais via endpoint OpenAI-compatible |
| **Coding** (preflight opcional) | LiveCodeBench, HumanEval+, MBPP+, BigCodeBench Hard (10 tasks cada) | Qualquer modelo GGUF local |
| **Writing** | MMLU-Pro, Chatbot Arena | Qualquer modelo GGUF local |
| **Vision** | MMMU-Pro, MMBench | Qualquer modelo GGUF local multimodal |

Troque em `autoresearch/benchmarks/bench_config.py`:

```python
INCLUDE_CODING = False
INCLUDE_AGENTIC_QUICK = True
INCLUDE_AGENTIC_FULL = True
```

---

## Documentação pra Agentes

Agentes trabalhando neste repo, leiam nesta ordem:

1. `AGENTS.md` (root) — DOX hierarchy, work contracts
2. `program.md` — Regras do protocolo Search
3. `GOLDEN-RULES.md` — Flags de performance, segurança, validação
4. `CONTEXT.md` — Terminologia e definições
5. `docs/discovery/discover-models.md` — Workflow de seleção de modelo
6. `docs/discovery/whichllm-reference.md` — Referência CLI do whichllm
7. `docs/discovery/llmfit-reference.md` — Referência CLI/TUI do llmfit
8. `docs/discovery/quantization-cascade.md` — Seleção de formato de quant
9. `docs/llamacpp-toolset.md` — Referência dos binários do llama.cpp

---

## Licença

MIT
