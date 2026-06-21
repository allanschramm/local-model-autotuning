# local-model-autoresearch

Auto-tuning de modelos locais via `llama.cpp`, baseado em [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Sistema autônomo (hill-climbing) que otimiza flags de runtime de LLMs locais avaliando configurações repetidamente e mantendo as melhorias.

**Dois modos de uso:**
- **Loop Interno**: O script `autoloop.py` realiza uma busca automática (Search) pelas melhores configurações editando `autoresearch/core/config.py`.
- **Agente Externo**: Um agente externo (LLM autônomo com acesso ao repositório) lê `program.md` (seu entrypoint absoluto) para obter as regras e objetivos, e a partir disso executa as rodadas de testes modificando unicamente a superfície de tuning em `autoresearch/core/config.py`.

> **Fluxo de Trabalho Usuário-Agente Importante:**
> 1. **O Usuário Humano** edita manualmente as diretrizes de escopo, tags de branch e configurações iniciais exclusivamente em `program.md`.
> 2. **O Agente Autônomo** lê as instruções de `program.md` e inicia o loop de benchmarks, atualizando `autoresearch/core/config.py` a cada melhoria.
> 3. **O Agente NUNCA deve editar `program.md`.** Este arquivo é o canal exclusivo de controle do usuário humano.


## O que este projeto faz

- Loop de avaliação reprodutível, sem surpresas.
- Tuning de **runtime** do `llama-server`: KV cache, flash-attention, batch, ubatch, threads, MTP/TurboQuant e parâmetros de geração. **Não re-quantiza o GGUF** — muda apenas como o modelo é servido.
- **Avaliação Unificada**: Cada rodada (Trial) executa todos os benchmarks ativos simultaneamente (Nexus para Retrieval, Claw para Agency, e opcionalmente Coding).

## Estrutura

```
autoresearch-public/
├── benchmark_search.py         # Unified AutoResearch Benchmark Runner (CLI)
├── autoloop.py                 # Loop autônomo automatizado (SearchStrategy)
├── autoresearch/               # Core package
│   ├── core/                   # LLM configs and runner
│   │   └── config.py           # ÚNICA superfície de edição do tuning (Baseline/Neighbor)
│   ├── benchmarks/             # Lógica de avaliação (Nexus, Claw, Coding)
│   └── runners/                # Running and tuning logic
├── scripts/                    # Utilitários
├── GOLDEN-RULES.md             # Regras estritas do projeto (Performance, Restrições)
├── CONTEXT.md                  # Terminologia e definições do domínio
├── AGENTS.md                   # Diretrizes para agentes LLM
└── models/                     # Coloque seus arquivos .gguf aqui
```

## Arquitetura e Terminologia

O sistema segue rigidamente as diretrizes definidas em `CONTEXT.md` e `program.md`. A terminologia principal abrange:

- **Search**: O processo geral de otimização contínua.
- **Round**: Uma iteração individual do Search.
- **Trial**: Execução atômica completa de todos os benchmarks contra uma única configuração.
- **Baseline**: A configuração atual que obteve a melhor pontuação. Mantida em `autoresearch/core/config.py`.
- **Neighbor**: Uma configuração gerada a partir do Baseline, modificando apenas um parâmetro (ex: alterar threads de 8 para 12).

### Fluxo de Avaliação Unificada

Em vez de testar domínios de forma isolada, cada Round requer que se execute todos os benchmarks habilitados:

1. **Nexus (Retrieval)**: Testa context-stress com histórico sintético. Exige que o modelo recupere tokens de override em memória.
2. **Claw (Agency)**: Testa tool-use (chamadas via JSON) e instruction-following em tarefas simuladas de browser.
3. **Coding (Opcional)**: Avalia a capacidade de geração de código Python com EvalPlus (HumanEval+ e MBPP+).

### Métrica Primária (Val Score)

O `Val Score` é a métrica escalar única que orienta decisões de keep/discard.
- Se Coding estiver habilitado: `80% Coding + 10% Nexus + 10% Claw`
- Sem Coding: `60% Claw + 40% Nexus`

**Throughput e Penalidades (TPS)**: 
- Existe um rendimento mínimo definido pelo **TPS Floor** (ex: 20 TPS). Uma avaliação abaixo dessa linha terá seu `Val Score` forçado a zero, independentemente de sua acurácia.
- Um **Speed Factor** aplica uma penalidade escalonada para o rendimento que fica entre o piso e o TPS esperado.

### Superfície de edição (Contrato de Tuning)

| Arquivo | Papel | Agente ou Loop Pode Editar? |
|---|---|---|
| `autoresearch/core/config.py` | Configurações de runtime do LLM | **Sim** (Apenas constantes) |
| `benchmark_search.py` | CLI para acionar testes | **Não** |
| `autoresearch/benchmarks/*` | Lógica interna de avaliação | **Não** |
| `results.tsv` | Arquivo canônico com as métricas do Trial | **Apenas append automático** |

> **Nota:** Todos os logs de performance ficam concentrados apenas no `results.tsv`. Nenhuma outra saída local ou log alternativo é permitido.

### Hardware Failsafes

- **VRAM Safety**: Se a estimativa de uso (`peak_vram_mb`) exceder as especificações do hardware alvo (ex: GPU de 8GB), a configuração deve ser pulada ou rodar via CPU offload para prevenir instabilidade/OOMs.
- **Flash Attention**: A flag `-fa` (Flash Attention) nunca deve ser desligada (`off`), devendo permanecer `on` sempre que possível.
- **Shared Memory & CPU Offload**: Offloads parciais para CPU ou uso de memória RAM compartilhada pelo driver de vídeo são suportados, mas penalizados. Se a taxa de processamento cair abaixo do **TPS Floor** (20 TPS), a pontuação (`Val Score`) é zerada.
- **Resiliência do Loop**: Falhas de inicialização do servidor, parâmetros inválidos ou quedas do runner são tratadas no nível do Trial, gravando `FAIL` no `results.tsv` sem interromper o loop de busca.


## 🚀 Quickstart: Agent-Driven Discovery + Tuning

**Recomendado para a maioria dos usuários.** Abra seu agente de coding favorito (Claude Code, Codex, Pi Agent, OpenCode) e peça:

> *"Descubra o melhor modelo para `<coding|writing|vision|ocr>` que cabe no meu PC, baixe e comece o auto-tuning."*

O agente vai seguir o workflow documentado em `docs/discovery/discover-models.md`:
1. Detectar hardware (GPU/VRAM/RAM)
2. Rodar `whichllm` para shortlist
3. Cruzar com SWE-bench / Aider / LiveCodeBench (coding) ou outros benchmarks por perfil
4. Plotar **Pareto frontier** (tok/s vs qualidade) e escolher o ponto ótimo
5. Editar `autoresearch/core/config.py` com o modelo escolhido
6. Subir `python3 autoloop.py --vram-limit-mb=<budget>` pra rodar overnight

Resultado de manhã: `results.tsv` com trials tentados + `config.py` no melhor ponto descoberto.

### Pré-requisitos

Antes de pedir pro agente, instale manualmente:

| Dep | Comando | Por quê |
|---|---|---|
| Python 3.11+ | `sudo apt install python3.11 python3.11-venv` | runtime do autoloop |
| CUDA Toolkit | `nvidia-smi` + driver NVIDIA | llama.cpp precisa compilar com `-DGGML_CUDA=ON` |
| `build-essential` + `cmake ≥ 3.14` | `sudo apt install build-essential cmake` | compilar llama.cpp |
| `uvx` (ou `uv`) | `pip install uv` | rodar `uvx whichllm@latest` |
| `huggingface_hub[cli]` | `pip install huggingface_hub[cli]` | baixar GGUF do HuggingFace |

Depois clonar e compilar o `llama.cpp` (fork recomendado pro auto-tuning sério, ver seção abaixo).

### One-liner de verificação

Antes de deixar o agente operar, rode:

```bash
bash scripts/setup-check.sh
```

Output verde = tudo OK pra rodar autoloop.

## 📦 Pré-requisitos Detalhados

- **Hardware**: GPU NVIDIA com ao menos 8 GB de VRAM (validado em modelo RTX 4060 8GB). Funciona também em Apple Silicon (Metal) e AMD (ROCm), com menor performance.
- **Sistema Operacional**: Linux ou WSL2 com drivers CUDA instalados. macOS e Windows nativo suportados via fork Metal/ROCm do llama.cpp.
- **Python**: Python 3.11+ (necessário para orchestrator, benchmark e autoloop).
- **HuggingFace CLI**: `huggingface-cli` (instalado via `pip install huggingface_hub[cli]`) para download e gerenciamento dos modelos.
- **Dependências de Build**: `git`, `build-essential`, `cmake >= 3.14`.
- **Servidor LLM**: O `llama-server` oriundo do repositório `llama.cpp` buildado explicitamente com suporte a CUDA.
- **`uvx` (Astral uv)**: para `uvx whichllm@latest` (discovery de modelos).
- **WSL `.wslconfig`** (recomendado pra inferência pesada em WSL2): limitar memória e adicionar swap.

```ini
# C:\Users\<you>\.wslconfig
[wsl2]
memory=24GB
swap=8GB
```

## 🛠️ Setup Manual Passo-a-Passo

```bash
# 1. Clone o projeto
git clone https://github.com/<owner>/local-model-autoresearch.git
cd local-model-autoresearch

# 2. Instale deps Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Crie diretório de modelos
mkdir -p models

# 4. Clone e compile llama.cpp (fork recomendado pra MTP/TurboQuant)
# Veja seção "Build do llama.cpp" abaixo
git clone https://github.com/<fork-owner>/llama.cpp.git

# 5. Rode o setup-check
bash scripts/setup-check.sh

# 6. (Opcional) Descubra modelos pro seu hardware
bash scripts/discover-models.sh

# 7. Edite config.py com o modelo escolhido
#    (o agente pode fazer isso se você tiver um agente de coding rodando)
$EDITOR autoresearch/core/config.py
# Set: MODEL = '<arquivo>.gguf'

# 8. Rode o autoloop
bash scripts/run-autoloop.sh
```

## 🎯 After-Tuning: Subir o Modelo pra Uso

Quando o autoloop termina (ou você quer testar uma config manualmente), basta:

```bash
# Mostra o comando que vai rodar (sem iniciar nada)
python3 scripts/serve-config.py print-cmd

# Inicia llama-server com a config atual, detached (sobrevive a fechar terminal)
python3 scripts/serve-config.py serve

# Verifica se tá rodando
python3 scripts/serve-config.py status

# Mata
python3 scripts/serve-config.py stop
```

Output típico do `status`:

```
Running: PID=12345 alias=gemma-4-26B-A4B-it-UD-Q4_K_M port=18080 host=127.0.0.1
  Health: OK
  Endpoint:    http://127.0.0.1:18080/v1
  Model name:  gemma-4-26B-A4B-it-UD-Q4_K_M

Plug into Pi Agent / Hermes Agent / Claude Code:
  base_url: http://127.0.0.1:18080/v1
  model:    gemma-4-26B-A4B-it-UD-Q4_K_M
```

`serve-config.py` lê `autoresearch/core/config.py` (AST-based — só pega literais string/int/float/bool), monta o comando `llama-server`, e sobe detached via `subprocess.Popen(start_new_session=True)`. Não precisa criar alias YAML manualmente.

**Fluxo completo**:
1. `bash scripts/run-autoloop.sh` → autoloop roda, escreve `config.py` no melhor baseline
2. `python3 scripts/serve-config.py serve` → sobe llama-server com esse config
3. Plug no Pi Agent / Hermes Agent / Claude Code com `base_url=http://127.0.0.1:18080/v1`
4. (Opcional) Repita enquanto autoloop melhora — a cada keep, restart do server com novo config

**Aliases persistentes (avançado)**: se você quiser um alias com nome amigável em vez de derivado do filename, crie `models/aliases/<name>/config.yaml` (schema em `models/aliases/INDEX.md`). Use `~/.local/bin/qwen-up <name>` pra subir. Esse fluxo é independente do `serve-config.py`.

## Build do `llama.cpp` com suporte a CUDA

Para que o loop funcione, você precisa compilar o `llama.cpp`. Certifique-se de ter o CUDA Toolkit instalado (`nvcc --version` deve funcionar).

### Passo a Passo de Compilação
Recomendamos clonar o `llama.cpp` **diretamente dentro da pasta raiz deste repositório** para que a detecção do binário seja 100% automática, sem precisar configurar variáveis de ambiente:

```bash
# Clone dentro da pasta raiz do projeto
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Configure o build.
# Use native para que o cmake detecte automaticamente a arquitetura da sua GPU
cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

# Compile
cmake --build build-cuda --config Release -j
```

> **Dica de Instalação Alternativa**: Se você preferir clonar o `llama.cpp` em outra pasta do sistema, exporte o caminho absoluto dele no seu terminal:
> `export AUTORESEARCH_LLAMA_CPP_ROOT="/caminho/para/llama.cpp"`

### Compilando forks (TurboQuant / MTP)
Para testar os modos avançados de KV cache (como `turbo2`, `turbo3`, `turbo4` e SPEC MTP), você precisará clonar forks específicos que possuem essas implementações de cache:
- **TurboQuant Fork**: `https://github.com/TheTom/llama-cpp-turboquant`
- **MTP & TurboQuant Fork**: `https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant`

Os comandos de compilação com CMake são idênticos:
```bash
# Clone o fork com nome 'llama.cpp' na pasta raiz do projeto
git clone https://github.com/TheTom/llama-cpp-turboquant.git llama.cpp
cd llama.cpp

# Configure e compile normalmente
cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native

cmake --build build-cuda --config Release -j
```

> **Por que fork?** O upstream `ggml-org/llama.cpp` aceita `--spec-type mtp` mas o autoloop local (`llama_runner.py`) faz probe do `--help` e prefere a flag específica do build. Forks TurboQuant expõem `--cache-type-k turbo2..4` que o upstream não tem. O `scripts/setup-check.sh` valida que seu build suporta os flags esperados.



## 📚 Workflow de Discovery (referência completa)

O workflow agent-driven resumido na seção Quickstart é detalhado em [`docs/discovery/discover-models.md`](docs/discovery/discover-models.md). Cobre:

- Como rodar `uvx whichllm@latest` corretamente (flags, filtros)
- Por que `whichllm` score NÃO é coding benchmark (caveat crítico)
- Como cruzar com SWE-bench / Aider / LiveCodeBench (coding)
- Como plotar Pareto frontier (tok/s vs qualidade)
- Como fazer handoff pro autoloop

## 🎯 Profiles Suportados

O autoloop atualmente roda **Coding** por padrão, mas a arquitetura suporta múltiplos perfis via `docs/discovery/`:

| Perfil | Benchmarks principais | Modelos candidatos |
|---|---|---|
| **Coding** (default) | SWE-bench Verified, Aider polyglot, HumanEval+ | Qwen3.6-27B, Qwen3.6-35B-A3B, Qwen3-30B-A3B, gpt-oss-20b |
| **Writing** | MMLU-Pro, Chatbot Arena ELO | Qwen3-14B, Gemma3-12B, GLM-4.7-Flash |
| **Vision** | MMMU-Pro, MMBench, RealWorldQA | Qwen3-VL, Gemma-4-26B-A4B (multimodal) |
| **OCR** | OCR-specific benchmarks, DocVQA | Specialist models (multimodal) |

Para mudar perfil, edite os includes em `autoresearch/core/config.py`:

```python
INCLUDE_CODING = True
INCLUDE_NEXUS = True   # Retrieval
INCLUDE_CLAW = False   # Agency (desabilitado por padrão)
CODING_TASK_LIMIT = 30  # Tasks per benchmark dataset
```

## Como Trabalhar com os Modelos GGUF

Coloque quaisquer arquivos modelo com a extensão `.gguf` na raiz do `/models/`.  
**Aviso Crítico:** O Search processa e faz o tuning de parâmetros *runtime* via servidor (KV cache, context limits e paralelos). O arquivo modelado original (`.gguf`) **NÃO sofre alterações**. Nenhuma re-quantização real é aplicada sobre os arquivos `.gguf`.

## 📊 Expectativas em Tempo de Execução

Se a sua primeira execução finalizar normalmente, um novo registro em `results.tsv` surgirá. No terminal o comportamento esperado assemelha-se a:
```text
val_score:        0.870000
memory_gb:        7.1
status:           keep
```

Para rodar overnight e acordar com resultados:

```bash
# Manhã
tail -50 results.tsv  # Vê os trials da noite
cat autoresearch/core/config.py  # Config final do melhor baseline
```

## 🤖 Como o Agente ou Usuário Manual Atuam

**Modo agente** (recomendado):
1. Abra Claude Code / Codex / Pi Agent apontando pro repo
2. Peça: "descubra o melhor modelo de `<profile>` pro meu PC, baixe e comece o auto-tuning"
3. O agente lê `program.md` para regras, segue `docs/discovery/discover-models.md` pra workflow
4. Resultado de manhã: `results.tsv` + config atualizado

**Modo manual** (sem agente):
1. Leia a fundo `program.md`.
2. As mudanças hipotéticas (Neighbors) residem apenas em `autoresearch/core/config.py`.
3. Inicie os benchmarks com o executor unificado `python benchmark_search.py`.
4. Analise as linhas inseridas no final do `results.tsv`.
5. Se houver avanço no `Val Score` — ou ocorrência de ganho no Pareto Tie-Breaker (+5% de TPS ou -5% em VRAM em empate) —, transforme o Neighbor na sua nova configuração Baseline.

## 📜 Licença

MIT — mesma do qualllm, karpathy/autoresearch, e da maioria dos componentes open-source usados aqui.
