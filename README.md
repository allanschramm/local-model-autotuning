# local-model-autoresearch

Auto-tuning de modelos locais via `llama.cpp`, baseado em [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Sistema autônomo (hill-climbing) que otimiza flags de runtime de LLMs locais avaliando configurações repetidamente e mantendo as melhorias.

Dois modos de uso:
- **Loop Interno**: O script `autoloop.py` realiza uma busca automática (Search) pelas melhores configurações editando `autoresearch/core/config.py`.
- **Agente Externo**: Um agente externo (LLM com acesso ao repo) lê APENAS `program.md` (o entrypoint absoluto), edita `autoresearch/core/config.py` e itera até achar a melhor configuração.

> **Aviso Importante:** Leia todo o readme para entender o processo. Altere o `program.md` manualmente para não causar um loop desnecessário; se solicitar para algum modelo externo alterar o `program.md`, ele pode vir a executar o loop de pesquisa sem que você queira.

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

## Pré-requisitos

- **Hardware**: GPU NVIDIA com ao menos 8 GB de VRAM (validado em modelo RTX 4060 8GB).
- **Sistema Operacional**: Linux ou WSL2 com drivers CUDA instalados.
- **Dependências de Build**: `git`, `build-essential`, `cmake >= 3.14`, `python3.11+`.
- **Servidor LLM**: O `llama-server` oriundo do repositório `llama.cpp` buildado explicitamente com suporte a CUDA.

## Setup Rápido

1. Clone o projeto para o seu ambiente.
2. Instale os requerimentos do Python:
   ```bash
   pip install -r requirements.txt
   ```
3. Crie o diretório para os modelos LLM, caso não exista:
   ```bash
   mkdir -p models
   ```
4. Baixe ou copie seus modelos `.gguf` para `models/`.
5. Identifique e configure as variáveis de ambiente necessárias para o projeto:
   ```bash
   export AUTORESEARCH_MODELS_DIR="$PWD/models"
   export AUTORESEARCH_LLAMA_CPP_ROOT="$HOME/llama.cpp" # Ajuste para o local em que você compilou o llama.cpp
   ```
6. Realize o primeiro Trial para confirmar se tudo está funcionando:
   ```bash
   python benchmark_search.py --model Qwen3.5-9B-Coder-MTP-Q4_K_M.gguf
   ```

## Build do `llama.cpp` com suporte a CUDA

Fluxo seguro usando o repositório base:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda --config Release -j
```

Para TurboQuant e MTP, recomenda-se explorar repositórios paralelos e forks mantidos de perto com essas propriedades ativadas (ex. `BoFan-tunning/llama.cpp-MTP-TurboQuant` ou `TheTom/llama-cpp-turboquant`).

## Como Trabalhar com os Modelos GGUF

Coloque quaisquer arquivos modelo com a extensão `.gguf` na raiz do `/models/`.  
**Aviso Crítico:** O Search processa e faz o tuning de parâmetros *runtime* via servidor (KV cache, context limits e paralelos). O arquivo modelado original (`.gguf`) **NÃO sofre alterações**. Nenhuma re-quantização real é aplicada sobre os arquivos `.gguf`.

## Expectativas em Tempo de Execução

Se a sua primeira execução finalizar normalmente, um novo registro em `results.tsv` surgirá. No terminal o comportamento esperado assemelha-se a:
```text
val_score:        0.870000
memory_gb:        7.1
status:           keep
```

## Como o Agente ou Usuário Manual Atuam

Caso não se utilize o utilitário autônomo `autoloop.py`, siga este passo-a-passo:
1. Leia a fundo `program.md`.
2. As mudanças hipotéticas (Neighbors) residem apenas em `autoresearch/core/config.py`.
3. Inicie os benchmarks com o executor unificado `python benchmark_search.py`.
4. Analise as linhas inseridas no final do `results.tsv`.
5. Se houver avanço no `Val Score` — ou ocorrência de ganho no Pareto Tie-Breaker (+5% de TPS ou -5% em VRAM em empate) —, transforme o Neighbor na sua nova configuração Baseline.
