# local-model-autoresearch

Auto-tuning de modelos locais via `llama.cpp`, baseado em [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Dois modos de uso:
- **Manual**: você edita `benchmark_search.py` e roda benchmarks diretamente.
- **Agente**: um agente externo (LLM com acesso a arquivos + terminal) lê `PROGRAM.md`, edita `benchmark_search.py` e itera automaticamente até achar a melhor configuração.



## O que este projeto faz

- Loop de avaliação reprodutível, sem surpresas.
- Tuning de **runtime** do `llama-server`: KV cache, flash-attention, batch, ubatch, threads, MTP/TurboQuant e parâmetros de geração. **Não re-quantiza o GGUF** — muda apenas como o modelo é servido.
- Dois harnesses de avaliação: fixture determinística interna (`benchmark_search.py`) e benchmark público ClawEval (`benchmark_search_claw.py`).



## Estrutura

```
autoresearch-public/
├── benchmark_search.py         # Tuning: fixture interna (data/memory_fixture.json)
├── benchmark_search_claw.py    # Tuning: benchmark público ClawEval (https://github.com/claw-eval/claw-eval)
├── benchmark_coding.py         # Avaliação de código via EvalPlus
├── prepare.py                  # Preparação fixa para fixture interna
├── prepare_claw.py             # Preparação fixa para ClawEval
├── run_grid.py                 # Grade de busca: kv_cache × max_tokens
├── data/memory_fixture.json    # Fixture determinística para avaliação
├── program.md                  # PROMPT DO AGENTE — coração do sistema
└── models/                     # Coloque seus .gguf aqui
```

**`program.md`** não é documentação genérica. É o prompt/contrato que o agente externo segue cegamente. Ele só tem acesso a:
- Este arquivo (`program.md`)
- `benchmark_search.py` (única superfície editável)
- Output dos comandos que executa

Se você está usando o modo agente, leia `program.md` antes de tudo.



## Arquitetura

O sistema é um loop de agente autônomo baseado em [karpathy/autoresearch](https://github.com/karpathy/autoresearch). O agente externo (LLM com acesso a arquivos + terminal) itera sobre configurações de runtime do `llama-server` até maximizar uma métrica composta.

### Papel do agente externo

O agente NÃO é um script dentro deste repo. É um LLM externo (Claude Code, Codex, etc) que:

1. Lê `program.md` — único prompt/contrato do experimento
2. Tem acesso apenas a:
   - Leitura de arquivos do repo
   - Execução de comandos (`python benchmark_search.py`, `git`, etc)
3. Edita **apenas** `benchmark_search.py`
4. Itera: edita → commita → roda benchmark → compara métricas → decide se mantém ou reverte

### Fluxo de avaliação (dual-pass)

Cada execução de `benchmark_search.py` roda duas avaliações fixas:

**Pass 1 — Retrieval (peso 0.40)**
- Usa `prepare.py` + fixture `data/memory_fixture.json`
- ~50k tokens de contexto sintético injetados como ruído
- Mede capacidade de encontrar token de override em memória

**Pass 2 — Agency (peso 0.60)**
- Usa `prepare_claw.py` + benchmark público ClawEval (https://github.com/claw-eval/claw-eval)
- 11 tarefas de tool-use (formatação JSON) e instruction-following
- Sem ruído de contexto

### Métrica primária

```
val_score = (val_agency × 0.6) + (val_retrieval × 0.4)
```

Penalidade: se `tokens_per_second` médio < 30 TPS, `val_score = 0.0` (descartado).

### Métricas secundárias

| Métrica | O que mede |
|---|---|
| `val_retrieval` | Score isolado do Pass 1 |
| `val_agency` | Score isolado do Pass 2 |
| `tokens_per_sec` | Throughput combinado |
| `total_seconds` | Tempo wall-clock |
| `peak_vram_mb` | VRAM de pico |
| `ctx_size` | Tamanho do contexto |
| `kv_cache` | Tipo de KV cache quantizado |

### Superfície de edição (contrato)

| Arquivo | Papel | Editável? |
|---|---|---|
| `program.md` | Contrato do agente | **Não** (fixo após baseline) |
| `prepare.py` | Harness Retrieval | **Não** |
| `prepare_claw.py` | Harness Agency | **Não** |
| `benchmark_search.py` | Tuning de runtime | **Sim** (único arquivo editável) |
| `data/memory_fixture.json` | Dados de teste | **Não** |
| `run_grid.py` | Grade de busca auxiliar | Opcional (pode ser usado para exploração inicial) |

### Restrições hard

- Hardware alvo: **RTX 4060 8GB**
- Contexto: **128k tokens** obrigatório (`ctx_size = 131072`)
- VRAM safety: se `peak_vram_mb >= 7900`, config é considerada insegura
- `--n-gpu-layers 999` deve ser mantido (sem CPU offload oculto)
- Não é permitido alterar fixture, scoring logic ou adicionar dependências

### Output esperado

Cada run bem-sucedido imprime:

```
---
val_score:        0.374961
val_retrieval:    0.624950
val_agency:       0.208302
tokens_per_sec:   138.77
total_seconds:    263.3
peak_vram_mb:     7762.0
ctx_size:         131072
kv_cache:         q4_0
model:            Qwen3.5-9B-Coder-MTP-Q4_K_M.gguf
eval_seconds:     255.425
```

### Logging e versionamento

- results.tsv: arquivo local (`.gitignore`) com colunas `commit\tval_score\tmemory_gb\tstatus\tdescription`
- O agente commita apenas se `val_score` melhorou; caso contrário, reverte para o melhor commit anterior
- Branch recomendado: `git checkout -b autoresearch/<tag>` a partir de `main`



## Pré-requisitos

Hardware:
- GPU NVIDIA com pelo menos 8 GB de VRAM (validado em RTX 4060 8GB).

Sistema:
- Linux ou WSL2 com CUDA.

Dependências:
- `git`
- `build-essential`
- `cmake >= 3.14`
- `python3.11+`
- `llama-server` buildado com CUDA



## Setup rápido

1. Clone este repo.
2. Instale dependências Python:
   ```bash
   pip install -r requirements.txt
   ```
   Se não houver `requirements.txt`, gere com:
   ```bash
   pip freeze > requirements.txt
   ```
3. Crie a pasta de modelos se não existir:
   ```bash
   mkdir -p models
   ```
4. Coloque o(s) modelo(s) `.gguf` em `models/`.
5. Aponte as variáveis de ambiente:
   ```bash
   export AUTORESEARCH_MODELS_DIR="$PWD/models"
   export AUTORESEARCH_LLAMA_CPP_ROOT="$HOME/llama.cpp"  # ajuste para seu path
   ```
6. Rode um teste rápido:
   ```bash
   python benchmark_search.py \
     --model gemma-4-E4B-it-Q4_K_M.gguf
   ```

**Se der erro `FAIL: llama-server not found`**: confira se `AUTORESEARCH_LLAMA_CPP_ROOT` aponta para um checkout do `llama.cpp` com `build-cuda/bin/llama-server`.



## Build do `llama.cpp` com CUDA

Fluxo confiável, derivado da máquina de referência:

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

Verificação:
```bash
./build-cuda/bin/llama-server --help | head
```

Dica: se demorar muito, reduza os jobs do build. `CMAKE_CUDA_ARCHITECTURES=89` cobre bem placas Ampere/RTX 30xx e 40xx. Ajuste para sua GPU se precisar.

### Forks com TurboQuant

Se quiser experimentar compressão de KV cache via TurboQuant, dois forks públicos mantêm essa implementação em cima do `llama.cpp`:

- [BoFan-tunning/llama.cpp-MTP-TurboQuant](https://github.com/BoFan-tunning/llama.cpp-MTP-TurboQuant) — MTP + TurboQuant com CUDA (branch `merge-mtp-turboquant`). Bom ponto de partida se você quer testar a fusão dos dois.
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — implementação de TurboQuant no `llama.cpp`, com suporte AMD/RDNA via Vulkan além de CUDA (branch `feature/turboquant-kv-cache`).

Ambos são forks públicos; verifique a compatibilidade com seu modelo/GPU antes de adotar.



## Modelos GGUF

Coloque arquivos `.gguf` em `models/`:

```
models/Qwen3.5-4B-Q4_K_M.gguf
models/gemma-4-E4B-it-Q4_K_M.gguf
```

Fontes comuns:
- Hugging Face (formato GGUF)
- conversores do próprio `llama.cpp`

**Importante**: o tuning NÃO modifica o `.gguf`. Você está ajustando parâmetros de runtime do `llama-server` (KV cache, batch, threads, etc), não re-quantizando o modelo.



## Variáveis de ambiente

| Variável | O que faz |
|---|---|
| `AUTORESEARCH_MODELS_DIR` | Pasta com `.gguf`. Default: `$PWD/models`. |
| `AUTORESEARCH_LLAMA_CPP_ROOT` | Checkout do `llama.cpp` com `build-cuda/bin/llama-server`. |

Ambas são opcionais se você passar `--model` com path absoluto.



## Parâmetros padrão

Editáveis em `benchmark_search.py` (CLI sobrescreve):

```text
CTX_SIZE = 131072
KV_CACHE_TYPE = q4_0
BATCH_SIZE = 512
UBATCH_SIZE = 128
THREADS = 8
PARALLEL = 1
NGL = 999
FLASH_ATTN = on
```

Exemplo rápido via CLI (funciona sem editar o arquivo):
```bash
python benchmark_search.py \
  --model Qwen3.5-9B-Q4_K_M.gguf \
  --ctx-size 65536 \
  --kv-cache q4_0 \
  --max-tokens 512
```



## MTP, KV cache e throughput

### MTP
Quando o modelo tem `MTP` no nome, o script já ativa `draft-mtp` automaticamente.

### KV cache quantizado
Use valores como `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`. KV mais compacto ajuda a caber em 8GB; KV maior aumenta qualidade a troco de VRAM.

### Throughput
O benchmark tem piso de 30 TPS. Se ficar abaixo disso, o `val_score` vai para zero.



## O que esperar na primeira execução

Saída esperada (exemplo):
```
val_score: 0.87
peak_vram_mb: 7120
tokens_per_second: 34.2
```

- `val_score`: 0-1, quanto mais alto melhor.
- `peak_vram_mb`: VRAM usada no pico.
- `tokens_per_second`: throughput. Piso em 30, abaixo disso val_score=0.

Resultados são salvos automaticamente pelo agente ou podem ser capturados do stdout.



## Loop do agente (modo autônomo)

Se você está usando um agente externo (ex: Claude Code, Codex):

1. Leia `program.md` — é a única fonte de verdade do que o agente deve fazer.
2. O agente edita **apenas** `benchmark_search.py`.
3. Uma mudança por vez.
4. Roda o benchmark, compara `val_score` e `peak_vram_mb`.
5. Commit apenas se melhorou.

Uso manual: ignore `program.md`, edite `benchmark_search.py` diretamente e rode os comandos você mesmo.



## Erros comuns

| Erro | Causa | Solução |
|---|---|---|
| `FAIL: llama-server not found` | `AUTORESEARCH_LLAMA_CPP_ROOT` errado | Confira o path do `llama.cpp` |
| `FAIL: Model ... not found` | `.gguf` não está em `models/` | Use `--model` ou mova o arquivo |
| OOM | Contexto muito grande para a VRAM | Reduza `--ctx-size`, use KV mais compacto (`q4_0`) ou modelo menor |
| `ModuleNotFoundError` | Dependências Python não instaladas | Rode `pip install -r requirements.txt` |
| `FileNotFoundError: models/` | Pasta não existe | Rode `mkdir -p models` antes |
