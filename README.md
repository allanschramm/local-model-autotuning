# local-model-autoresearch

Um repo para tunar modelos locais via `llama.cpp`. A ideia é simples: subir o servidor, colocar um GGUF, rodar o benchmark, e ir ajustando até o modelo responder rápido e bem na sua máquina.



## O que este projeto faz

- Loop de avaliação reprodutível, sem surpresas.
- Tuning de runtime: KV cache, flash-attention, batch, ubatch, threads, MTP/TurboQuant e parâmetros de geração.
- Harness de avaliação fixo separado da superfície de busca.



## Estrutura

- `benchmark_search.py` / `benchmark_search_claw.py` — tuning.
- `benchmark_coding.py` — avaliação de código via EvalPlus.
- `prepare.py` / `prepare_claw.py` — contrato fixo de avaliação.
- `run_grid.py` — grade de `kv_cache` x `max_tokens`.
- `data/memory_fixture.json` — fixture determinística.
- `program.md` — o regulamento do loop.



## Pré-requisitos

Hardware:
- GPU NVIDIA com pelo menos 8 GB de VRAM (este projeto foi validado em RTX 4060 8GB).

Sistema:
- Linux ou WSL2 com CUDA.

Dependências:
- `git`
- `build-essential`
- `cmake >= 3.14`
- `python3.11+` + `uv` ou `pip`
- `llama-server` buildado com CUDA



## Setup rápido

1. Clone este repo.
2. Coloque o(s) modelo(s) `.gguf` em `models/`.
3. Aponte `AUTORESEARCH_LLAMA_CPP_ROOT` para um checkout do `llama.cpp` que tenha `build-cuda/bin/llama-server`.
4. Rode um teste rápido:

```bash
export AUTORESEARCH_MODELS_DIR="$PWD/models"
export AUTORESEARCH_LLAMA_CPP_ROOT="$HOME/llama.cpp"  # ajuste

python benchmark_search.py \
  --model gemma-4-E4B-it-Q4_K_M.gguf
```



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



## Modelos GGUF

Coloque arquivos `.gguf` em `models/`:

- `models/Qwen3.5-4B-Q4_K_M.gguf`
- `models/gemma-4-E4B-it-Q4_K_M.gguf`

Fontes comuns:
- Hugging Face (formato GGUF)
- conversores do próprio `llama.cpp`



## Variáveis que economizam tempo

- `AUTORESEARCH_MODELS_DIR` — pasta com `.gguf`.
- `AUTORESEARCH_LLAMA_CPP_ROOT` — checkout do `llama.cpp` com `build-cuda/bin/llama-server`.



## Parâmetros padrão

Edite em `benchmark_search.py`:

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

Exemplo rápido via CLI:
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



## Loop do agente

O fluxo está em `program.md`:

1. Leia `benchmark_search.py` como única superfície editável.
2. Faça uma mudança por vez.
3. Rode e compare `val_score` e `peak_vram_mb`.
4. Guarde o commit só se melhorou.



## Erros comuns

- `FAIL: llama-server not found` — confira `AUTORESEARCH_LLAMA_CPP_ROOT`.
- `FAIL: Model ... not found` — coloque o `.gguf` em `models/` ou use `--model`.
- OOM — reduza `--ctx-size`, use KV mais compacto ou modelo menor.
