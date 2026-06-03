# local-model-autoresearch

Repo público focado em usar e avaliar modelos locais com `llama.cpp`. O objetivo é deixar simples: tenha o `llama-server` rodando, coloque um modelo GGUF, faça a avaliação e depois ajuste parâmetros como cache quantization, MTP ou TurboQuant sem precisar caçar flags.

<br>
## O que este projeto faz

- Fornece um loop de avaliação reprodutível com fixture fixa.
- Expõe pontos de entrada editáveis para tuning de runtime: KV cache, flash-attention, batch/ubatch, threads, MTP/TurboQuant e parâmetros de geração.
- Mantém o harness de avaliação como contrato estável, separado da superfície de busca.

## Estrutura do repositório

```text
local-model-autoresearch/
  benchmark_search.py
  benchmark_search_claw.py
  benchmark_coding.py
  prepare.py
  prepare_claw.py
  run_grid.py
  evalplus_wrapper.py
  data/memory_fixture.json
  README.md
```

## Pré-requisitos

- Hardware recomendado: GPU NVIDIA com pelo menos 8 GB de VRAM (este projeto foi validado em RTX 4060 8GB).
- Sistema: Linux com CUDA instalado (WSL2 com Ubuntu também funciona).
- Dependências de sistema/build:
  - `git`
  - `build-essential` (gcc/g++/make)
  - `cmake` >= 3.14
  - `ninja-build` (opcional, mas recomendado)
  - `python3.11+` + `uv` ou `pip`
  - `curl` (opcional para verificação do servidor)
- `llama.cpp` buildado com CUDA:
  - `cmake` disponível no PATH
  - Toolkit CUDA instalado e detectável pelo CMake
  - Saída do build: `build-cuda/bin/llama-server`

## Primeiro uso

1. Clone o repositório.
2. Coloque o(s) modelos em `models/`.
3. Garanta que `AUTORESEARCH_LLAMA_CPP_ROOT` aponte para uma árvore do `llama.cpp` que contenha `build-cuda/bin/llama-server`.
4. Rode:

```bash
export AUTORESEARCH_MODELS_DIR="$PWD/models"
export AUTORESEARCH_LLAMA_CPP_ROOT="$HOME/workspace/Nexus-System/llama.cpp"

python benchmark_search.py --model gemma-4-E4B-it-Q4_K_M.gguf
```

## Build do `llama.cpp` com CUDA

Este projeto depende do `llama-server` buildado localmente. Segue um fluxo confiável, derivado da configuração em uso na máquina de referência:

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

Se precisar reduzir tempo de build, prefira `-DCMAKE_BUILD_TYPE=Release` e limitar jobs (`-j`) conforme CPU.

Atenção: `CMAKE_CUDA_ARCHITECTURES=89` está alinhado com a máquina de referência (Ampere/RTX 30xx/40xx compatível). Ajuste para sua GPU se necessário.

## Modelos GGUF

Coloque arquivos `.gguf` na pasta `models/`, por exemplo:

- `models/Qwen3.5-4B-Q4_K_M.gguf`
- `models/gemma-4-E4B-it-Q4_K_M.gguf`

Fontes comuns:
- Hugging Face em formato GGUF
- conversores oficiais do próprio `llama.cpp`

## Configurações importantes

Variáveis de ambiente (sem mexer no código a toda hora):

- `AUTORESEARCH_MODELS_DIR` — pasta com `.gguf`.
- `AUTORESEARCH_LLAMA_CPP_ROOT` — checkout do `llama.cpp` com `build-cuda/bin/llama-server`.

Flags padrão utilizadas (podem ser sobrescritas por script/git):

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

Se quiser testar rápido via CLI, use:

```bash
python benchmark_search.py \
  --model Qwen3.5-9B-Q4_K_M.gguf \
  --ctx-size 65536 \
  --kv-cache q4_0 \
  --max-tokens 512
```

## MTP, TurboQuant e otimizações relevantes

### MTP
Quando o nome do modelo contém `MTP`, o script já habilita o rastro `draft-mtp` automaticamente. O valor padrão atual usa `spec-draft-n-max=1` e o mesmo `KV_CACHE_TYPE` do modelo base.

### KV cache quantization
Ajuste o KV cache por parâmetro:

```bash
python benchmark_search.py --kv-cache q5_1
```

Valores comuns: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`.

### Throughput
O benchmark possui piso de throughput (30 TPS). Passar de `q4_0` para `q8_0` pode aumentar qualidade da geração, mas custa mais VRAM; testar com cautela.

## Scripts e papéis

- `benchmark_search.py` / `benchmark_search_claw.py` — tuning de runtime e prompts.
- `benchmark_coding.py` — avaliação de código via EvalPlus.
- `prepare.py` / `prepare_claw.py` — contratos fixos de avaliação.
- `run_grid.py` — executa uma grade de `kv_cache` x `max_tokens` para comparação.
- `data/memory_fixture.json` — fixture determinística para reprodução.

## FAQ rápido

- `FAIL: llama-server not found`: verifique `AUTORESEARCH_LLAMA_CPP_ROOT`.
- `FAIL: Model ... not found`: coloque o `.gguf` em `models/` ou use `--model`.
- OOM/VRAM alta: reduza `--ctx-size`, use KV cache mais compacto ou modelo menor.
- Quero usar outro modelo: apenas substitua o arquivo `.gguf` e informe em `--model`.
