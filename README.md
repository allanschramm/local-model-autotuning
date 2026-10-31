# Local model auto-tuning

Otimize o runtime de um modelo GGUF para **TPS × contexto configurado**, sob um teto fixo de PPL. A Search é finita, preserva o contexto mínimo e mantém alternativas Pareto.

Consulte o [histórico de versões](CHANGELOG.md) para mudanças, migração e limites de validação.

## Instalação e execução

Crie uma venv com Python 3.11+ e instale o runtime Python:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item autoresearch/core/config.py.example autoresearch/core/config.py
```

Em Linux/macOS use `python3`, `./venv/bin/python` e `cp`. A cópia do Baseline é para clones novos: preserve um `config.py` existente.

Provisione um runtime llama.cpp pré-compilado compatível e defina `AUTORESEARCH_LLAMA_CPP_ROOT`. A referência é `llama.cpp-releases/upstream/b11577`; runtime e pesos não acompanham o clone. Veja [o layout dos modelos](models/README.md).

Configure o GGUF, engine e sampler no Baseline local. Revise o [plano de exemplo](autoresearch/tuning-plan.example.json), preservando o maior valor entre o contexto do Baseline e o mínimo exigido pelo seu uso:

```powershell
.\venv\Scripts\python.exe -m autoresearch.tuning --plan autoresearch/tuning-plan.example.json
```

O protocolo mede PPL com referência fixa e TPS num servidor monitorado com a alocação completa de contexto. Alocação e ocupação do prompt são evidências diferentes. Veja [TUNING.md](autoresearch/TUNING.md) para método, limites e seleção.

Tentativas e fronteira ficam em `results.db`; evidências e Fingerprints ficam em `tuning-artifacts/`. Baseline e seleção do launcher não são substituídos automaticamente. O dashboard somente leitura inicia com `python -m ui` pela venv, em `http://127.0.0.1:18765`.

## Escopo

Este repositório entrega auto-tuning, proteções de recursos, provisionamento e documentação de performance. Claw/Coding/SWE/DM, rankings de qualidade e comandos antigos de Trial foram retirados. O benchmark com Pi permanece na branch `staging/independent-tuning-benchmark`, para extração posterior. O [índice de migração](docs/retired-quality.md) aponta para os originais arquivados; [extratos de performance](docs/performance/README.md) preservam os achados de runtime. O conteúdo de ensino já foi migrado para o projeto independente `ailocal-teach`; este repositório não mantém uma pasta local de curso.

A limpeza não migra ou apaga `results.db`, Baseline, modelos ou aliases. O dashboard lê somente tabelas de tuning e preserva as tabelas históricas.

## Verificação

Instale `requirements-dev.txt` na venv e execute:

```powershell
.\venv\Scripts\python.exe scripts/run_validate.py
.\venv\Scripts\python.exe scripts/test_tuning_e2e.py
.\venv\Scripts\python.exe scripts/test_tuning_consumers_e2e.py
.\venv\Scripts\python.exe scripts/test_tuning_install_e2e.py
```

Os E2Es retêm SQLite e logs com modelos/hardware simulados; não estabelecem performance real de GPU. Testes reais usam o harness com runtime e Baseline configurados, preservando contexto, budgets e watchdogs.

## Licença

Código próprio: [PolyForm Noncommercial 1.0.0](LICENSE), AILOCAL. Uso comercial exige autorização separada. Materiais de terceiros mantêm suas licenças; veja [NOTICE.md](NOTICE.md).
