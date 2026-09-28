# 2026-09-28 — Rebase sobre a main pós PR #85 + Trial E2E de validação

> Sessão de realinhamento. Rebase da branch `allanschramm/rust` sobre a main que
> recebeu o PR #85 (migração `--mmap`/`--no-mmap`/`--mlock` → `--load-mode` +
> drift guard), seguido da prova E2E que a validação pendente exigia.

## Goal

1. Rebasear os 6 commits do deepening sobre `origin/main` pós PR #85, resolvendo
   os conflitos sem perder nenhum dos dois lados.
2. Rodar a validação E2E de verdade (`benchmark_search.py --validation`) e
   confirmar que o Trial chega ao `results.db` — o artefato que o contrato de
   testing do repo exige como prova.

## Hardware

- **Class:** discrete_gpu host. Trial em GPU com `-ngl -1` (full offload).
- **VRAM_LIMIT_MB:** 7900 (Baseline).
- **OS family:** Windows.
- **llama.cpp engine/tag:** `b11149` (0.5.0-dev, commit d2e54583c), via
  `AUTORESEARCH_LLAMA_CPP_ROOT`. CUDA backend.
- **Model:** `empero-ai/Qwen3.8-4B-Q4_K_M.gguf` (2.78 GB), dense,
  `block_count=33`.

## Setup

```powershell
# Store de modelos real apontado para a worktree (junctions; models/ e gitignored).
cd models
foreach ($d in @("bartowski","empero-ai","ornith-ai","unsloth")) {
  New-Item -ItemType Junction -Path $d `
    -Target "D:\Dev\ailocal-nexus-system\ailocal-model-autotuning\models\$d"
}

# Baseline (config.py e gitignored) apontada para um modelo que cabe no budget.
# edit: 'MODEL': 'empero-ai/Qwen3.8-4B-Q4_K_M.gguf'

git rebase origin/main
.\venv\Scripts\python.exe benchmark_search.py --desc rebase-e2e-proof `
  --validation --no-agentic-quick
```

`--no-agentic-quick` porque o vendor tree `claw-eval/` vive no repo principal e
nao nesta worktree; sem ele o Trial morre em "No agentic quick tasks found", o
que nao tem relacao com o refactor.

## Commands (reproducible)

```powershell
git fetch origin
git branch backup/pre-rebase-20260928 HEAD     # antes de qualquer rebase
git rebase origin/main
ruff check autoresearch tests scripts ui autoloop.py benchmark_search.py
.\venv\Scripts\python.exe -m pytest tests\ -q
.\venv\Scripts\python.exe scripts\smoke_bindings.py
git push --force-with-lease origin allanschramm/rust
```

## Findings

### Rebase: 1 conflito, so em documento

Conflito unico em `autoresearch/AGENTS.md`, na bullet do `VRAM_LIMIT_MB`: a
main tinha acrescentado a nota do `--load-mode` nela, e o deepening tinha
acrescentado duas bullets novas logo abaixo. Resolucao: manter a bullet da main
(que carrega a nota do `--load-mode`) + as duas do deepening. Nenhum dos 4
arquivos de codigo conflituou — `llama_runner.py`, `evaluation.py` e
`autoloop.py` auto-mergearam, o que confirma que o deepening e a migracao de
flag sao ortogonais.

Verificado pos-rebase: os 5 emitters de `load_mode_flag` do PR #85 continuam
intactos (`llama_runner._build_cmd`, `evaluation.run_llama_bench_validation`,
`model_up.fingerprint_flags`, `serve-config.build_args`,
`autoloop._compile_alias_flags`). Os dois `--no-mmap` remanescentes em
`runners/run.py:78,322` sao o argparse do proprio harness, nao flags enviadas
ao llama.cpp — correto.

### Trial E2E: passou

```
[arch] dense block_count=33 n-cpu-moe=None (dense)
[vram-preflight] est=5320MB limit=7900MB ok=True
[host-preflight] est=5320MB budget=27790MB ok=True
[bench] tg 512: 71.4 t/s
[OK] Bench validation passed: tg 71.4 t/s >= 20.0
[VRAM] NVML initialized successfully. High-frequency 20ms sampling enabled.
[VRAM] CUDA-free guard active (floor=256MB; ...)

EVALUATION COMPLETE
Model:          empero-ai/Qwen3.8-4B-Q4_K_M.gguf
Combined TPS:     71.4 (Threshold: >= 20.0)
Bench tg:         71.4 t/s
Peak VRAM:        3.9 GB
```

Exit 0. O server subiu com `--load-mode mmap --cache-reuse 256`, o drift guard
(`assert_flags_supported`) passou sem rejeitar, e o Trial gravou no
`results.db`.

**Artefato** (`results.db`, 3 linhas — as duas anteriores guardar o porque de
cada falha, o que e a prova de que a decisao de veredito gravou o motivo certo):

| trial_id | status | outcome | tps | diagnostic |
|---|---|---|---|---|
| `75a1edea…` | incomplete | OK | 71.4 | (vazio — passou) |
| `852a1119…` | rejected | MODEL_REJECTED | 0.0 | `gguf_init_from_file: failed to open GGUF file '…/model.gguf'` |
| `63880c4b…` | rejected | MODEL_REJECTED | 0.0 | `error: invalid argument: --mmap` |

`status: incomplete` (e nao `ok`) porque foi uma validation sem coding/agentic
tiers; o throughput gate passou e e o que a validation prova.

### O drift guard teria pego o bug antes

O PR #85 nao so corrigiu a flag: adicionou `assert_flags_supported(cmd, binary)`,
que extrai as long flags do cmd, le o `--help` do binario real (cache por path) e
levanta `FlagDriftError` nomeando a flag e o build. Esta ligado em
`evaluation.py:594` (cli bench), `llama_runner.py:1379` (server build),
`model_up.py:385` e `serve-config.py:381,431`. A lição fica registrada: da próxima
vez que um Trial morrer com "invalid argument: <flag>", o diagnóstico já vem
apontando o build — não precisa de bisect manual.

## Errors

Nenhum no rebase. Dois obstaculos de ambiente durante a prova, ambos externos ao
refactor e ambos ja resolvidos:

1. `claw-eval/` ausente nesta worktree → `--no-agentic-quick`.
2. `models/` vazio nesta worktree → junctions para o store real + Baseline
   apontada para o 4B. Nada disso e versionado (`models/` e `config.py` sao
   ambos gitignored).

## Decisions

- **Rebase, nao merge.** 1 conflito so, em documento; a main virou ancestral e o
  historico ficou linear. Backup em `backup/pre-rebase-20260928` antes de comecar.
- **Force-push com `--force-with-lease`.** Reescreve apenas o historico desta
  branch, que nao tem outro consumidor alem do proprio worktree. A main nao foi
  tocada (permanece em `a5b244c`, o merge do PR #85).
- **`model_up.py` continua Python, e o trabalho de flag tambem.** A fronteira do
  deepening e: kernel CPU-bound em Rust, politica e decisao em Python puro,
  orquestacao/flags/lifecycle em Python, launcher em Python. O `--load-mode` era
  escopo da worktree Python e foi resolvido la via PR #85; aqui só foi herdado.
- **Prova E2E fechada** com artefato no `results.db`, que era o item que ficou
  explicitamente aberto (unverified) no commit `eb500e6`.
