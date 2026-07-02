---
name: local-model-alias
description: "Cria um alias novo em `models/aliases/<name>/config.yaml` para o launcher global `model-up`. Use quando um baseline novo for descoberto (ex: novo flag combo, novo modelo, novo TPS medido) e Allan precisar subir o modelo manualmente depois."
---

# local-model-alias

Skill pra criar alias reutilizável de configuração do llama-server. Allan usa `model-up` no terminal (sem navegar até o projeto) pra subir o modelo com a config validada. Aliás = nome que aparece na API OpenAI-compatible (usado por Pi Agent, Hermes Agent, etc).

## Quando invocar

- Baseline novo descoberto (novo TPS medido, novo flag combo, novo modelo testado).
- Allan diz: "cria um alias pra X" / "salva essa config" / "preciso usar isso depois".
- Após teste concluído no autoloop/manual com resultado válido.

## Inputs necessários

Pergunte se Allan não passar:
- `name` — slug do alias (kebab-case, sem espaços). Ex: `qwen3.6-35b-mtp-active`.
- `model` — path absoluto do GGUF. Descobrir via `ls models/` ou symlinks em `models/`.
- `port` — porta OpenAI-compatible. Default `18080` se não especificado.
- `flags` — lista de flags llama-server validadas. Já testadas e que deram o TPS registrado.
- `metrics` (opcional) — `{tps, measured_at, measured_by, prompt_ref, notes}`.

## Procedimento

1. Validar inputs:
   - `model` deve existir (`os.path.exists`).
   - `name` único (não pode duplicar dir existente em `models/aliases/`).
   - `flags` deve ser lista de strings (cada uma pode conter `--flag value`, separado por espaço — `model-up` faz shlex.split).

2. Criar dir: `models/aliases/<name>/`

3. Escrever `config.yaml`:
```yaml
alias: <alias_name>          # nome que aparece na API
model: <absolute_path>       # GGUF
port: <port>                 # default 18080
host: 127.0.0.1
description: <one-liner>
flags:
  - --jinja              # required for Pi Agent
  - --ctx-size 131072
  - --n-gpu-layers 99
  - --n-cpu-moe <N>      # ~80% das camadas MoE no CPU. Dica: 20% na VRAM. Ex: 40 MoE layers → n-cpu-moe 32.
  # BeeLlama flags p/ draft/MTP model:
  # --spec-draft-n-cpu-moe <N>  (--n-cpu-moe-draft) — análogo pro draft model
  # --spec-draft-cpu-moe        (--cpu-moe-draft) — todo MoE do draft no CPU
  # ... flags validadas no teste
metrics:
  tps: <number>
  measured_at: <YYYY-MM-DD>
  measured_by: <M3 | Allan>
  prompt_ref: <path/to/session/doc>  # se aplicável
  notes: <observações>
status: ready
```

4. Atualizar `models/aliases/INDEX.md` — adicionar linha na tabela.

5. Reportar pra Allan:
```
Alias `<name>` criado. Uso:
  model-up              # sobe este (é o default se for único)
  model-up <name>       # sobe específico
  model-up list         # lista aliases
  model-up status       # health check
  model-up stop         # mata server
Plug em Pi/Hermes: model=<alias_name>, base_url=http://127.0.0.1:<port>/v1
```

## Não fazer

- Não criar alias sem teste empírico que valide o TPS.
- Não duplicar alias existente.
- Não hardcodar paths Windows (`C:\...`) — sempre caminhos relativos ou env vars.
- Não tocar `~/.local/bin/model-up` — é o launcher, não muda por alias.

## Exemplo

Input Allan: "cria um alias pro MTP ativo que vai dar 28 tok/s"

```yaml
# models/aliases/qwen3.6-35b-mtp-active/config.yaml
alias: qwen3.6-35b-mtp-active
model: models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf
port: 18080
host: 127.0.0.1
description: Qwen3.6-35B-A3B MoE com MTP ativa. Próximo teste.
flags:
  - --jinja
  - --ctx-size 131072
  - --n-gpu-layers 99
  - --n-cpu-moe 32         # 20% na VRAM: 8/40 → 32
  - --cache-type-k q4_0
  - --cache-type-v q4_0
  - --spec-type mtp
  - --spec-draft-n-max 2
  - --flash-attn on
  - --threads 8
  - --threads-batch 8
  - --batch-size 512
  - --ubatch-size 128
metrics:
  tps: null
  measured_at: null
  measured_by: TBD
  notes: PROJETADO. Não medido ainda.
status: untested
```

Note: `status: untested` enquanto não rodar. `status: ready` após validação empírica.

## Referências

- Launcher: `~/.local/bin/model-up` (Python script, lido em modo global)
- Convenção: `models/aliases/<name>/config.yaml`
- INDEX: `models/aliases/INDEX.md`
- Ground truth INDEX: `models/aliases/INDEX.md`
- Session log exemplo: `docs/sessions/2026-06-19-mtp-baseline.md`
