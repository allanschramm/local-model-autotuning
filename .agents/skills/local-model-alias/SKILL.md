---
name: local-model-alias
description: "Cria um alias novo em `models/aliases/<name>/config.yaml` para o launcher global `qwen-up`. Use quando um baseline novo for descoberto (ex: novo flag combo, novo modelo, novo TPS medido) e Allan precisar subir o modelo manualmente depois."
---

# local-model-alias

Skill pra criar alias reutilizável de configuração do llama-server. Allan usa `qwen-up` no terminal (sem navegar até o projeto) pra subir o modelo com a config validada. Aliás = nome que aparece na API OpenAI-compatible (usado por Pi Agent, Hermes Agent, etc).

## Quando invocar

- Baseline novo descoberto (novo TPS medido, novo flag combo, novo modelo testado).
- Allan diz: "cria um alias pra X" / "salva essa config" / "preciso usar isso depois".
- Após teste concluído no autoloop/manual com resultado válido.

## Inputs necessários

Pergunte se Allan não passar:
- `name` — slug do alias (kebab-case, sem espaços). Ex: `qwen3.6-35b-mtp-active`.
- `model` — path absoluto do GGUF. Descobrir via `ls /home/shark/models/` ou symlinks em `models/`.
- `port` — porta OpenAI-compatible. Default `18080` se não especificado.
- `flags` — lista de flags llama-server validadas. Já testadas e que deram o TPS registrado.
- `metrics` (opcional) — `{tps, measured_at, measured_by, prompt_ref, notes}`.

## Procedimento

1. Validar inputs:
   - `model` deve existir (`os.path.exists`).
   - `name` único (não pode duplicar dir existente em `models/aliases/`).
   - `flags` deve ser lista de strings (cada uma pode conter `--flag value`, separado por espaço — `qwen-up` faz shlex.split).

2. Criar dir: `models/aliases/<name>/`

3. Escrever `config.yaml`:
```yaml
alias: <alias_name>          # nome que aparece na API
model: <absolute_path>       # GGUF
port: <port>                 # default 18080
host: 127.0.0.1
description: <one-liner>
flags:
  - --ctx-size 8192
  - --n-gpu-layers 99
  - --n-cpu-moe 32
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
  qwen-up              # sobe este (é o default se for único)
  qwen-up <name>       # sobe específico
  qwen-up status
  qwen-up stop
Plug em Pi/Hermes: model=<alias_name>, base_url=http://127.0.0.1:<port>/v1
```

## Não fazer

- Não criar alias sem teste empírico que valide o TPS.
- Não duplicar alias existente.
- Não hardcodar paths Windows (`C:\...`) — sempre WSL `/home/shark/...`.
- Não tocar `~/.local/bin/qwen-up` — é o launcher, não muda por alias.

## Exemplo

Input Allan: "cria um alias pro MTP ativo que vai dar 28 tok/s"

```yaml
# models/aliases/qwen3.6-35b-mtp-active/config.yaml
alias: qwen3.6-35b-mtp-active
model: /home/shark/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
port: 18080
host: 127.0.0.1
description: Qwen3.6-35B-A3B com flag MTP ativa. Próximo teste.
flags:
  - --ctx-size 8192
  - --n-gpu-layers 99
  - --n-cpu-moe 32
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
  tps: 28
  measured_at: 2026-06-19
  measured_by: TBD
  notes: PROJETADO. Não medido ainda.
status: untested
```

Note: `status: untested` enquanto não rodar. `status: ready` após validação empírica.

## Referências

- Launcher: `~/.local/bin/qwen-up` (Python script, lido em modo global)
- Convenção: `models/aliases/<name>/config.yaml`
- INDEX: `models/aliases/INDEX.md`
- Session log exemplo: `docs/sessions/2026-06-19-mtp-baseline.md`
