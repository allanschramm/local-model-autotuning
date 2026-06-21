# Session 2026-06-19 (parte 2) — Alias System + Launcher

## Goal
Allan pediu um launcher global pra subir o modelo sem navegar até o projeto. Requisito: `qwen-up` em qualquer terminal WSL, fecha terminal, server segue rodando. Pra plugar em Pi Agent / Hermes Agent (harness OpenAI-compatible).

## Decisão de design

Tentativa #1 (rejeitada): navegar até `models/aliases/<name>/` + rodar script local. Allan explicitamente rejeitou: "Não quero ter que navegar até o projeto nem nada do tipo".

Decisão final: launcher Python único em `~/.local/bin/qwen-up`, lê `models/aliases/<name>/config.yaml` (path absoluto hardcoded no script — não depende de CWD).

## Arquivos criados

| Path | Função |
|---|---|
| `~/.local/bin/qwen-up` | Launcher Python, global PATH, ~6.6 KB. Detach via `subprocess.Popen(start_new_session=True)`. |
| `models/aliases/qwen3.6-35b-mtp-baseline/config.yaml` | Alias do baseline atual (22.5 tok/s, sem flag MTP). |
| `models/aliases/INDEX.md` | Tabela de todos aliases. |
| `.agents/skills/local-model-alias/SKILL.md` | Skill Mavis pra criar novos aliases. |

## Subcomandos

```bash
qwen-up                # sobe default (primeiro do INDEX ou o DEFAULT_ALIAS hardcoded)
qwen-up <name>         # sobe alias específico
qwen-up list           # lista todos
qwen-up status         # PID + alias + port + health
qwen-up stop           # mata
```

## Validação

Testes feitos nesta sessão:

1. `qwen-up list` (login shell) → listou 1 alias corretamente.
2. `qwen-up` (start default) → PID 91623, OK ready at http://127.0.0.1:18080.
3. `qwen-up status` (nova WSL session) → server ainda vivo, prova de que sobreviveu ao exit do shell anterior.
4. `curl POST /v1/chat/completions` → resposta OK, model name `qwen3.6-35b-mtp-baseline` registrado.
5. `qwen-up stop` → killed clean, PID file removed.

Bug encontrado e corrigido durante validação:
- `--ctx-size 8192` no YAML era string única → argumento inválido pro llama-server. Fix: `shlex.split()` no launcher pra quebrar cada flag string em args separadas.
- `qwen-up status` lia `/proc/PID/cmdline` com split em espaço, mas cmdline usa null bytes. Fix: split em `\0`.

## Erro M3 corrigido

Tentativa #1 falhou: qwen-up retornou "command not found" via `wsl -d Ubuntu-24.04 -- bash -c "qwen-up list"`. Causa: bash não-login não sourcea `.bashrc`, então PATH não tinha `~/.local/bin`. Corrigido usando `bash -lc`. Em terminal interativo (caso real do Allan) o PATH é sourced normalmente — funciona sem flag.

## Comportamento do thinking mode

Qwen3.6 com defaults gera `<reasoning_content>` separado de `content`. 30 tokens de max_tokens foram todos consumidos pelo thinking, `content` ficou vazio. Allan precisa ou aumentar `max_tokens` (>200) ou setar `--chat-template-kwargs '{"enable_thinking": false}'` (vai em `flags:` do alias).

## Integração com Pi/Hermes

```yaml
# Pi Agent ~/.pi/agent/models.json ou Hermes config
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:18080/v1",
      "models": [{"id": "qwen3.6-35b-mtp-baseline"}]
    }
  }
}
```

## Pendente (próximo passo)

- Allan decide se quer adicionar `enable_thinking: false` por default nos flags do alias (UX melhor pra harness).
- Próximo teste LLM: criar alias `qwen3.6-35b-mtp-active` com `--spec-type mtp` ligado e medir.
