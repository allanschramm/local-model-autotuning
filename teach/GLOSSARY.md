# Glossário — Workshop AILOCAL Essentials

Canonical terms for this teaching workspace. Student HTML: [reference/glossario.html](reference/glossario.html).

## Terms

**Motor**:
Programa que carrega pesos e gera tokens (ex.: llama-server). Não é o arquivo do modelo.
_Avoid_: “a IA”, “o ChatGPT local” (ambíguo)

**Modelo**:
Pesos no disco, tipicamente um `.gguf`.
_Avoid_: “engine”, “checkpoint treinado agora”

**Quant**:
Compressão dos pesos (Q4/Q5/Q8…). Menos bits → menos VRAM; TPS costuma subir.
_Avoid_: “qualidade do modelo” (Semana 1 = velocidade)

**VRAM**:
Memória da GPU onde pesos + KV competem. Dense deve caber na VRAM física neste curso.
_Avoid_: “precisa caber tudo sempre” (MoE/offload = Dia 3)

**KV / contexto**:
Memória do histórico; controlada em grande parte por `-c`.
_Avoid_: “memória do chat” sem distinguir de VRAM de pesos

**TPS**:
Tokens por segundo — métrica de velocidade da Semana 1.
_Avoid_: “inteligência”, “Elo”, “benchmark de qualidade”

**API local**:
HTTP na máquina (ex.: `127.0.0.1:18080`), em geral formato OpenAI-compatível.

**Harness**:
Cliente da API (script, app, agente, IDE).

**Offload**:
Parte do modelo fora da VRAM. Dense: evitar. MoE: ferramenta do Dia 3.

**OOM**:
Memória esgotada. Em denso, cortar contexto/KV, remover draft ou escolher GGUF menor — nunca “spill and hope”.
