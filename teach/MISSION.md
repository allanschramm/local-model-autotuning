# Mission: Curso IA local — performance e depois qualidade

## Why
Formar alunos a **rodar LLMs no próprio PC com TPS alto** (Semana 1) e, em seguida, a **cuidar da qualidade do uso** (Semana 2: sampling, skills, gates). O repositório é a sala de aula a partir do Dia 2 da Semana 1; o Dia 1 ancora o entendimento visual no **LM Studio**.

## Success looks like
### Semana 1 — Performance bruta (escopo fechado)
- Mini-glossário Dia 1 (motor, modelo, quant, VRAM, KV, TPS, API/harness) — fora do quiz; offload/MoE só apontado ao Dia 3
- Distinguir **motor de inferência** vs **modelo** vs **quant**
- Baixar e escolher GGUF com cabe na GPU (heurística de performance)
- Usar o modelo via **API OpenAI-compatible** em qualquer harness — primeiro vendo o fluxo no **LM Studio**
- No Dia 2+: repetir o fluxo com **llama.cpp** e **tunar flags** para TPS
- No Dia 3: correr **MoE maior que a VRAM** (offload) no llama.cpp (+ teaser de outro motor)
- No Dia 4: executar um **use-case** com planeamento → tasks → LLM local

### Semana 2 — Qualidade (pode mudar)
- Sampling (temp, top-k, repetition penalty…)
- Skills / MCPs para dev
- Sandbox, hooks, quality gates
- Use-case final com o kit completo

## Constraints
- Material em **pt-BR** (flags e nomes de ferramenta verbatim)
- Semana 1: **só performance / TPS** — não qualidade do modelo, não fine-tune, não treino
- Dia 1: só LM Studio (lente visual) — **este repo começa no Dia 2** com llama.cpp
- Quizzes com gabarito hasheado; opções em pt-BR
- Material das lições vive em `teach/` (entrega); o clone/uso operacional do repo = Dia 2+

## Out of scope (Semana 1)
- Avaliar “qual modelo é mais inteligente”
- Fine-tuning / treinar pesos
- Deep dive speculative decoding (MTP só se entrares depois como bónus de TPS)
- Cloud como caminho principal
