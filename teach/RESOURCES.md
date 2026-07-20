# IA local — Recursos

## Conhecimento

### Semana 1 — Performance
- [LM Studio](https://lmstudio.ai/)
  App local com UI + servidor API. Use for: Dia 1 (ver o fluxo antes do CLI).
- [LM Studio — Local Server / OpenAI compatible](https://lmstudio.ai/docs)
  Docs oficiais do server. Use for: expor modelo via API para qualquer harness.
- [Hugging Face Hub — Download](https://huggingface.co/docs/huggingface_hub/guides/download)
  `hf download`. Use for: onde baixar GGUF.
- [llama.cpp (ggml-org)](https://github.com/ggml-org/llama.cpp)
  Motor CLI/server. Use for: Dia 2+.
- [docs/llamacpp-toolset.md](../docs/llamacpp-toolset.md)
  Flags neste checkout. Use for: tunar TPS no Dia 2.
- [docs/discovery/quantization-cascade.md](../docs/discovery/quantization-cascade.md)
  Escolha de quant vs VRAM. Use for: escolher modelo (performance).
- [docs/models/vitriol-technique.md](../docs/models/vitriol-technique.md)
  MoE / offload. Use for: Dia 3.

### Semana 2 — Qualidade (provisório)
- Cards/sampling nos model cards em `docs/models/` — Use for: temp/top-k quando S2D1 fechar.
- Documentação Cursor Skills / MCP (quando a aula fixar links) — TBD na curadoria.

## Sabedoria (comunidades)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — relatos de VRAM/TPS (filtrar marketing).
- [llama.cpp Discussions](https://github.com/ggml-org/llama.cpp/discussions) — flags e builds.

## Lacunas
- Link canónico LM Studio “Local Server” pode mudar de URL — validar na véspera do Dia 1.
- Motor “surpresa” do Dia 3: não documentar até o operador revelar.
- Semana 2 live: tópico em aberto.
