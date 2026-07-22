# 0004. Aprimoramento da Jornada do Aluno Leigo até a IA Local Completa

- **Data:** 2026-07-22
- **Status:** Substituído por 0008

## Contexto
A transição do aluno leigo do ambiente gráfico (LM Studio no Dia 1) para a linha de comando (`llama.cpp` no Dia 2) era o principal gargalo de atrito. Além disso, faltava uma ferramenta de verificação do ambiente e a Semana 2 do curso continha apenas esqueletos das lições de automação, ferramentas e qualidade.

## Decisões Tomadas
1. **Ferramenta de Recomendação de Hardware (`scripts/check_hardware.py`):** Criado script utilitário em Python que detecta RAM e VRAM dedicada via NVIDIA-SMI e recomenda o modelo GGUF ideal, tamanho de contexto (`-c`) e camadas na GPU (`-ngl`).
2. **Script de Auto-Validação e TPS (`scripts/verify_setup.py`):** Criada validação interativa que testa se o servidor local está ativo na porta 8080 e mede o desempenho real em tokens por segundo (TPS).
3. **Conclusão da Semana 2 (Lições s2d2 a s2d4):**
   - `s2d2-skills-mcps.html`: Conceito de MCP (Model Context Protocol) e Skills.
   - `s2d3-sandbox-hooks-gates.html`: Proteção de ambiente com Sandboxes, Pre-Tool Hooks e Quality Gates.
   - `s2d4-usecase-completo.html`: Projeto integrado final e checklist de 5 passos da autonomia local.
4. **Atualização do Portal e Currículo (`index.html` & `CURRICULUM.md`):** Atualizada a barra de progresso para 10 lições e adicionado o atalho para as ferramentas CLI de auxílio.

## Consequências
- A jornada do aluno leigo agora possui um feedback loop fechado com diagnósticos de terminal amigáveis.
- Esta conclusão foi substituída: somente Módulo 0 e Semana 1 estão publicados; a Semana 2 voltou ao estado de desenho curricular.
