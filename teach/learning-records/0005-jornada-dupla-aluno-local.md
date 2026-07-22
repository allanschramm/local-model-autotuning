# 0005. Jornada Dupla do Aluno 100% Local (Agente + Navegador Offline)

- **Data:** 2026-07-22
- **Status:** Aceito

## Contexto
O modelo de aprendizado de IA Local deste repositório não utiliza portal web nem servidores externos. Todo o fluxo ocorre dentro do checkout do repositório clonado localmente pelo aluno.

## Decisões Tomadas
1. **Entradas da Jornada Dupla:**
   - **Modo Agente (`/teach` skill):** O aluno utiliza o assistente de IA como seu tutor interativo no terminal/IDE. O agente guia as aulas, aplica retrieval practice e executa utilitários como `scripts/check_hardware.py` e `scripts/verify_setup.py`.
   - **Modo Navegador (`teach/index.html`):** O aluno abre o arquivo `index.html` localmente em seu navegador offline para navegar pelo mapa de lições, salvar progresso no `localStorage` e treinar a retenção.
2. **Fortalecimento de Memória de Longo Prazo (*Storage Strength*):**
   - Integrado o widget **Flash Quiz** em `teach/index.html` com treino de memória ativa sobre VRAM, quants, flags `llama.cpp` e MCPs.
3. **Agilidade no Terminal:**
   - Adicionados botões de cópia rápida para os comandos CLI de hardware e benchmark diretamente em `teach/index.html` e nas lições de troubleshooting (`s0d2`).

## Consequências
- A experiência do aluno é 100% independente da nuvem e totalmente offline.
- A retenção de longo prazo e a familiaridade com comandos de terminal aumentam sem exigir qualquer infraestrutura externa.

Cross-link: [[MISSION.md]], [[AGENTS.md]], [[CURRICULUM.md]].
