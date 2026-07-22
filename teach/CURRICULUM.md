# Currículo: Workshop AILOCAL Essentials

> 📌 **Guia Interativo do Aluno:** Abra o [teach/index.html](index.html) no navegador para ver o mapa visual de lições e salvar seu progresso.
>
> **Gate:** “Concluir” só libera depois de acertar todos os quizzes da lição.
>
> **Referências HTML:** [Glossário](reference/glossario.html) · [Flags llama.cpp](reference/llamacpp-flags.html)

## 🚀 Checklist de Preparação Inicial do Aluno (Multiplataforma)

Siga os 5 passos abaixo no terminal conforme o seu sistema operacional para preparar seu ambiente local:

### 🪟 Windows (PowerShell)
1. **Instalar Python (se necessário):** `winget install Python.Python.3.12`
2. **Criar Ambiente Virtual:** `python -m venv venv`
3. **Instalar Dependências:** `.\venv\Scripts\pip install -r requirements.txt`
4. **Diagnóstico de Hardware:** `.\venv\Scripts\python.exe scripts/check_hardware.py`
5. **Validação de TPS e Servidor (depois da Semana 1):** `.\venv\Scripts\python.exe scripts/verify_setup.py` — **pule agora** se ainda não subiu o server na porta 8080.

### 🍎 macOS / 🐧 Linux (Terminal)
1. **Instalar Python (se necessário):** `brew install python` (Mac) / `sudo apt install python3 python3-venv` (Linux)
2. **Criar Ambiente Virtual:** `python3 -m venv venv`
3. **Instalar Dependências:** `./venv/bin/pip install -r requirements.txt`
4. **Diagnóstico de Hardware:** `./venv/bin/python scripts/check_hardware.py`
5. **Validação de TPS e Servidor (depois da Semana 1):** `./venv/bin/python scripts/verify_setup.py` — **pule agora** se ainda não subiu o server na porta 8080.

---

## Módulo 0 — Fundação Conceitual do Zero (para leigos)

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Como funcionam as IAs, Nuvem vs Local, Hardware (CPU/GPU/VRAM), Tokenização | [s0d1](lessons/s0d1-fundamentos-ia-hardware.html) |
| **Dia 2** | Troubleshooting e Solução de Erros Comuns (OOM, Portas, CUDA, Prompt Templates) | [s0d2](lessons/s0d2-troubleshooting-erros-comuns.html) |

---

## Semana 1 — Desempenho bruto (fechada)

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Mini-glossário + motores de inferência, baixar/escolher modelos, API local | [s1d1](lessons/s1d1-lmstudio-avisos-motores-api.html) |
| **Dia 2** | Mesmo fluxo com **llama.cpp** + ajustar parâmetros de velocidade (TPS) | [s1d2](lessons/s1d2-llamacpp-flags-tps.html) |
| **Dia 3** | **MoE** maior que a GPU (divisão de carga / offload no llama.cpp) | [s1d3](lessons/s1d3-moe-maior-que-a-vram.html) |
| **Dia 4** | Caso de uso: planejar → dividir tarefas → executar com IA local | [s1d4](lessons/s1d4-usecase-fluxo-zero.html) |

---

## Semana 2 — Qualidade dos LLMs e Ferramentas (completa)

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Parâmetros de qualidade (temperatura, top-k, repetição, simulador) | [s2d1](lessons/s2d1-parametros-qualidade.html) |
| **Dia 2** | Skills pra dev, MCPs (Model Context Protocol) e ferramentas externas | [s2d2](lessons/s2d2-skills-mcps.html) |
| **Dia 3** | Ambientes isolados (sandbox), automações (hooks) e portões de segurança (gates) | [s2d3](lessons/s2d3-sandbox-hooks-gates.html) |
| **Dia 4** | Caso de uso final integrado de ponta a ponta (IA Local + Clientes + MCPs) | [s2d4](lessons/s2d4-usecase-completo.html) |
