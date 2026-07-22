# Missão: Workshop AILOCAL Essentials

## Por que existe
Ensinar qualquer pessoa a **rodar IA Local no próprio computador com alto desempenho**. Módulo 0 e Semana 1 formam a jornada publicada; a Semana 2 está em desenho curricular.

## Como é o sucesso
### Módulo 0 — Fundação Conceitual do Zero (para leigos)
- O que é um LLM de forma intuitiva (pesos no disco, contexto na memória, tokens)
- Nuvem vs Local (privacidade, custo zero por token, controle, offline)
- Hardware 101 (CPU vs GPU, RAM vs VRAM, o papel crítico da memória de vídeo)
- O milagre da Quantização (GGUF Q4 vs Q8 - analogia com compressão de imagem)
- Calculadora interativa de compatibilidade de VRAM

### Semana 1 — Desempenho bruto (escopo fechado)
- Mini-glossário Dia 1 (motor, modelo, quant, VRAM, KV, TPS, API/harness) — simplificado para leigos
- Distinguir **motor de inferência** vs **modelo** vs **quant**
- Baixar e escolher modelos GGUF focando no suporte da GPU (heurística de velocidade)
- Usar o modelo via **API local compatível com OpenAI** em qualquer aplicativo
- No Dia 2+: praticar o fluxo com **llama.cpp** e **ajustar parâmetros** para TPS (velocidade)
- No Dia 3: rodar **modelos MoE maiores que a memória de vídeo** (offload) no llama.cpp
- No Dia 4: executar um **caso de uso prático** com planejamento → tarefas → IA local

### Semana 2 — Qualidade (em construção, não publicada)
- Parâmetros de geração (temperatura, top-k, repetição)
- Skills / MCPs para desenvolvedores
- Ambientes isolados (sandbox), automações (hooks) e portões de qualidade (gates)
- Caso de uso final completo

## Regras e Restrições
- Material em **pt-BR** simples e didático para leigos
- Experiência 100% local: o aluno estuda clonando o repo e utilizando o tutor de IA (`/teach`) ou abrindo `teach/index.html` no navegador local (sem portal/servidor remoto)
- Semana 1: **apenas desempenho / velocidade (TPS)** — sem avaliação de inteligência ou treinamento de modelos
- Quizzes: gabarito com hash de segurança; opções simples em pt-BR sem menção ao LM Studio
- Progresso no guia: Concluir exige quiz e prática; a rota simulada conta, mas fica sinalizada como prática real pendente
- O foco do repositório é capacitar qualquer pessoa a rodar IA local do zero em sua própria máquina

## Fora de Escopo (Semana 1)
- Avaliar "qual modelo é mais inteligente"
- Treinamento ou fine-tuning de modelos
- Decodificação especulativa avançada
- Dependência da nuvem como solução principal
