# 0006. Gate de quizzes para marcar lição concluída

Progresso no guia (`index.html`) não pode mais ser marcado só com o botão Concluir: cada slot exige acertar todos os quizzes daquela lição (`teach_quiz_pass_v1` via `assets/progress.js`). Resposta errada permite retry; lições já marcadas antes do gate permanecem concluídas. Também corrigido HTML/hash quebrado em `s2d4-q1`. Scripts de quiz/progress são **clássicos** (sem ES modules) para funcionar em `file://` — modules disparam CORS no Chrome.

Cross-link: [[MISSION.md]], [[AGENTS.md]], [[CURRICULUM.md]].
