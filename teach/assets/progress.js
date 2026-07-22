/**
 * Shared lesson progress + quiz gate (localStorage).
 * Classic script (no ES modules) so file:// works offline.
 * "Concluir" on index.html unlocks only after every quiz in the lesson slot passes.
 */
(function (global) {
  const QUIZ_PASS_KEY = "teach_quiz_pass_v1";

  /** lessonSlot → required quiz ids (must match data-lesson on each .quiz) */
  const LESSON_QUIZZES = {
    s0d1: ["s0d1-q1", "s0d1-q2", "s0d1-q3"],
    s0d2: ["s0d2-q1", "s0d2-q2"],
    s1d1: ["s1d1-q1", "s1d1-q2", "s1d1-q3"],
    s1d2: ["s1d2-q1", "s1d2-q2"],
    s1d3: ["s1d3-q1", "s1d3-q2"],
    s1d4: ["s1d4-q1", "s1d4-q2"],
    s2d1: ["s2d1-q1", "s2d1-q2"],
    s2d2: ["s2d2-q1", "s2d2-q2"],
    s2d3: ["s2d3-q1", "s2d3-q2"],
    s2d4: ["s2d4-q1", "s2d4-q2"],
  };

  /** Curriculum order for "próxima lição" CTA (href relative to teach/index.html) */
  const LESSON_ORDER = [
    {
      id: "s0d1",
      href: "lessons/s0d1-fundamentos-ia-hardware.html",
      title: "Módulo 0 · Dia 1 — Fundamentos e hardware",
    },
    {
      id: "s0d2",
      href: "lessons/s0d2-troubleshooting-erros-comuns.html",
      title: "Módulo 0 · Dia 2 — Troubleshooting",
    },
    {
      id: "s1d1",
      href: "lessons/s1d1-lmstudio-avisos-motores-api.html",
      title: "Semana 1 · Dia 1 — Motores, GGUF e API",
    },
    {
      id: "s1d2",
      href: "lessons/s1d2-llamacpp-flags-tps.html",
      title: "Semana 1 · Dia 2 — llama.cpp e TPS",
    },
    {
      id: "s1d3",
      href: "lessons/s1d3-moe-maior-que-a-vram.html",
      title: "Semana 1 · Dia 3 — MoE / offload",
    },
    {
      id: "s1d4",
      href: "lessons/s1d4-usecase-fluxo-zero.html",
      title: "Semana 1 · Dia 4 — Caso de uso",
    },
    {
      id: "s2d1",
      href: "lessons/s2d1-parametros-qualidade.html",
      title: "Semana 2 · Dia 1 — Amostragem",
    },
    {
      id: "s2d2",
      href: "lessons/s2d2-skills-mcps.html",
      title: "Semana 2 · Dia 2 — Skills e MCPs",
    },
    {
      id: "s2d3",
      href: "lessons/s2d3-sandbox-hooks-gates.html",
      title: "Semana 2 · Dia 3 — Sandbox / hooks / gates",
    },
    {
      id: "s2d4",
      href: "lessons/s2d4-usecase-completo.html",
      title: "Semana 2 · Dia 4 — Projeto integrado",
    },
  ];

  function lessonSlotFromQuizId(quizId) {
    return String(quizId || "").replace(/-q\d+$/, "");
  }

  function getPassedQuizzes() {
    try {
      const raw = JSON.parse(localStorage.getItem(QUIZ_PASS_KEY) || "[]");
      return Array.isArray(raw) ? raw : [];
    } catch (e) {
      return [];
    }
  }

  function markQuizPassed(quizId) {
    const arr = getPassedQuizzes();
    if (!arr.includes(quizId)) {
      arr.push(quizId);
      try {
        localStorage.setItem(QUIZ_PASS_KEY, JSON.stringify(arr));
      } catch (e) {
        console.warn("Não foi possível salvar progresso do quiz:", e);
      }
    }
    return arr;
  }

  function isQuizPassed(quizId) {
    return getPassedQuizzes().includes(quizId);
  }

  function isLessonCleared(lessonSlot) {
    const need = LESSON_QUIZZES[lessonSlot] || [];
    if (need.length === 0) return false;
    const got = {};
    getPassedQuizzes().forEach(function (q) {
      got[q] = true;
    });
    return need.every(function (q) {
      return got[q];
    });
  }

  function quizPassCount(lessonSlot) {
    const need = LESSON_QUIZZES[lessonSlot] || [];
    const got = {};
    getPassedQuizzes().forEach(function (q) {
      got[q] = true;
    });
    const done = need.filter(function (q) {
      return got[q];
    }).length;
    return { done: done, total: need.length };
  }

  function quizProgressLabel(lessonSlot) {
    const counts = quizPassCount(lessonSlot);
    if (counts.total === 0) return "";
    if (counts.done >= counts.total) {
      return "Quizzes OK (" + counts.done + "/" + counts.total + ") — pode marcar Concluído no guia";
    }
    return "Quizzes: " + counts.done + "/" + counts.total + " — acerte todos para desbloquear Concluir";
  }

  /** First lesson in curriculum order not yet marked Concluído. */
  function getNextLesson(completedIds) {
    const done = {};
    (completedIds || []).forEach(function (id) {
      done[id] = true;
    });
    for (let i = 0; i < LESSON_ORDER.length; i++) {
      if (!done[LESSON_ORDER[i].id]) return LESSON_ORDER[i];
    }
    return null;
  }

  global.TeachProgress = {
    QUIZ_PASS_KEY: QUIZ_PASS_KEY,
    LESSON_QUIZZES: LESSON_QUIZZES,
    LESSON_ORDER: LESSON_ORDER,
    lessonSlotFromQuizId: lessonSlotFromQuizId,
    getPassedQuizzes: getPassedQuizzes,
    markQuizPassed: markQuizPassed,
    isQuizPassed: isQuizPassed,
    isLessonCleared: isLessonCleared,
    quizPassCount: quizPassCount,
    quizProgressLabel: quizProgressLabel,
    getNextLesson: getNextLesson,
  };
})(typeof window !== "undefined" ? window : globalThis);
