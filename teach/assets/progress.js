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

  global.TeachProgress = {
    QUIZ_PASS_KEY: QUIZ_PASS_KEY,
    LESSON_QUIZZES: LESSON_QUIZZES,
    lessonSlotFromQuizId: lessonSlotFromQuizId,
    getPassedQuizzes: getPassedQuizzes,
    markQuizPassed: markQuizPassed,
    isQuizPassed: isQuizPassed,
    isLessonCleared: isLessonCleared,
    quizPassCount: quizPassCount,
    quizProgressLabel: quizProgressLabel,
  };
})(typeof window !== "undefined" ? window : globalThis);
