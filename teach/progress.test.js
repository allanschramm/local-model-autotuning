const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const assert = require("node:assert/strict");
const vm = require("node:vm");

function loadProgress(initial = {}) {
  const saved = new Map(Object.entries(initial));
  const context = {
    console,
    localStorage: {
      getItem(key) {
        return saved.has(key) ? saved.get(key) : null;
      },
      setItem(key, value) {
        saved.set(key, String(value));
      },
    },
  };
  context.globalThis = context;
  vm.createContext(context);
  vm.runInContext(
    fs.readFileSync(path.join(__dirname, "assets", "progress.js"), "utf8"),
    context,
  );
  return context.TeachProgress;
}

test("published journey ends after the six Module 0 and Week 1 lessons", () => {
  const progress = loadProgress();

  assert.deepEqual(
    Array.from(progress.LESSON_ORDER, (lesson) => lesson.id),
    ["s0d1", "s0d2", "s1d1", "s1d2", "s1d3", "s1d4"],
  );
  assert.equal(
    progress.getNextLesson(["s0d1", "s0d2", "s1d1", "s1d2", "s1d3", "s1d4", "s2d1"]),
    null,
  );
});

test("lesson readiness requires quizzes and either practice route", () => {
  const progress = loadProgress({
    teach_quiz_pass_v1: JSON.stringify(["s0d2-q1", "s0d2-q2"]),
  });

  assert.equal(progress.isLessonReady("s0d2"), false);
  progress.markPractice("s0d2", "simulated");
  assert.equal(progress.isLessonReady("s0d2"), true);
  assert.equal(progress.getPracticeMode("s0d2"), "simulated");

  progress.markPractice("s0d2", "real");
  assert.equal(progress.getPracticeMode("s0d2"), "real");
});

test("legacy Week 2 state remains stored but never enters the published journey", () => {
  const progress = loadProgress({
    teach_practice_pass_v1: JSON.stringify({ s2d1: "simulated" }),
  });

  assert.equal(progress.getPracticeMode("s2d1"), "simulated");
  assert.equal(progress.isPublishedLesson("s2d1"), false);
  assert.equal(progress.isLessonReady("s2d1"), false);
  assert.equal(progress.getNextLesson(["s2d1"]), progress.LESSON_ORDER[0]);
});
