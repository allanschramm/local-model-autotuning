/**
 * Shared quiz widget.
 * Correct answers are SHA-256(salt|lessonId|letter) — no plaintext key in HTML.
 * Classic script (no ES modules) so file:// works offline.
 * Depends on progress.js (window.TeachProgress). Wrong answers allow retry.
 */
(function () {
  const SALT = "teach-lmai-v1";
  const TP = window.TeachProgress;
  if (!TP) {
    console.error("quiz.js requires progress.js loaded first");
    return;
  }

  /* Minimal SHA-256 (public domain style) for file:// fallback */
  function sha256Fallback(ascii) {
    function rotr(n, x) {
      return (x >>> n) | (x << (32 - n));
    }
    function utf8(str) {
      return unescape(encodeURIComponent(str));
    }
    const K = [
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    const msg = utf8(ascii);
    const l = msg.length;
    const words = [];
    for (let i = 0; i < l; i++) words[i >> 2] |= msg.charCodeAt(i) << (24 - (i % 4) * 8);
    words[l >> 2] |= 0x80 << (24 - (l % 4) * 8);
    words[(((l + 8) >> 6) << 4) + 15] = l * 8;
    let H0 = 0x6a09e667,
      H1 = 0xbb67ae85,
      H2 = 0x3c6ef372,
      H3 = 0xa54ff53a,
      H4 = 0x510e527f,
      H5 = 0x9b05688c,
      H6 = 0x1f83d9ab,
      H7 = 0x5be0cd19;
    const w = new Array(64);
    for (let i = 0; i < words.length; i += 16) {
      let a = H0,
        b = H1,
        c = H2,
        d = H3,
        e = H4,
        f = H5,
        g = H6,
        h = H7;
      for (let j = 0; j < 64; j++) {
        if (j < 16) w[j] = words[i + j] | 0;
        else {
          const g0 = w[j - 15];
          const g1 = w[j - 2];
          const s0 = rotr(7, g0) ^ rotr(18, g0) ^ (g0 >>> 3);
          const s1 = rotr(17, g1) ^ rotr(19, g1) ^ (g1 >>> 10);
          w[j] = (w[j - 16] + s0 + w[j - 7] + s1) | 0;
        }
        const S1 = rotr(6, e) ^ rotr(11, e) ^ rotr(25, e);
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + K[j] + w[j]) | 0;
        const S0 = rotr(2, a) ^ rotr(13, a) ^ rotr(22, a);
        const maj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + maj) | 0;
        h = g;
        g = f;
        f = e;
        e = (d + t1) | 0;
        d = c;
        c = b;
        b = a;
        a = (t1 + t2) | 0;
      }
      H0 = (H0 + a) | 0;
      H1 = (H1 + b) | 0;
      H2 = (H2 + c) | 0;
      H3 = (H3 + d) | 0;
      H4 = (H4 + e) | 0;
      H5 = (H5 + f) | 0;
      H6 = (H6 + g) | 0;
      H7 = (H7 + h) | 0;
    }
    function hex(n) {
      return ("00000000" + (n >>> 0).toString(16)).slice(-8);
    }
    return hex(H0) + hex(H1) + hex(H2) + hex(H3) + hex(H4) + hex(H5) + hex(H6) + hex(H7);
  }

  async function sha256Hex(text) {
    if (globalThis.crypto && crypto.subtle && globalThis.isSecureContext) {
      const data = new TextEncoder().encode(text);
      const buf = await crypto.subtle.digest("SHA-256", data);
      return Array.from(new Uint8Array(buf))
        .map(function (b) {
          return b.toString(16).padStart(2, "0");
        })
        .join("");
    }
    return sha256Fallback(text);
  }

  async function hashChoice(lessonId, choice) {
    return sha256Hex(SALT + "|" + lessonId + "|" + choice);
  }

  function updateGateBanner(slot) {
    const el = document.getElementById("quiz-gate-banner");
    if (!el || !slot) return;
    el.textContent = TP.quizProgressLabel(slot);
    el.classList.toggle("cleared", TP.isLessonCleared(slot));
  }

  function ensureGateBanner(slot) {
    let el = document.getElementById("quiz-gate-banner");
    if (el) {
      updateGateBanner(slot);
      return el;
    }
    el = document.createElement("aside");
    el.id = "quiz-gate-banner";
    el.className = "quiz-gate-banner";
    el.setAttribute("role", "status");
    const wrap = document.querySelector(".wrap") || document.body;
    const firstQuiz = wrap.querySelector("[data-quiz]");
    if (firstQuiz) wrap.insertBefore(el, firstQuiz);
    else wrap.appendChild(el);

    const link = document.createElement("p");
    link.className = "quiz-gate-link";
    link.innerHTML =
      '<a href="../index.html">Voltar ao guia</a> para marcar Concluído (só libera com quizzes OK).';
    el.after(link);
    updateGateBanner(slot);
    return el;
  }

  function lockQuizCorrect(root, feedback) {
    const buttons = Array.from(root.querySelectorAll("[data-choice]"));
    buttons.forEach(function (b) {
      b.disabled = true;
      b.setAttribute("aria-pressed", "false");
      b.removeAttribute("data-correct");
    });
    root.classList.add("quiz-passed");
    feedback.textContent = root.dataset.ok || "Correto — já registrado.";
    feedback.className = "feedback ok";
  }

  function mountQuiz(root) {
    const buttons = Array.from(root.querySelectorAll("[data-choice]"));
    const feedback = root.querySelector("[data-feedback]");
    const lessonId = root.dataset.lesson;
    const expected = (root.dataset.answerHash || "").toLowerCase();
    const slot = TP.lessonSlotFromQuizId(lessonId);

    ensureGateBanner(slot);

    if (TP.isQuizPassed(lessonId)) {
      lockQuizCorrect(root, feedback);
      updateGateBanner(slot);
      return;
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", async function () {
        buttons.forEach(function (b) {
          b.setAttribute("aria-pressed", "false");
          b.disabled = true;
        });
        btn.setAttribute("aria-pressed", "true");

        const digest = await hashChoice(lessonId, btn.dataset.choice);
        const ok = digest === expected;
        btn.dataset.correct = ok ? "true" : "false";

        if (!ok) {
          feedback.textContent = root.dataset.bad || "Não é essa — tente de novo.";
          feedback.className = "feedback bad";
          buttons.forEach(function (b) {
            b.disabled = false;
            b.removeAttribute("data-correct");
            b.setAttribute("aria-pressed", "false");
          });
          return;
        }

        feedback.textContent = root.dataset.ok || "Correto.";
        feedback.className = "feedback ok";
        TP.markQuizPassed(lessonId);
        updateGateBanner(slot);
      });
    });
  }

  document.querySelectorAll("[data-quiz]").forEach(mountQuiz);
})();
