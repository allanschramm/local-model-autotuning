# Visual Pack (Pi, No Rank Write)

**Goal:** YouTube-style proof that the daily model works — run one frozen
workspace prompt in Pi and watch it fix something on camera. Pi is the IQ
that gates daily use ([ADR 0014](../adr/0014-fingerprint-bus-product-split.md)).

## Same Fingerprint file as daily Pi

The visual pack pins the **same** `fingerprints/<stem>.json` the TPS climb
wrote — the engine Pi sees is the engine that won TPS. Serve it with the
launcher only:

```bash
.\venv\Scripts\python.exe scripts/model_up.py <alias>
```

Point Pi at the local OpenAI-compat bind (`http://127.0.0.1:18080` by
default), paste the frozen prompt, and watch. Engine change after TPS means
a new Fingerprint file, a new Pi run, and a new optional numeric row —
never reuse a visual run across engines.

## The frozen sample

- [`visual-pack-prompt.md`](./visual-pack-prompt.md) — prompt 01, notes-CLI
  fix in an empty workspace. Frozen: new tasks get a new numbered file.

## Camera rubric (optional, private)

A short checklist for the recording, kept out of the repo: the app runs,
state persists across runs, Pi narrates the root cause in one sentence.
Still not rank — a pass here elects nothing.

## Non-goals (hard)

- Visual results **never** write `results.tsv` / `results.db`, Pareto status,
  or `on_front`. No new TSV column, no rank writer — the stub is docs-only
  by design.
- No full-stack CRUD/API bench here — that is a later pack (ADR 0014 v1
  non-goals). No vendored third-party launcher trees.
- Sampler stays user/card choice, as in the TPS climb.

## Related

- [ADR 0014](../adr/0014-fingerprint-bus-product-split.md) — Fingerprint bus,
  visual-vs-numeric split.
- [`good-enough-tuning.md`](./good-enough-tuning.md) — TPS climb that writes
  the Fingerprint this pack pins.
- [`agentic-coding-benchmarks.md`](./agentic-coding-benchmarks.md) — the
  optional numeric pack (frozen pass/fail, comparable scores).
