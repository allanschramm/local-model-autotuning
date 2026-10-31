---
name: inference-research
description: >
  Research inference-performance levers — llama.cpp flags and features,
  inference engines and forks, quantization and KV-cache techniques,
  speculative decoding, context optimization — on the operator rig, and land
  the findings in `docs/`. Scope is engine-side only: this
  skill does NOT search for or shortlist new models (that is discover-models /
  the validation skill). Use whenever the user asks to research inference, find
  flags/engines/techniques to raise TPS/ctx or cut VRAM, check whether "X is
  worth trying" on this rig, or verify a vendor claim against primary sources
  and documented measurements — including questions about dflash/dspark, MTP,
  speculative decoding, KV cache, quantization, ngram, or a specific engine.
  Pure desk research: read + web research only — never probe, run, or
  download. Workflow: read repo docs → learn what the repo already knows →
  fact-check and extend via web research → doc it in `docs/` → commit on
  operator go-ahead.
---

# inference-research

Investigate an inference-performance question against primary sources and
land the finding in `docs/` — the only tree this skill touches. Desk
research only: read the repo, learn what it already knows, fact-check and
extend via web research, then document the durable finding. Model choice is
out of scope (that is `docs/discovery/discover-models.md` and the `validation`
skill); this skill feeds only the engine side of a chosen model.

## Scope

**In:** engine flags/features on the pinned build; inference engines and
forks; quantization formats and KV-cache techniques; speculative decoding
(MTP, ngram, DFlash/DSpark, Eagle); context optimization (ctx axis, KV
compression, compaction); hardware-specific tuning for the 8 GB-class rig
(Baseline `VRAM_LIMIT_MB`); vendor-claim verification.

**Out (do not drift):** new model search / candidate shortlists (`whichllm`,
HF browsing for models → `discover-models.md`); harness-side IQ extraction
(parked until a harness fork); training or fine-tuning; running Trials
(bounded tuning); validation (`validation` skill); autonomous background loops.

**Touch only `docs/`.** A research pass edits `docs/discovery/`,
`docs/sessions/`, and this skill — nothing else: no code, no config, no
aliases, no measured-results files, no harness trees.

## Method

Workflow: **read repo docs → learn what the repo already knows → fact-check
and extend via web research → doc it in `docs/` → commit on operator
go-ahead.** Desk research only: no commands executed, no engine probes, no
downloads, no Trials. Each phase below.

1. **Read repo docs first.** Walk the repo's own knowledge before touching
   external sources: root `AGENTS.md` + `CONTEXT.md` (domain language and
   contracts), measured history (session logs, results data),
   `docs/sessions/` and `docs/discovery/` (prior findings, TBDs, dead ends),
   and the pinned-build notes (`docs/llamacpp-toolset.md`, `docs/models/`
   cards). A measured dead end stays dead unless the engine or setup
   changed. Do not re-research what the repo already knows.
2. **Learn what the repo already knows.** Collect every `**TBD:**` /
   open-question item in the repo docs. For each, apply the clarity gate:
   *"Is it clear how X works and how it affects inference?"* Answer **yes**
   from repo knowledge → state it and move to the next item. Answer **no** →
   that item is a research target. Understanding before writing: a finding
   is ready to document when you can explain the mechanism and why it
   matters on this rig, not just collect URLs.
3. **Web research: fact-check every claim — always.** Before anything lands
   in a doc, each claim gets a `web_search` pass against current sources.
   This is not conditional on repo docs being thin: it runs for every
   finding, including ones the repo already states, because engines ship
   fast — flags get renamed or deprecated, defaults change, releases
   supersede the repo's pinned build, and external measurements update.
   Search official docs, vendor releases, GitHub PR/issue API
   (`api.github.com`), HF model cards + API (tree, metadata, **discussions** —
   community measurements), vendor papers and blogs. Follow each claim to the
   source that owns it; date it; reference by URL, never copy-paste.
   External measurements are evidence with an explicit label, not proof for
   this rig. A claim that survives the search is stated with its sources; a
   claim the search contradicts is corrected, never copied from stale repo
   text.
4. **Doc it — `docs/` only.** Every finding lands in one of three shapes:
   - **Answer** — mechanism explained with its sources; phrased so a later
     pass can test it on this rig (switch on/off, same model, same tasks;
     e.g. a flag, KV format, spec type, ctx size).
   - **TBD** — flag support or claim not verifiable without running the
     pinned engine; mark **TBD:** and keep it in the doc's Open questions.
     A research pass only sharpens the prior.
   - **Dead end** — with measured evidence (repo session logs / results
     data, or an external measurement cited with URL + date and labeled
     **external / unverified on this rig**); do not re-run without new
     evidence or a changed setup.
   Prefer **patching existing** docs. Durable technique facts →
   `docs/discovery/` (stable contracts only). Resolve TBDs with evidence or
   sharpen their prior; never silently drop an open question. Single-day
   capture → `docs/sessions/YYYY-MM-DD-<topic>.md` following the sessions
   schema (Goal/Hardware/Findings/Decisions/Open questions;
   hardware-class language; no SKU, no absolute paths, no PII; cross-link
   related logs). Session logs are not edited after completion; add a
   follow-up file for new work.
5. **Commit — on operator go-ahead.** Prepare a docs-only commit
   (`docs/discovery/`, `docs/sessions/`, skill edits) and ask for the
   explicit command before committing. Never commit `models/aliases/`,
   machine Baseline, private paths, or harness code — a research pass
   touches `docs/` only.
6. **Constraints.** Pure desk research: no shell commands, no engine probes
   (`--help`, `setup-check.sh`), no venv/harness module runs, no downloads,
   no Trials, no `config.py` edits. Investigation = `read` / `grep` / `glob`
   / `web_search` on repo files and URLs only. Creating a new file (doc or
   skill) requires operator confirmation. Never autoloop.

## Frame the answer

Answer per item: verdict first (clear from repo knowledge, or researched),
then evidence (URLs with dates — each web-search fact-checked — or repo
session logs / measured results), then the falsifiable experiment for this
rig, then uncertainty and tradeoffs. If the item was already clear in repo
docs, say so, still run the fact-check pass on it, and move on — research
only what the repo cannot answer, but verify what the repo already knows.
State uncertainty at the claim and name the tradeoff; pick the boring option
when evidence is thin. Link measured numbers (results data, session logs)
instead of repeating them. Mark anything unverified as **unverified** — an
edit or an external measurement is not local proof.

## Anti-patterns (learned in this repo)

- Executing anything during a research pass — engine probes (`--help`,
  `setup-check.sh`), venv/harness module runs, downloads, Trials. This skill
  is desk research; probing and measuring are separate passes.
- Answering a repo-doc question the docs already answer — run the clarity
  gate first; research only what is unclear.
- Landing any claim without its web-search fact-check pass — the search runs
  for every finding, even ones the repo already states; engine releases,
  flag renames, and deprecations outpace static docs.
- Treating external measurements as local proof — label them **external /
  unverified on this rig** even when N-reproducible on other hardware.
- Re-running a measured dead end without new evidence or a changed setup
  (e.g. DFlash/DSpark on MoE + `n-cpu-moe`; separate draft models on hybrid
  SSM).
- Trusting vendor TPS/IQ numbers over the repo's own measurements.
- Answering with a model pick from this skill — model selection is out of
  scope.
- Editing code while investigating; a research pass changes docs only.
- Researching without the repo-docs read first — re-discovering ground the
  repo already measured.
- Committing more than the discoveries with a research pass
  (`models/aliases/`, machine Baseline, harness code) — the commit covers
  docs only, and only on operator go-ahead.

## Verification

- Every claim cites its source: a URL with access/extraction date, or a repo
  session log / measured results — and every claim passed the web-search
  fact-check pass before landing (no stale flags, renames, or deprecated
  defaults copied from older docs).
- External measurements are labeled **external / unverified on this rig**
  until a local measurement exists; vendor claims are labeled unverified.
- TBD items appear as **TBD:** and live in the doc's Open questions until
  resolved; evidence-backed resolution removes the marker, and
  measurement-only gaps stay open with the falsifiable experiment named.
- A research pass ends with docs updated and a docs-only commit prepared —
  the commit lands only on explicit operator command, and contains nothing
  else (aliases, Baseline, code stay out).
