"""scripts/check_leaderboard_docs.py — leaderboard doc integrity checks (issue #69).

- Per-table duplicate model names (case-insensitive). The Day and Night tables
  mirror the same membership on purpose (ADR 0017), so cross-section repeats
  are correct and must NOT be flagged.
- Day/Night membership must be the same set (same complete-vector leaderboard).
- Doc Day/Night tables must match `rank_results.py --mode pareto` output
  byte-for-byte (never hand-patched).
"""

from __future__ import annotations

from scripts import check_leaderboard_docs as cld

DOC = """\
# Model Leaderboard (Local Rig)

## Day table (near-ties broken by TPS)

| # | Model | ctx | TPS | agentic | coding |
|---|---|---|---|---|---|
| 1 | `Qwen3.8-4B-Q4_K_M.gguf` | 131k | 74.9 | 0.8667 | 0.6400 |
| 2 | `Nanbeige4.2-3B-Q4_K_M.gguf` | 32k | 53.8 | 0.4000 | 0.1800 |

## Night table (near-ties broken by ctx)

| # | Model | ctx | TPS | agentic | coding |
|---|---|---|---|---|---|
| 1 | `Qwen3.8-4B-Q4_K_M.gguf` | 131k | 74.9 | 0.8667 | 0.6400 |
| 2 | `Nanbeige4.2-3B-Q4_K_M.gguf` | 32k | 54.4 | 0.0000 | 0.3300 |
"""


def test_extract_day_and_night_table_model_names():
    day = cld.extract_table_models(DOC, "Day table")
    night = cld.extract_table_models(DOC, "Night table")
    assert day == ["Qwen3.8-4B-Q4_K_M.gguf", "Nanbeige4.2-3B-Q4_K_M.gguf"]
    assert night == ["Qwen3.8-4B-Q4_K_M.gguf", "Nanbeige4.2-3B-Q4_K_M.gguf"]


def test_extract_table_models_missing_section_is_empty():
    assert cld.extract_table_models("no tables here", "Day table") == []
    assert cld.extract_table_models(DOC, "Missing table") == []


def test_within_section_duplicate_detected_case_insensitive():
    doc = DOC.replace(
        "| 2 | `Nanbeige4.2-3B-Q4_K_M.gguf` | 32k | 53.8 | 0.4000 | 0.1800 |",
        "| 2 | `qwen3.8-4b-q4_k_m.gguf` | 131k | 74.9 | 0.8667 | 0.6400 |",
    )
    dupes = cld.find_duplicates(doc)
    assert dupes == {"Day table": ["qwen3.8-4b-q4_k_m.gguf"], "Night table": []}


def test_day_night_mirror_is_not_a_duplicate():
    # ADR 0017: Day and Night list the SAME membership — cross-section
    # repeats are by design and must stay clean.
    assert cld.find_duplicates(DOC) == {"Day table": [], "Night table": []}


def test_find_duplicates_reports_duplicated_names_sorted():
    doc = (
        "| # | Model | ctx | TPS | agentic | coding |\n"
        "|---|---|---|---|---|---|\n"
        "| 1 | `B.gguf` | 1 | 1 | 0.1 | 0.1 |\n"
        "| 2 | `a.gguf` | 1 | 1 | 0.1 | 0.1 |\n"
        "| 3 | `A.GGUF` | 1 | 1 | 0.1 | 0.1 |\n"
        "| 4 | `b.gguf` | 1 | 1 | 0.1 | 0.1 |\n"
    )
    dupes = cld.find_duplicates(doc)
    assert dupes == {"(untitled)": ["a.gguf", "b.gguf"]}


def test_day_night_membership_must_match():
    doc = DOC.replace(
        "| 2 | `Nanbeige4.2-3B-Q4_K_M.gguf` | 32k | 54.4 | 0.0000 | 0.3300 |",
        "",
    )
    assert cld.day_night_membership_equal(doc) is False
    assert cld.day_night_membership_equal(DOC) is True


def test_doc_tables_match_tool_output():
    tool = (
        "DAY  (pick=#1)\n"
        "|  # | Model                                           |  ctx |   TPS | agentic | coding |\n"
        "|----|-------------------------------------------------|------|-------|---------|--------|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf                          | 131k |  74.9 |  0.8667 | 0.6400 |\n"
        "\n"
        "NIGHT  (pick=#1)\n"
        "|  # | Model                                           |  ctx |   TPS | agentic | coding |\n"
        "|----|-------------------------------------------------|------|-------|---------|--------|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf                          | 131k |  74.9 |  0.8667 | 0.6400 |\n"
    )
    doc = (
        "# Leaderboard\n\n"
        "## Day table (near-ties broken by TPS)\n\n"
        "DAY  (pick=#1)\n"
        "|  # | Model                                           |  ctx |   TPS | agentic | coding |\n"
        "|----|-------------------------------------------------|------|-------|---------|--------|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf                          | 131k |  74.9 |  0.8667 | 0.6400 |\n"
        "\n"
        "## Night table (near-ties broken by ctx)\n\n"
        "NIGHT  (pick=#1)\n"
        "|  # | Model                                           |  ctx |   TPS | agentic | coding |\n"
        "|----|-------------------------------------------------|------|-------|---------|--------|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf                          | 131k |  74.9 |  0.8667 | 0.6400 |\n"
    )
    assert cld.doc_tables_match(doc, tool)


def test_doc_tables_mismatch_detected():
    tool = (
        "DAY  (pick=#1)\n"
        "|  # | Model | ctx | TPS | agentic | coding |\n"
        "|----|---|---|---|---|---|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n"
        "\n"
        "NIGHT  (pick=#1)\n"
        "|  # | Model | ctx | TPS | agentic | coding |\n"
        "|----|---|---|---|---|---|\n"
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n"
    )
    # Hand-patch: Night TPS row differs.
    doc = tool.replace("74.9 |  0.8667 | 0.6400 |", "44.2 |  0.5333 | 0.5900 |")
    assert not cld.doc_tables_match(doc, tool)


GOOD_DOC = (
    "# Leaderboard\n\n"
    "## Day table (near-ties broken by TPS)\n\n"
    "DAY  (pick=#1)\n"
    "|  # | Model | ctx | TPS | agentic | coding |\n"
    "|----|---|---|---|---|---|\n"
    "|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n"
    "\n"
    "## Night table (near-ties broken by ctx)\n\n"
    "NIGHT  (pick=#1)\n"
    "|  # | Model | ctx | TPS | agentic | coding |\n"
    "|----|---|---|---|---|---|\n"
    "|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n"
)


def _run(tmp_path, doc_text, capsys):
    doc = tmp_path / "board.md"
    doc.write_text(doc_text, encoding="utf-8")
    code = cld.main(["--doc", str(doc), "--tsv", str(tmp_path / "results.tsv")])
    out = capsys.readouterr()
    return code, out


def test_main_unseeded_store_skips_parity(tmp_path, capsys):
    code, out = _run(tmp_path, GOOD_DOC, capsys)
    assert code == 0
    assert "duplicate model names: none" in out.out
    assert "parity check: skipped" in out.out


def test_main_duplicate_fails(tmp_path, capsys):
    doc = GOOD_DOC.replace(
        "## Night table",
        "## Day table (near-ties broken by TPS)\n\nDAY  (pick=#1)\n|  # | Model | ctx | TPS | agentic | coding |\n|----|---|---|---|---|---|\n|  2 | qwen3.8-4b-q4_k_m.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n\n## Night table",
    )
    code, out = _run(tmp_path, doc, capsys)
    assert code == 1
    assert "DUPLICATE" in out.err


def test_main_membership_mismatch_fails(tmp_path, capsys):
    doc = GOOD_DOC.replace(
        "|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |\n\n## Night", "## Night"
    ).replace("NIGHT  (pick=#1)\n", "NIGHT  (pick=#1)\n", 1)
    # Night now lists a model Day lacks → membership mismatch.
    doc = GOOD_DOC.replace(
        "## Night table (near-ties broken by ctx)\n\nNIGHT  (pick=#1)\n|  # | Model | ctx | TPS | agentic | coding |\n|----|---|---|---|---|---|\n|  1 | Qwen3.8-4B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |",
        "## Night table (near-ties broken by ctx)\n\nNIGHT  (pick=#1)\n|  # | Model | ctx | TPS | agentic | coding |\n|----|---|---|---|---|---|\n|  1 | Other-9B-Q4_K_M.gguf | 131k |  74.9 |  0.8667 | 0.6400 |",
    )
    code, out = _run(tmp_path, doc, capsys)
    assert code == 1
    assert "MEMBERSHIP MISMATCH" in out.err
