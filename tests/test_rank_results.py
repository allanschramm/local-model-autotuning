"""rank_results.py — Pareto / Day / Night views over results.tsv."""

from __future__ import annotations

import json

from scripts import rank_results as rr

CJ = json.dumps({"CTX_SIZE": 65536, "TEMP": 0.4, "TOP_P": 0.95, "TOP_K": 20})


def test_pareto_front_passes_through_every_complete_point():
    # ADR 0017: rank membership = every complete model; no cross-model exclusion.
    points = [
        rr.Point("A", ctx=65536, tps=30.0, agentic=0.6, coding=0.6),  # strong all-round
        rr.Point("B", ctx=65536, tps=100.0, agentic=0.2, coding=0.2),  # fast weak
        rr.Point("C", ctx=32768, tps=20.0, agentic=0.5, coding=0.5),  # beaten by A
    ]
    assert {p.model for p in rr.pareto_front(points)} == {"A", "B", "C"}


def test_day_table_lists_every_complete_model_quality_first():
    # ADR 0017: a fast weak model may not outrank a quality model, and a
    # slow quality model may never be excluded. IQ strictly rules outside
    # the ±0.05 near-tie band.
    points = [
        rr.Point("fast_weak", ctx=65536, tps=166.0, agentic=0.2, coding=0.2),
        rr.Point("slow_smart", ctx=32768, tps=35.0, agentic=0.62, coding=0.62),
    ]
    ranked = rr.day_table(points)
    assert [p.model for p in ranked] == ["slow_smart", "fast_weak"]


def test_day_table_near_tie_band_breaks_by_tps():
    # 0.02 apart → same band → higher TPS first (even though ctx is worse).
    points = [
        rr.Point("slower", ctx=131072, tps=64.0, agentic=0.47, coding=0.58),
        rr.Point("faster", ctx=32768, tps=166.0, agentic=0.45, coding=0.60),
        rr.Point("weaker", ctx=65536, tps=999.0, agentic=0.2, coding=0.2),
    ]
    ranked = rr.day_table(points)
    assert [p.model for p in ranked] == ["faster", "slower", "weaker"]


def test_day_table_outside_band_iq_rules():
    # 0.12 apart → IQ decides even against a huge TPS gap.
    points = [
        rr.Point("fast_weak", ctx=65536, tps=166.0, agentic=0.35, coding=0.35),
        rr.Point("smart", ctx=32768, tps=64.0, agentic=0.47, coding=0.58),
    ]
    ranked = rr.day_table(points)
    assert [p.model for p in ranked] == ["smart", "fast_weak"]


def test_near_tie_band_exact_half_point_is_tie_and_beyond_is_not():
    # |diff| == 0.05 → tie; |diff| == 0.06 → quality gap.
    tie = [
        rr.Point("a", ctx=65536, tps=40.0, agentic=0.50, coding=0.50),
        rr.Point("b", ctx=65536, tps=80.0, agentic=0.45, coding=0.45),
    ]
    assert [p.model for p in rr.day_table(tie)] == ["b", "a"]
    gap = [
        rr.Point("a", ctx=65536, tps=40.0, agentic=0.51, coding=0.51),
        rr.Point("b", ctx=65536, tps=80.0, agentic=0.45, coding=0.45),
    ]
    assert [p.model for p in rr.day_table(gap)] == ["a", "b"]


def test_near_tie_band_does_not_chain_through_middle_points():
    # Band membership is anchored at the band's highest member: 0.60 vs 0.56
    # is a tie, but 0.52 sits 0.08 below the anchor — a real quality gap
    # decides outright, never through chained middle points (user story 6).
    points = [
        rr.Point("top", ctx=65536, tps=20.0, agentic=0.60, coding=0.60),
        rr.Point("mid", ctx=65536, tps=999.0, agentic=0.56, coding=0.56),
        rr.Point("low", ctx=65536, tps=900.0, agentic=0.52, coding=0.52),
    ]
    ranked = rr.day_table(points)
    assert [p.model for p in ranked] == ["mid", "top", "low"]


def test_night_table_near_tie_band_breaks_by_ctx():
    # Near-tie quality → larger ctx first on Night, even against higher TPS.
    points = [
        rr.Point("small", ctx=32768, tps=166.0, agentic=0.45, coding=0.60),
        rr.Point("large", ctx=131072, tps=64.0, agentic=0.47, coding=0.58),
    ]
    ranked = rr.night_table(points)
    assert [p.model for p in ranked] == ["large", "small"]


def test_night_table_lists_every_complete_model_outside_band_by_iq():
    points = [
        rr.Point("weak", ctx=131072, tps=100.0, agentic=0.3, coding=0.3),
        rr.Point("smart", ctx=32768, tps=35.0, agentic=0.62, coding=0.62),
    ]
    ranked = rr.night_table(points)
    assert [p.model for p in ranked] == ["smart", "weak"]


def test_pick_day_night_return_table_heads():
    points = [
        rr.Point("smart", ctx=65536, tps=35.0, agentic=0.62, coding=0.62),
        rr.Point("weak", ctx=65536, tps=166.0, agentic=0.3, coding=0.3),
    ]
    assert rr.pick_day(points).model == "smart"
    assert rr.pick_night(points).model == "smart"


def test_cli_floor_flags_removed():
    # ADR 0017: floors are historical notes, not CLI knobs.
    import pytest

    for flag in ("--day-tps-floor", "--night-ctx-floor"):
        with pytest.raises(SystemExit):
            rr.parse_args([flag, "50"])


def test_build_vectors_merges_best_valid_scores_ignores_keep_and_pollution():
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "42.1",
            "tps": "42.1",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "39.5",  # pollution
            "bench_tg": "",
            "tps": "",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "50.0",
            "tps": "50.0",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "N.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "0.400000",
            "bench_tg": "20.0",
            "tps": "20.0",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert [p.model for p in complete] == ["M.gguf"]
    m = complete[0]
    assert m.agentic == 0.6
    assert m.coding == 0.57
    assert m.tps == 50.0  # max across Trials for the basename
    assert m.ctx == 65536
    assert [p.model for p in incomplete] == ["N.gguf"]


def test_build_vectors_reads_tps_ctx_from_description_when_columns_empty():
    rows = [
        {
            "model": "L.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "",
            "tps": "",
            "ctx": "",
            "config_json": CJ,
            "description": "L.gguf kv=f16 ctx=65536 TPS=166.4 bench_tg=166.4 | claw-full",
        },
        {
            "model": "L.gguf",
            "category": "10-task",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.350000",
            "bench_tg": "",
            "tps": "",
            "ctx": "",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert len(complete) == 1
    assert complete[0].tps == 166.4
    assert complete[0].ctx == 65536


def test_build_vectors_representative_run_carries_its_own_ctx_tps():
    # ADR 0017: the row is the single best-IQ run; ctx/TPS come from that
    # run — never a composite of best-per-axis across different runs.
    fp_a = json.dumps({"CTX_SIZE": 32768, "TEMP": 0.4})
    fp_b = json.dumps({"CTX_SIZE": 65536, "TEMP": 0.6})
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "100.0",
            "ctx": "32768",
            "config_json": fp_a,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "90.0",
            "ctx": "32768",
            "config_json": fp_a,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "agentic": "0.8000",
            "coding": "0.700000",
            "bench_tg": "30.0",
            "tps": "30.0",
            "ctx": "65536",
            "config_json": fp_b,
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert incomplete == []
    assert len(complete) == 1
    rep = complete[0]
    assert rep.model == "M.gguf"
    assert rep.agentic == 0.8
    assert rep.coding == 0.7
    assert rep.tps == 30.0  # best-IQ run's TPS, not the 100.0 of the slower-IQ run
    assert rep.ctx == 65536  # best-IQ run's ctx, not the merged max across runs


def test_build_vectors_split_config_runs_do_not_complete_a_display_vector():
    # ADR 0017: best-per-axis merging across runs is retired for display;
    # agentic @ one Baseline + coding @ another = no single measured run.
    fp_a = json.dumps({"CTX_SIZE": 65536, "TEMP": 0.4})
    fp_b = json.dumps({"CTX_SIZE": 32768, "TEMP": 0.6})
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "40.0",
            "ctx": "65536",
            "config_json": fp_a,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "55.0",
            "ctx": "32768",
            "config_json": fp_b,
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert complete == []
    assert [p.model for p in incomplete] == ["M.gguf"]


def test_build_vectors_legacy_rows_without_config_json_still_complete():
    # Basename merge: both axes measured → complete even without config_json.
    rows = [
        {
            "model": "L.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
        {
            "model": "L.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert incomplete == []
    assert [p.model for p in complete] == ["L.gguf"]
    assert complete[0].fp is None


def test_build_vectors_uses_axis_columns_when_populated():
    # Combined modern write path: agentic-full row with both columns populated,
    # no separate 10-task row.
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "status": "on_front",
            "val_score": "0.466700",
            "agentic": "0.4667",
            "coding": "0.490000",
            "ctx": "100000",
            "tps": "47.3",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert [p.model for p in complete] == ["M.gguf"]
    assert complete[0].agentic == 0.4667
    assert complete[0].coding == 0.49
    assert complete[0].ctx == 100000
    assert incomplete == []


def test_pick_returns_fingerprint_hint_from_best_claw_row():
    # ADR 0012: Point carries Fingerprint hint from best-claw row for Baseline load.
    from autoresearch.core.classify import fp_from_config_json

    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "42.1",
            "tps": "42.1",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "40.0",
            "tps": "40.0",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert len(complete) == 1
    expected_fp = fp_from_config_json(CJ)
    assert complete[0].fp == expected_fp
    assert complete[0].fp is not None
    day = rr.pick_day(complete)
    night = rr.pick_night(complete)
    assert day is not None and night is not None
    assert day.fp == expected_fp
    assert night.fp == expected_fp
    legacy = rr.Point("L.gguf", ctx=65536, tps=10.0, agentic=0.5, coding=0.5)
    assert legacy.fp is None


def test_day_and_night_tables_are_aligned_columns():
    # ADR 0017: both tables list every complete model; rows carry the
    # representative run's measured values.
    front = [
        rr.Point("POCKET.gguf", ctx=65536, tps=35.7, agentic=0.6667, coding=0.6150),
        rr.Point("MTP.gguf", ctx=32768, tps=63.7, agentic=0.4667, coding=0.5800),
        rr.Point("FAST.gguf", ctx=65536, tps=166.4, agentic=0.6000, coding=0.3500),
        rr.Point("KAT.gguf", ctx=65536, tps=30.2, agentic=0.6000, coding=0.6400),
    ]
    report = rr.format_report(front, [], mode="pareto")
    assert "DAY" in report
    assert "NIGHT" in report
    assert "| #" in report
    assert "Model" in report.splitlines()[1]
    day_section, night_section = report.split("NIGHT", 1)
    # Every complete model appears in BOTH sections.
    for model in ("POCKET.gguf", "MTP.gguf", "FAST.gguf", "KAT.gguf"):
        assert model in day_section
        assert model in night_section
    # Day order: KAT (0.64) and POCKET (0.615) are within the ±0.05 band →
    # TPS decides (POCKET 35.7 > KAT 30.2); then MTP (0.4667), FAST (0.35).
    day_lines = [line for line in day_section.splitlines() if line.startswith("|")]
    models_in_order = [line.split("|")[2].strip() for line in day_lines[2:]]
    assert models_in_order == ["POCKET.gguf", "KAT.gguf", "MTP.gguf", "FAST.gguf"]
    # POCKET's Day row keeps its own (sub-50) TPS: floors no longer filter.
    assert "35.7" in day_section


def test_live_top_of_table_order_locked_in_issue_70():
    # T6 lock-in: live-store top rows under the new math (measured 2026-09-10).
    # Qwen3.8-4B stays #1 on both tables; Tiel-Coder sits directly under its
    # quality peers (never hidden); LFM2.5-8B sorts below finished models by IQ
    # despite the fastest TPS on the board.
    front = [
        rr.Point("Qwen3.8-4B-Q4_K_M.gguf", ctx=131072, tps=74.9, agentic=0.8667, coding=0.64),
        rr.Point("Ornith-1.5-9B-Q4_K_M.gguf", ctx=65536, tps=43.2, agentic=0.80, coding=0.615),
        rr.Point("POCKET-35B-Q3_K_M.gguf", ctx=65536, tps=35.7, agentic=0.6667, coding=0.615),
        rr.Point(
            "Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
            ctx=65536,
            tps=31.3,
            agentic=0.60,
            coding=0.64,
        ),
        rr.Point("Ornith-1.5-35B-Q4_K_M.gguf", ctx=65536, tps=28.8, agentic=0.7333, coding=0.63),
        rr.Point(
            "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
            ctx=65536,
            tps=28.2,
            agentic=0.8667,
            coding=0.64,
        ),
        rr.Point("LFM2.5-8B-A1B-Q4_K_M.gguf", ctx=65536, tps=182.2, agentic=0.2667, coding=0.38),
        # Synthetic ctx-diverse peer inside the band: slow but long-context, so
        # Day (TPS lens) and Night (ctx lens) must disagree on it. Catches a
        # Day/Night tie_key swap that identical orders would hide.
        rr.Point("Wide-Slow.gguf", ctx=131072, tps=20.0, agentic=0.62, coding=0.62),
    ]
    expected_day = [
        "Qwen3.8-4B-Q4_K_M.gguf",
        "Ornith-1.5-9B-Q4_K_M.gguf",
        "POCKET-35B-Q3_K_M.gguf",
        "Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
        "Ornith-1.5-35B-Q4_K_M.gguf",
        "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "Wide-Slow.gguf",
        "LFM2.5-8B-A1B-Q4_K_M.gguf",
    ]
    expected_night = [
        "Qwen3.8-4B-Q4_K_M.gguf",
        "Wide-Slow.gguf",
        "Ornith-1.5-9B-Q4_K_M.gguf",
        "POCKET-35B-Q3_K_M.gguf",
        "Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf",
        "Ornith-1.5-35B-Q4_K_M.gguf",
        "Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "LFM2.5-8B-A1B-Q4_K_M.gguf",
    ]
    assert [p.model for p in rr.day_table(front)] == expected_day
    assert [p.model for p in rr.night_table(front)] == expected_night
    assert rr.pick_day(front).model == "Qwen3.8-4B-Q4_K_M.gguf"
    assert rr.pick_night(front).model == "Qwen3.8-4B-Q4_K_M.gguf"


def test_build_vectors_ignores_morris_screen_tps():
    # ADR 0016: screen probes (reps=1) must not set the basename TPS axis.
    rows = [
        {"model": "M.gguf", "outcome": "", "tps": "30.0", "agentic": "0.5", "coding": "0.5"},
        {
            "model": "M.gguf",
            "outcome": "OK",
            "evaluation_profile": "morris-screen",
            "category": "morris-screen",
            "tps": "999.0",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert complete[0].tps == 30.0
