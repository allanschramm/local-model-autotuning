from pathlib import Path

GLOSSARY = Path(__file__).resolve().parents[1] / "GLOSSARY.md"


def test_trial_status_language():
    text = GLOSSARY.read_text(encoding="utf-8")
    assert "Trial Status" in text
    assert "keep/discard" not in text
    assert "Pareto Tie-Breaker" not in text
