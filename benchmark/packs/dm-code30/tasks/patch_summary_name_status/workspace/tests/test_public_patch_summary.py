from patch_summary import summarize_name_status


def test_basic_name_status_groups():
    result = summarize_name_status([
        "A\tdocs/tracing.md",
        "M\tdm_agent/core/agent.py",
        "D\told.py",
    ])
    assert result["added"] == ["docs/tracing.md"]
    assert result["modified"] == ["dm_agent/core/agent.py"]
    assert result["deleted"] == ["old.py"]
