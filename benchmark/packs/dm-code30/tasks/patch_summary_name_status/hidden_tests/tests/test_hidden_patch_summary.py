from patch_summary import summarize_name_status


def test_renames_and_blank_lines_are_supported():
    result = summarize_name_status([
        "",
        "R100\told_name.py\tnew_name.py",
        "M\tz_last.py",
        "M\ta_first.py",
    ])
    assert result["renamed"] == [
        {"from": "old_name.py", "to": "new_name.py"}
    ]
    assert result["modified"] == ["a_first.py", "z_last.py"]


def test_unknown_status_is_reported():
    result = summarize_name_status(["??\tuntracked.txt"])
    assert result["unknown"] == [{"status": "??", "path": "untracked.txt"}]
