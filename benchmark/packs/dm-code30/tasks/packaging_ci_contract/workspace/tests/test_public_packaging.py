from packaging_contract import ci_check_commands, ci_python_versions


def test_ci_runs_pytest():
    assert "python -m pytest" in ci_check_commands()


def test_ci_has_supported_python_versions():
    assert "3.11" in ci_python_versions()
