from config_loader import load_config


def test_cli_overrides_env_and_file():
    result = load_config(
        {"timeout": 10, "debug": False},
        env={"DM_TIMEOUT": "20", "DM_DEBUG": "false"},
        cli_args={"timeout": "5", "debug": "true"},
    )
    assert result["timeout"] == 5
    assert result["debug"] is True


def test_defaults_are_not_mutated_between_runs():
    load_config({"retries": 9})
    assert load_config({})["retries"] == 2
