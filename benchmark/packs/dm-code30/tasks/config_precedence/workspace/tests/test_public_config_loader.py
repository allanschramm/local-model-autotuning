from config_loader import load_config


def test_file_overrides_defaults():
    assert load_config({"timeout": 10})["timeout"] == 10


def test_env_overrides_file_and_coerces_timeout():
    result = load_config({"timeout": 10}, env={"DM_TIMEOUT": "15"})
    assert result["timeout"] == 15
