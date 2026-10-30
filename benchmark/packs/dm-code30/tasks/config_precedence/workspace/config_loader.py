import os

DEFAULTS = {"timeout": 30, "debug": False, "retries": 2}


def load_config(file_config=None, env=None, cli_args=None):
    env = env or os.environ
    file_config = file_config or {}
    cli_args = cli_args or {}
    config = DEFAULTS.copy()
    config.update(cli_args)
    config.update(file_config)
    if "DM_TIMEOUT" in env:
        config["timeout"] = env["DM_TIMEOUT"]
    if "DM_DEBUG" in env:
        config["debug"] = env["DM_DEBUG"]
    return config
