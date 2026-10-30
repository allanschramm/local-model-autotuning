import os

from defaults import DEFAULTS, ENV_PREFIX


def resolve(env=None, overrides=None):
    """Resolve settings from defaults, environment and explicit overrides."""
    env = os.environ if env is None else env
    settings = dict(DEFAULTS)
    for key, value in env.items():
        if key.startswith(ENV_PREFIX):
            settings[key[len(ENV_PREFIX):].lower()] = value
    settings.update(overrides or {})
    return settings
