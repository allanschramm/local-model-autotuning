_MISSING = object()


def to_int(value, default=_MISSING):
    """Coerce a configuration value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        if default is _MISSING:
            raise ValueError(f'cannot coerce {value!r} to int')
        return default
