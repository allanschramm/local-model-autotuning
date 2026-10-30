def flatten_config(data: dict, prefix: str = '') -> dict:
    """Flatten a nested configuration dict into dotted keys."""
    flat = {}
    for key, value in data.items():
        path = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, path))
        else:
            flat[path] = value
    return flat
