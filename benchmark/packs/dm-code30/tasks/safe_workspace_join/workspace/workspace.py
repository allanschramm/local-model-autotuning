from pathlib import Path


def safe_join(root, requested):
    root_path = Path(root).resolve()
    candidate = Path(str(root_path) + "/" + requested).resolve()
    if not str(candidate).startswith(str(root_path)):
        raise ValueError("path escapes workspace")
    return candidate
