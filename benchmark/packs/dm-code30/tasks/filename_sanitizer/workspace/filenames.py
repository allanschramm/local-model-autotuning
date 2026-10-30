import re

BAD = r'[/\\]'


def sanitize(name: str) -> str:
    """Return a safe single-segment file name."""
    cleaned = re.sub(BAD, '_', name).strip()
    return cleaned or 'untitled'
