def slugify(value: str) -> str:
    """Return a URL slug for the input text."""
    return value.strip().lower().replace(" ", "-")
