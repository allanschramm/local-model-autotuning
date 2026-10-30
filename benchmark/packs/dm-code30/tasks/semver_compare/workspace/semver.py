def compare(left: str, right: str) -> int:
    """Compare two semantic version strings."""
    if left == right:
        return 0
    return -1 if left < right else 1
