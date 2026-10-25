"""Exclusive-max clamp (bug)."""


def clamp(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value >= hi:
        return hi - 1
    return value
