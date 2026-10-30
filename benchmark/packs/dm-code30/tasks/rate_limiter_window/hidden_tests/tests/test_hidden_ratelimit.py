import pytest

from ratelimit import SlidingWindowLimiter


def test_keys_are_independent():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.allow("a", 0.0) is True
    assert limiter.allow("b", 0.0) is True
    assert limiter.allow("a", 1.0) is False


def test_window_slides_and_is_half_open():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.allow("a", 0.0) is True
    assert limiter.allow("a", 9.9) is False
    assert limiter.allow("a", 10.0) is True


def test_rejected_calls_are_not_recorded():
    limiter = SlidingWindowLimiter(limit=1, window_seconds=10)
    assert limiter.allow("a", 0.0) is True
    assert limiter.allow("a", 5.0) is False
    assert limiter.allow("a", 10.0) is True


def test_expired_entries_are_discarded():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    for step in range(50):
        limiter.allow("a", float(step) * 10)
    stored = sum(
        len(value) if hasattr(value, '__len__') else 1
        for value in vars(limiter).values()
        if isinstance(value, (list, dict))
    )
    assert stored <= 10


def test_invalid_configuration():
    with pytest.raises(ValueError):
        SlidingWindowLimiter(limit=0)
    with pytest.raises(ValueError):
        SlidingWindowLimiter(limit=1, window_seconds=0)
