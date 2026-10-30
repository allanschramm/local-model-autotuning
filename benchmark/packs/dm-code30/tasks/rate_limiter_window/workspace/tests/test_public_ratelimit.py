from ratelimit import SlidingWindowLimiter


def test_allows_up_to_limit():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    assert limiter.allow("a", 0.0) is True
    assert limiter.allow("a", 1.0) is True
    assert limiter.allow("a", 2.0) is False
