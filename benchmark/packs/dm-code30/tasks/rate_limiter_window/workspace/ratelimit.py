class SlidingWindowLimiter:
    """Allow at most `limit` calls per key within a sliding window."""

    def __init__(self, limit: int = 3, window_seconds: float = 60.0):
        self.limit = limit
        self.window_seconds = window_seconds
        self.calls = []

    def allow(self, key: str, now: float) -> bool:
        if len(self.calls) >= self.limit:
            return False
        self.calls.append(now)
        return True
