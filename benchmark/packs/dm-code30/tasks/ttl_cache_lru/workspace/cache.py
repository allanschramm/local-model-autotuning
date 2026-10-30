import time


class TTLCache:
    def __init__(self, max_size: int = 128, ttl_seconds: float = 60, clock=None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.clock = clock or time.monotonic
        self._data = {}

    def set(self, key, value):
        self._data[key] = (value, self.clock())

    def get(self, key, default=None):
        return self._data.get(key, (default, 0))[0]
