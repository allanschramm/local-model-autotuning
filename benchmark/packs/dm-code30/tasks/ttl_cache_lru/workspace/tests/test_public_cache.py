from cache import TTLCache


class Clock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now


def test_get_set_returns_value():
    clock = Clock()
    cache = TTLCache(ttl_seconds=10, clock=clock)
    cache.set("a", 1)
    assert cache.get("a") == 1


def test_expired_value_returns_default():
    clock = Clock()
    cache = TTLCache(ttl_seconds=5, clock=clock)
    cache.set("a", 1)
    clock.now = 6
    assert cache.get("a", "missing") == "missing"
