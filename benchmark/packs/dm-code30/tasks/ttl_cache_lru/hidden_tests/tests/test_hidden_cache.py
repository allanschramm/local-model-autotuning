from cache import TTLCache


class Clock:
    def __init__(self):
        self.now = 0
    def __call__(self):
        return self.now


def test_lru_eviction_uses_recent_gets():
    clock = Clock()
    cache = TTLCache(max_size=2, ttl_seconds=100, clock=clock)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_expired_entries_are_removed_before_evicting_live_entries():
    clock = Clock()
    cache = TTLCache(max_size=2, ttl_seconds=5, clock=clock)
    cache.set("old", 1)
    clock.now = 6
    cache.set("new", 2)
    cache.set("third", 3)
    assert cache.get("old") is None
    assert cache.get("new") == 2
    assert cache.get("third") == 3
