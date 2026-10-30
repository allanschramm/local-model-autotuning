class JobRunner:
    """Run keyed jobs at most once."""

    def __init__(self):
        self.results = {}

    def submit(self, key: str, work):
        if self.results.get(key):
            return self.results[key]
        result = work()
        self.results[key] = result
        return result

    def has_run(self, key: str) -> bool:
        return bool(self.results.get(key))
