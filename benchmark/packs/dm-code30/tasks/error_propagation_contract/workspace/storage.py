class StorageError(Exception):
    def __init__(self, message: str, code: str = 'io'):
        super().__init__(message)
        self.code = code


class Storage:
    """Low-level storage. code is "missing" when the key is absent."""

    def __init__(self, data=None, failure=None):
        self.data = data or {}
        self.failure = failure

    def read(self, key):
        if self.failure is not None:
            raise self.failure
        if key not in self.data:
            raise StorageError(f'no such key: {key}', code='missing')
        return self.data[key]
