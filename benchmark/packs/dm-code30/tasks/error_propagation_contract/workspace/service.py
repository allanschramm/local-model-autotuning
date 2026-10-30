from errors import ServiceError
from storage import StorageError


def fetch(storage, key):
    """Read a key and translate storage failures to the public contract."""
    try:
        return storage.read(key)
    except StorageError as exc:
        raise ServiceError(str(exc))
