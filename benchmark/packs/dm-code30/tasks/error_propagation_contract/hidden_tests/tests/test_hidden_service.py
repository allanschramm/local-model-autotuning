import pytest

from errors import NotFound, Unavailable
from service import fetch
from storage import Storage, StorageError


def test_missing_key_becomes_not_found():
    with pytest.raises(NotFound) as info:
        fetch(Storage({'a': 1}), 'zzz')
    assert isinstance(info.value.__cause__, StorageError)


def test_other_storage_failure_becomes_unavailable():
    broken = Storage(failure=StorageError('disk on fire', code='io'))
    with pytest.raises(Unavailable) as info:
        fetch(broken, 'a')
    assert isinstance(info.value.__cause__, StorageError)


def test_programming_errors_are_not_swallowed():
    broken = Storage(failure=TypeError('bad call'))
    with pytest.raises(TypeError):
        fetch(broken, 'a')
