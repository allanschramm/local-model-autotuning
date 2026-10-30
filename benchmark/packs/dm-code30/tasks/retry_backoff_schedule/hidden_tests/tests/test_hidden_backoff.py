import pytest

from backoff import schedule


def test_cap_is_applied():
    assert schedule(6, base=1.0, cap=8.0) == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


def test_zero_attempts():
    assert schedule(0) == []


def test_base_equal_to_cap():
    assert schedule(3, base=5.0, cap=5.0) == [5.0, 5.0, 5.0]


def test_invalid_arguments():
    with pytest.raises(ValueError):
        schedule(-1)
    with pytest.raises(ValueError):
        schedule(3, base=0)
    with pytest.raises(ValueError):
        schedule(3, base=10.0, cap=5.0)
