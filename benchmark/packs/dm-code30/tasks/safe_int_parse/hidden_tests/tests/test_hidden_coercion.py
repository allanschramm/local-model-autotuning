import pytest

from coercion import to_int


def test_bool_is_rejected():
    assert to_int(True, default=-1) == -1
    with pytest.raises(ValueError):
        to_int(False)


def test_float_and_decimal_string_are_rejected():
    assert to_int(1.5, default=-1) == -1
    assert to_int("1.0", default=-1) == -1


def test_whitespace_and_sign_are_accepted():
    assert to_int("  -42  ") == -42
    assert to_int("+7") == 7


def test_empty_and_none():
    assert to_int("", default=3) == 3
    assert to_int(None, default=3) == 3
    with pytest.raises(ValueError):
        to_int("   ")
