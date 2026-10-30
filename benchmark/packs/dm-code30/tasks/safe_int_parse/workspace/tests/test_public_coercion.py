from coercion import to_int


def test_plain_values():
    assert to_int("42") == 42
    assert to_int(7) == 7


def test_default_on_bad_value():
    assert to_int("abc", default=0) == 0
