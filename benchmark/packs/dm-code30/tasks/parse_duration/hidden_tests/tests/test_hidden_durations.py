import pytest

from durations import parse_duration


def test_spaces_and_three_parts():
    assert parse_duration("1h 30m 15s") == 5415
    assert parse_duration("  2m10s  ") == 130


def test_bare_integer_is_seconds():
    assert parse_duration("90") == 90


def test_units_may_appear_in_any_order():
    assert parse_duration("30m1h") == 5400


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_duration("")
    with pytest.raises(ValueError):
        parse_duration("10x")
    with pytest.raises(ValueError):
        parse_duration("1h2h")
    with pytest.raises(ValueError):
        parse_duration("-5s")
