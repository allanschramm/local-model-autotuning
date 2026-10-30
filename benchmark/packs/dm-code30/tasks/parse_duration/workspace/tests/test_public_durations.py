from durations import parse_duration


def test_single_unit():
    assert parse_duration("45s") == 45
    assert parse_duration("2d") == 172800


def test_combined_units():
    assert parse_duration("1h30m") == 5400
