from datetime import date

from billing import period_end


def test_simple_month():
    assert period_end(date(2024, 3, 15), 1) == date(2024, 4, 14)


def test_year_rollover():
    assert period_end(date(2024, 12, 10), 1) == date(2025, 1, 9)
