import pytest

from datetime import date

from billing import period_end


def test_short_target_month_is_clamped():
    assert period_end(date(2024, 1, 31), 1) == date(2024, 2, 28)
    assert period_end(date(2024, 3, 31), 1) == date(2024, 4, 29)


def test_leap_year():
    assert period_end(date(2024, 1, 29), 1) == date(2024, 2, 28)
    assert period_end(date(2023, 1, 29), 1) == date(2023, 2, 27)


def test_multi_month_periods():
    assert period_end(date(2024, 1, 31), 12) == date(2025, 1, 30)
    assert period_end(date(2024, 8, 31), 6) == date(2025, 2, 27)


def test_invalid_month_count():
    with pytest.raises(ValueError):
        period_end(date(2024, 1, 1), 0)
