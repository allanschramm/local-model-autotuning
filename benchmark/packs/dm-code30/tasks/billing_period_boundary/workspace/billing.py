from datetime import date, timedelta


def period_end(start: date, months: int = 1) -> date:
    """Return the last day of a billing period."""
    year = start.year + (start.month - 1 + months) // 12
    month = (start.month - 1 + months) % 12 + 1
    return date(year, month, start.day) - timedelta(days=1)
