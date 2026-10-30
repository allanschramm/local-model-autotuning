import pytest

from pagination import page


ITEMS = [{'id': n} for n in range(1, 6)]


def test_cursor_is_exclusive():
    window, cursor = page(ITEMS, 2, 2)
    assert [item['id'] for item in window] == [3, 4]
    assert cursor == 4


def test_last_page_reports_no_next_cursor():
    window, cursor = page(ITEMS, 4, 2)
    assert [item['id'] for item in window] == [5]
    assert cursor is None


def test_page_larger_than_remaining():
    window, cursor = page(ITEMS, None, 99)
    assert len(window) == 5
    assert cursor is None


def test_empty_items_and_bad_arguments():
    assert page([], None, 2) == ([], None)
    with pytest.raises(ValueError):
        page(ITEMS, None, 0)
    with pytest.raises(ValueError):
        page(ITEMS, 99, 2)
