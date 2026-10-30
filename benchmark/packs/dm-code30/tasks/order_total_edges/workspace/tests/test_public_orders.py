import pytest

from orders import order_total


def test_total_without_discount():
    items = [{"price": 10, "quantity": 2}, {"price": 5}]
    assert order_total(items) == 25


def test_percentage_discount():
    assert order_total([{"price": 20}], discount_percent=10) == 18
