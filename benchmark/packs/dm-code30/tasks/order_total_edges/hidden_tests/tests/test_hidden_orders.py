import pytest

from orders import order_total


def test_tax_after_discount():
    assert order_total([{"price": 50, "quantity": 2}], 25, 0.10) == 82.5


def test_invalid_discount_range():
    with pytest.raises(ValueError):
        order_total([{"price": 10}], -1)
    with pytest.raises(ValueError):
        order_total([{"price": 10}], 101)
