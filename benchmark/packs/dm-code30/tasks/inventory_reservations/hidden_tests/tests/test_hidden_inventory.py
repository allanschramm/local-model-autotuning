import pytest

from inventory import Inventory


def test_add_accumulates_and_exact_match_reserves():
    inv = Inventory()
    inv.add("pen", 2)
    inv.add("pen", 3)
    assert inv.reserve("pen", 5) is True
    assert inv.stock["pen"] == 0


def test_invalid_and_insufficient_quantities():
    inv = Inventory()
    inv.add("bag", 1)
    assert inv.reserve("missing", 1) is False
    assert inv.reserve("bag", 2) is False
    with pytest.raises(ValueError):
        inv.add("bag", -1)
    with pytest.raises(ValueError):
        inv.reserve("bag", -1)
