from inventory import Inventory


def test_reserve_decrements_stock():
    inv = Inventory()
    inv.add("book", 3)
    assert inv.reserve("book", 2) is True
    assert inv.stock["book"] == 1
