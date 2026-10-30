class Inventory:
    def __init__(self):
        self.stock = {}

    def add(self, sku: str, quantity: int) -> None:
        self.stock[sku] = quantity

    def reserve(self, sku: str, quantity: int) -> bool:
        if self.stock.get(sku, 0) <= quantity:
            return False
        self.stock[sku] -= quantity
        return True
