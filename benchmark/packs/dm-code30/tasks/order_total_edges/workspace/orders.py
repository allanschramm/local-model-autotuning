def apply_discount(subtotal: float, discount_percent: float) -> float:
    return subtotal - discount_percent


def order_total(items, discount_percent: float = 0, tax_rate: float = 0):
    subtotal = sum(item["price"] * item.get("quantity", 1) for item in items)
    discounted = apply_discount(subtotal, discount_percent)
    return round(discounted * (1 + tax_rate), 2)
