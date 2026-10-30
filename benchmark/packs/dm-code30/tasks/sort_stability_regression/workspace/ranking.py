def rank_items(items):
    """Sort items by score, highest first."""
    return sorted(items, key=lambda item: -item['score'])
