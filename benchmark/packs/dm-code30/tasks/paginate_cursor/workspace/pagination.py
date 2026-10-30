def page(items, cursor=None, page_size=2):
    """Return one page of items plus the cursor for the next page."""
    ids = [item['id'] for item in items]
    start = 0
    if cursor is not None:
        start = ids.index(cursor)
    window = items[start:start + page_size]
    return window, window[-1]['id']
