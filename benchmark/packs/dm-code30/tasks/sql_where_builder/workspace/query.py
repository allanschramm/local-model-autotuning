def build_where(filters: dict):
    """Build a parameterised WHERE clause from a filter mapping."""
    if not filters:
        return '', []
    parts = []
    params = []
    for column, value in filters.items():
        parts.append(f"{column} = '{value}'")
    return ' AND '.join(parts), params
