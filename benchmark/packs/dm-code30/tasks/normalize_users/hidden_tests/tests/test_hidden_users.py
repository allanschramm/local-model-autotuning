from users import normalize_users


def test_filters_deduplicates_and_sorts():
    rows = [
        {"email": "b@example.com", "name": " Bea ", "active": True},
        {"email": "invalid", "name": "Skip", "active": True},
        {"email": "a@example.com", "name": "Ann", "active": False},
        {"email": "A@Example.com", "name": "Ada", "active": True},
        {"email": "b@example.com", "name": "Duplicate", "active": True},
    ]
    assert normalize_users(rows) == [
        {"email": "a@example.com", "name": "Ada"},
        {"email": "b@example.com", "name": "Bea"},
    ]
