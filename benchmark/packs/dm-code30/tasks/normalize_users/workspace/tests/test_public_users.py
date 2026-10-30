from users import normalize_users


def test_trims_and_lowercases_email_and_name():
    rows = [{"email": "  Ada@Example.COM ", "name": " Ada "}]
    assert normalize_users(rows) == [
        {"email": "ada@example.com", "name": "Ada"}
    ]
