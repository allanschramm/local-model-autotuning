from serializers import serialize_user
from users import User


def test_serialize_user_uses_full_name():
    user = User(1, "Ada", "Lovelace", "ada@example.com")
    assert serialize_user(user) == {
        "id": 1,
        "display_name": "Ada Lovelace",
        "email": "ada@example.com",
    }
