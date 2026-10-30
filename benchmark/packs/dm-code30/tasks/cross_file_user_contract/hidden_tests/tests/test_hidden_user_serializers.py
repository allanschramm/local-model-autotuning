from serializers import serialize_user
from users import User


def test_display_name_trims_missing_parts():
    user = User(2, " Grace ", " ", "grace@example.com")
    assert serialize_user(user)["display_name"] == "Grace"


def test_display_name_falls_back_to_email():
    user = User(3, " ", "", "missing@example.com")
    assert serialize_user(user)["display_name"] == "missing@example.com"
