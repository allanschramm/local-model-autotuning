from csv_row import parse_row


def test_plain_fields():
    assert parse_row("a,b,c") == ["a", "b", "c"]
