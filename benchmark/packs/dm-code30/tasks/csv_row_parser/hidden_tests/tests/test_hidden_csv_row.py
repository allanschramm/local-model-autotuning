import pytest

from csv_row import parse_row


def test_quoted_field_with_comma():
    assert parse_row('a,"b,c",d') == ["a", "b,c", "d"]


def test_escaped_quote_inside_quotes():
    assert parse_row('"say ""hi""",x') == ['say "hi"', "x"]


def test_empty_fields_and_empty_line():
    assert parse_row("a,,b") == ["a", "", "b"]
    assert parse_row("") == [""]


def test_unterminated_quote_is_rejected():
    with pytest.raises(ValueError):
        parse_row('a,"unclosed')
