from query import build_where


def test_empty_filters():
    assert build_where({}) == ('', [])
