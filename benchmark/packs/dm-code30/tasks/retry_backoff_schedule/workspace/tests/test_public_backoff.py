from backoff import schedule


def test_growth():
    assert schedule(3, base=1.0, cap=60.0) == [1.0, 2.0, 4.0]
