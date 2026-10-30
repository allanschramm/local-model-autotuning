from settings import resolve


def test_defaults_only():
    assert resolve(env={})['host'] == 'localhost'


def test_override_wins():
    assert resolve(env={}, overrides={'port': 9})['port'] == 9
