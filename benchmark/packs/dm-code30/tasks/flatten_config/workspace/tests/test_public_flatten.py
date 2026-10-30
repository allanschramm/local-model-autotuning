from flatten import flatten_config


def test_nested():
    assert flatten_config({"a": {"b": 1}}) == {"a.b": 1}


def test_flat_input():
    assert flatten_config({"a": 1}) == {"a": 1}
