import pytest

from flatten import flatten_config


def test_empty_nested_dict_is_a_leaf():
    assert flatten_config({"a": {}, "b": 1}) == {"a": {}, "b": 1}


def test_deep_nesting():
    data = {"a": {"b": {"c": {"d": 4}}}}
    assert flatten_config(data) == {"a.b.c.d": 4}


def test_lists_are_leaves():
    data = {"a": [{"b": 1}]}
    assert flatten_config(data) == {"a": [{"b": 1}]}


def test_empty_input_and_invalid_keys():
    assert flatten_config({}) == {}
    with pytest.raises(ValueError):
        flatten_config({"a.b": 1})
    with pytest.raises(ValueError):
        flatten_config({1: 2})
