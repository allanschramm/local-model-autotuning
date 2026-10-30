import pytest

from query import build_where


def test_values_are_parameterised_not_inlined():
    clause, params = build_where({'name': "O'Brien"})
    assert clause == 'name = ?'
    assert params == ["O'Brien"]


def test_keys_are_sorted_and_params_follow_placeholder_order():
    clause, params = build_where({'b': 2, 'a': 1})
    assert clause == 'a = ? AND b = ?'
    assert params == [1, 2]


def test_none_becomes_is_null():
    clause, params = build_where({'deleted_at': None})
    assert clause == 'deleted_at IS NULL'
    assert params == []


def test_list_becomes_in_clause():
    clause, params = build_where({'id': [1, 2, 3]})
    assert clause == 'id IN (?, ?, ?)'
    assert params == [1, 2, 3]


def test_empty_list_never_matches():
    clause, params = build_where({'id': []})
    assert clause == '1 = 0'
    assert params == []


def test_invalid_column_name_is_rejected():
    with pytest.raises(ValueError):
        build_where({'id; DROP TABLE users': 1})
