import pytest

from intervals import merge


def test_touching_intervals_are_merged():
    assert merge([(1, 3), (3, 5)]) == [(1, 5)]


def test_unsorted_input_is_handled():
    assert merge([(5, 6), (1, 3), (2, 4)]) == [(1, 4), (5, 6)]


def test_input_list_is_not_mutated():
    data = [(5, 6), (1, 3)]
    merge(data)
    assert data == [(5, 6), (1, 3)]


def test_fully_contained_interval():
    assert merge([(1, 10), (2, 3)]) == [(1, 10)]


def test_empty_and_invalid():
    assert merge([]) == []
    with pytest.raises(ValueError):
        merge([(5, 1)])
