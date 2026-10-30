from intervals import merge


def test_overlapping():
    assert merge([(1, 4), (2, 6)]) == [(1, 6)]


def test_disjoint():
    assert merge([(1, 2), (5, 6)]) == [(1, 2), (5, 6)]
