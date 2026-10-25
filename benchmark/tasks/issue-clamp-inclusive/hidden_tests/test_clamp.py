import clamp as m


def test_inclusive_max():
    assert m.clamp(10, 0, 10) == 10
    assert m.clamp(-1, 0, 10) == 0
    assert m.clamp(3, 0, 10) == 3
