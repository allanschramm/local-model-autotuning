from solution import find_Rotations


def test_original_mbpp_assertions():

    assert find_Rotations("aaaa") == 1
    assert find_Rotations("ab") == 2
    assert find_Rotations("abc") == 3
