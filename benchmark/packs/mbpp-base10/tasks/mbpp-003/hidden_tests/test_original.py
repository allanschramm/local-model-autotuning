from solution import is_not_prime


def test_original_mbpp_assertions():

    assert is_not_prime(1) == True
    assert is_not_prime(2) == False
    assert is_not_prime(10) == True
    assert is_not_prime(35) == True
    assert is_not_prime(37) == False
