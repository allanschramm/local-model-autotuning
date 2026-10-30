from stats import summarize


def test_even_count_and_original_order_is_preserved():
    data = [10, 2, 4, 8]
    assert summarize(data)["median"] == 6
    assert data == [10, 2, 4, 8]


def test_empty_input():
    assert summarize([]) == {
        "count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "median": None,
    }
