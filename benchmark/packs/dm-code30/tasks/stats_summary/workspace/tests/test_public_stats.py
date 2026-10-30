from stats import summarize


def test_odd_count_summary():
    assert summarize([3, 1, 2]) == {
        "count": 3,
        "min": 1,
        "max": 3,
        "mean": 2,
        "median": 2,
    }
