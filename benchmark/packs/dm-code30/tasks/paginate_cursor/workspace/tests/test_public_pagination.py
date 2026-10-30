from pagination import page


ITEMS = [{'id': n} for n in range(1, 6)]


def test_first_page():
    window, cursor = page(ITEMS, None, 2)
    assert [item['id'] for item in window] == [1, 2]
    assert cursor == 2
