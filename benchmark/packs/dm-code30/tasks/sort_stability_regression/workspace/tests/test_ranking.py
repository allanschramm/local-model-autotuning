from ranking import rank_items


def test_orders_by_score():
    items = [{'name': 'a', 'score': 1}, {'name': 'b', 'score': 3}]
    assert [item['name'] for item in rank_items(items)] == ['b', 'a']
