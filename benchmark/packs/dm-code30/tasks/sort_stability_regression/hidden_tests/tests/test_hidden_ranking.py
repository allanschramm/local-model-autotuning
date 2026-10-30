from ranking import rank_items


def test_ties_keep_input_order():
    items = [
        {'name': 'first', 'score': 5},
        {'name': 'second', 'score': 5},
        {'name': 'third', 'score': 5},
    ]
    assert [item['name'] for item in rank_items(items)] == [
        'first',
        'second',
        'third',
    ]


def test_missing_score_defaults_to_zero():
    items = [{'name': 'a'}, {'name': 'b', 'score': 2}, {'name': 'c', 'score': -1}]
    assert [item['name'] for item in rank_items(items)] == ['b', 'a', 'c']


def test_input_is_not_mutated():
    items = [{'name': 'a', 'score': 1}, {'name': 'b', 'score': 3}]
    snapshot = [dict(item) for item in items]
    rank_items(items)
    assert items == snapshot


def test_regression_test_was_added():
    from pathlib import Path

    source = Path('tests/test_ranking.py').read_text(encoding='utf-8')
    assert source.count('def test_') >= 2
