import words as m


def test_whitespace_kinds():
    assert m.word_count("a\nb\tc") == 3
    assert m.word_count("one two") == 2
    assert m.word_count("   ") == 0
