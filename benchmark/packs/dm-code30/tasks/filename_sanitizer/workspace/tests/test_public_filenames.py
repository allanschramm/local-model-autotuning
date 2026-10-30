from filenames import sanitize


def test_replaces_separators():
    assert sanitize('a/b') == 'a_b'


def test_empty_input():
    assert sanitize('') == 'untitled'
