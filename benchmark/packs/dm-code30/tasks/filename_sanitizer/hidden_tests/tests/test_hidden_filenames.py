from filenames import sanitize


def test_windows_illegal_characters():
    assert sanitize('re:port*v2?.txt') == 're_port_v2_.txt'


def test_collapses_and_trims():
    assert sanitize('a///b') == 'a_b'
    assert sanitize('  ..name..  ') == 'name'
    assert sanitize('___x___') == 'x'


def test_unicode_is_preserved():
    assert sanitize('报告-2024.txt') == '报告-2024.txt'


def test_reserved_device_names():
    assert sanitize('CON') == '_CON'
    assert sanitize('nul.txt') == '_nul.txt'
    assert sanitize('COM1') == '_COM1'
    assert sanitize('CONFIG') == 'CONFIG'


def test_only_bad_characters_and_length_cap():
    assert sanitize('///') == 'untitled'
    assert sanitize('...') == 'untitled'
    assert len(sanitize('x' * 300)) == 100
