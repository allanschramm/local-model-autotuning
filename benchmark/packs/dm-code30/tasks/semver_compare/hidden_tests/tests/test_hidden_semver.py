import pytest

from semver import compare


def test_numeric_not_lexical():
    assert compare("1.10.0", "1.9.0") == 1
    assert compare("2.0.0", "10.0.0") == -1


def test_prerelease_is_lower_than_release():
    assert compare("1.0.0-alpha", "1.0.0") == -1
    assert compare("1.0.0", "1.0.0-beta") == 1


def test_prerelease_ordering():
    assert compare("1.0.0-alpha.1", "1.0.0-alpha.2") == -1
    assert compare("1.0.0-alpha", "1.0.0-beta") == -1


def test_invalid_versions():
    with pytest.raises(ValueError):
        compare("1.2", "1.2.3")
    with pytest.raises(ValueError):
        compare("x.y.z", "1.2.3")
