from semver import compare


def test_equal_versions():
    assert compare("1.2.3", "1.2.3") == 0


def test_patch_difference():
    assert compare("1.2.3", "1.2.4") == -1
