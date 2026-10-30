from pathlib import Path

import pytest

from workspace import safe_join


def test_blocks_absolute_path(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, str(tmp_path.parent / "outside.txt"))


def test_blocks_sibling_prefix_escape(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sibling = tmp_path / "repo-other" / "file.txt"
    with pytest.raises(ValueError):
        safe_join(root, "../repo-other/file.txt")


def test_normalizes_dot_segments_inside_root(tmp_path):
    assert safe_join(tmp_path, "src/../README.md") == tmp_path / "README.md"
