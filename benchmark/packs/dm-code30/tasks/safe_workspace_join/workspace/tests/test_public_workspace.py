from pathlib import Path

import pytest

from workspace import safe_join


def test_allows_nested_relative_path(tmp_path):
    assert safe_join(tmp_path, "src/app.py") == tmp_path / "src" / "app.py"


def test_blocks_parent_traversal(tmp_path):
    with pytest.raises(ValueError):
        safe_join(tmp_path, "../outside.txt")
