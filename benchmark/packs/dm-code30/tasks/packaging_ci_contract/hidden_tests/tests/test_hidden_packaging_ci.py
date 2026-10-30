from pathlib import Path

from packaging_contract import (
    ci_check_commands,
    ci_install_command,
    ci_python_versions,
    dev_extra_dependencies,
    requires_python_specifier,
)


def _read(path):
    return Path(path).read_text(encoding='utf-8')


def test_packaging_contract_declares_python_310_floor():
    assert requires_python_specifier() == ">=3.10"
    assert ci_python_versions() == ["3.10", "3.11", "3.12"]
    pyproject = _read('pyproject.toml')
    assert 'requires-python = ">=3.10"' in pyproject
    assert "Programming Language :: Python :: 3.10" in pyproject
    assert "Programming Language :: Python :: 3.11" in pyproject
    assert "Programming Language :: Python :: 3.12" in pyproject


def test_dev_extra_and_ci_install_are_consistent():
    dependencies = dev_extra_dependencies()
    for tool in ('pytest', 'ruff', 'black'):
        assert any(dep == tool or dep.startswith(f'{tool}>') for dep in dependencies)
    assert ci_install_command() == 'python -m pip install -e ".[dev]"'
    pyproject = _read('pyproject.toml')
    for tool in ('pytest', 'ruff', 'black'):
        assert tool in pyproject
    workflow = _read('.github/workflows/ci.yml')
    normalized = workflow.replace("'", '"')
    assert 'python -m pip install -e ".[dev]"' in normalized


def test_ci_runs_all_keyless_checks_in_order():
    expected = [
        'python -m pytest',
        'python -m ruff check dm_agent tests',
        'python -m black --check .',
    ]
    assert ci_check_commands() == expected
    workflow = _read('.github/workflows/ci.yml')
    positions = [workflow.index(command) for command in expected]
    assert positions == sorted(positions)
    for version in ('3.10', '3.11', '3.12'):
        assert version in workflow
