from pathlib import Path

from cli_docs import CONFIG_OPTIONS, render_config_table


def _table_flags(table):
    rows = [line for line in table.splitlines() if line.startswith('| `--')]
    return [row.split('|')[1].strip().strip('`') for row in rows]


def test_all_config_options_are_documented_and_sorted():
    table = render_config_table()
    expected_flags = sorted(item['flag'] for item in CONFIG_OPTIONS)
    assert _table_flags(table) == expected_flags
    for item in CONFIG_OPTIONS:
        assert f"`{item['flag']}`" in table
        assert f"`{item['env']}`" in table
        assert f"`{item['default']}`" in table


def test_docs_embed_generated_table_between_markers():
    docs = Path('docs/configuration.md').read_text(encoding='utf-8')
    expected = render_config_table()
    assert '<!-- CONFIG_TABLE -->' in docs
    assert '<!-- /CONFIG_TABLE -->' in docs
    assert expected in docs
    assert 'DM_RETRIES' in docs
