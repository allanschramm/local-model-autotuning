from cli_docs import render_config_table


def test_config_table_mentions_provider_and_timeout():
    table = render_config_table()
    assert "--provider" in table
    assert "--timeout" in table
