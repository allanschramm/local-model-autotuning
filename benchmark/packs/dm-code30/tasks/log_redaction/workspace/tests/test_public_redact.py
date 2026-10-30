from redact import redact_event


def test_top_level_secret():
    assert redact_event({'password': 'hunter2'})['password'] == '***'
