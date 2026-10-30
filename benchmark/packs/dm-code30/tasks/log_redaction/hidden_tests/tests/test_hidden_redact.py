import copy

from redact import redact_event


def test_input_is_not_mutated():
    event = {'password': 'hunter2'}
    snapshot = copy.deepcopy(event)
    redact_event(event)
    assert event == snapshot


def test_partial_and_case_insensitive_key_match():
    event = {'user_PASSWORD': 'x', 'Api_Key': 'y', 'AUTHORIZATION': 'z'}
    result = redact_event(event)
    assert set(result.values()) == {'***'}


def test_nested_dicts_and_lists():
    event = {
        'ctx': {'auth': {'token': 'abc'}},
        'items': [{'secret': 1}, {'name': 'ok'}],
    }
    result = redact_event(event)
    assert result['ctx']['auth']['token'] == '***'
    assert result['items'][0]['secret'] == '***'
    assert result['items'][1]['name'] == 'ok'


def test_non_string_values_and_none():
    result = redact_event({'token': 12345, 'secret': None, 'keep': 1})
    assert result['token'] == '***'
    assert result['secret'] is None
    assert result['keep'] == 1
