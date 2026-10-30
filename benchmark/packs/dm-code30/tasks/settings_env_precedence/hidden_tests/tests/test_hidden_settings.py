import pytest

from settings import resolve


def test_env_values_are_coerced_to_default_type():
    result = resolve(env={'APP_PORT': '9090', 'APP_TIMEOUT': '0.5'})
    assert result['port'] == 9090
    assert isinstance(result['port'], int)
    assert result['timeout'] == 0.5


def test_bool_env_forms():
    for text in ('1', 'true', 'YES', 'On'):
        assert resolve(env={'APP_DEBUG': text})['debug'] is True
    for text in ('0', 'false', 'NO', 'Off'):
        assert resolve(env={'APP_DEBUG': text})['debug'] is False


def test_unknown_prefixed_vars_are_ignored():
    result = resolve(env={'APP_MYSTERY': 'x'})
    assert 'mystery' not in result
    assert set(result) == {'host', 'port', 'debug', 'timeout'}


def test_precedence_order():
    result = resolve(env={'APP_PORT': '1'}, overrides={'port': 2})
    assert result['port'] == 2


def test_bad_env_value_names_the_key():
    with pytest.raises(ValueError) as info:
        resolve(env={'APP_PORT': 'not-a-number'})
    assert 'port' in str(info.value)
