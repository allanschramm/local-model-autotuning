import pytest

from jobs import JobRunner


def test_falsy_results_are_still_cached():
    runner = JobRunner()
    calls = []

    def work():
        calls.append(1)
        return None

    assert runner.submit('a', work) is None
    assert runner.submit('a', work) is None
    assert len(calls) == 1
    assert runner.has_run('a') is True


def test_zero_and_empty_string_are_cached():
    runner = JobRunner()
    assert runner.submit('z', lambda: 0) == 0
    assert runner.submit('z', lambda: 99) == 0
    assert runner.submit('e', lambda: '') == ''
    assert runner.submit('e', lambda: 'x') == ''


def test_failure_is_not_cached_and_retries():
    runner = JobRunner()
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('boom')
        return 'ok'

    with pytest.raises(RuntimeError):
        runner.submit('f', flaky)
    assert runner.has_run('f') is False
    assert runner.submit('f', flaky) == 'ok'
    assert len(attempts) == 2


def test_keys_are_independent():
    runner = JobRunner()
    assert runner.submit('a', lambda: 1) == 1
    assert runner.submit('b', lambda: 2) == 2
    assert runner.has_run('c') is False
