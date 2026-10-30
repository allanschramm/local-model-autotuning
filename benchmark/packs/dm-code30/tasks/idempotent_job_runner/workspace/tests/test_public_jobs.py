from jobs import JobRunner


def test_runs_once():
    runner = JobRunner()
    calls = []
    work = lambda: calls.append(1) or 'done'
    assert runner.submit('a', work) == 'done'
    assert runner.submit('a', work) == 'done'
    assert len(calls) == 1
