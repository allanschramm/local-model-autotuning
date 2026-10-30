from retry import should_retry


def test_retry_policy_includes_timeout_and_rate_limit():
    assert should_retry(status_code=408, attempt=1, max_attempts=3) is True
    assert should_retry(status_code=429, attempt=2, max_attempts=3) is True


def test_retry_budget_is_attempt_less_than_max_attempts():
    assert should_retry(status_code=503, attempt=3, max_attempts=3) is False
    assert should_retry(exception=RuntimeError("boom"), attempt=3, max_attempts=3) is False


def test_exception_retries_before_budget_is_exhausted():
    assert should_retry(exception=TimeoutError(), attempt=1, max_attempts=2) is True
