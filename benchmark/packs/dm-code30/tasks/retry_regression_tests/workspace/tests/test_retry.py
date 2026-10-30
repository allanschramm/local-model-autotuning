from retry import should_retry


def test_retries_server_errors():
    assert should_retry(status_code=503, attempt=1, max_attempts=3) is True


def test_does_not_retry_bad_request():
    assert should_retry(status_code=400, attempt=1, max_attempts=3) is False
