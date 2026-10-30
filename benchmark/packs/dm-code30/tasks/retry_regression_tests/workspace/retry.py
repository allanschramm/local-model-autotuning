def should_retry(status_code=None, exception=None, attempt=1, max_attempts=3):
    if attempt > max_attempts:
        return False
    if exception is not None:
        return True
    if status_code is None:
        return False
    return status_code >= 500
