def schedule(attempts: int, base: float = 1.0, cap: float = 60.0):
    """Return the backoff delay for each retry attempt."""
    delays = []
    for index in range(attempts):
        delays.append(base * 2 ** index)
    return delays
