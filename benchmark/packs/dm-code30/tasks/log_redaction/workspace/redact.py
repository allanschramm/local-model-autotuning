SENSITIVE = ('password', 'secret', 'token', 'api_key', 'authorization')


def redact_event(event: dict) -> dict:
    """Replace sensitive values in a log event with a placeholder."""
    for key in event:
        if key in SENSITIVE:
            event[key] = '***'
    return event
