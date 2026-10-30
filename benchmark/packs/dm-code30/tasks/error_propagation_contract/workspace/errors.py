"""Public error contract for the service layer.

NotFound     -> the requested key does not exist
Unavailable  -> the backend failed for any other reason

Both must carry the original low-level exception as __cause__ so callers
can log the root cause. Programming errors (TypeError, ValueError raised
by our own code) must propagate unchanged.
"""


class ServiceError(Exception):
    pass


class NotFound(ServiceError):
    pass


class Unavailable(ServiceError):
    pass
