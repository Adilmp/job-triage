"""Errors shared by anything that talks to an outside service (ATS boards, Jev, Ollama)."""


class RetryableError(Exception):
    """Try again later: 429, 5xx, timeouts, connection errors. `retry_after` is the site's Retry-After in seconds,
    when it sent one; honour it rather than the default backoff (apply.workable.com asks for ~24h after a burst)."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(Exception):
    """Retrying won't help: the board is gone (404), access is denied, or the response isn't what the API promises."""
