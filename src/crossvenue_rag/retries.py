"""Retrying an HTTP request: which statuses are worth another attempt, and how long to wait.

The corpus fetcher and the model client both call `with_retries`, so a 404 is never
retried and a 429 always is, whichever of them hits it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# The statuses worth another attempt: a timeout (408), a rate limit (429) and the
# server errors that pass. Any other 4xx is the request's fault and fails at once.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 529})
# The status a caller reports for a network error, which is always retried.
NETWORK_ERROR = 0
# Each retry waits this many times as long as the one before (exponential backoff).
BACKOFF_FACTOR = 2


@dataclass(frozen=True)
class Backoff:
    """How often a request is tried again and how long each wait is.

    The wait doubles from `first_seconds` per retry, or is the server's `Retry-After`
    when that asks for longer; `sleep` waits it out (tests pass a fake).
    """

    max_retries: int
    first_seconds: float
    sleep: Callable[[float], None]


@dataclass(frozen=True)
class Attempt[T]:
    """One try of a request: its result, the HTTP status, and any `Retry-After` in seconds."""

    result: T
    status: int
    retry_after: str | None = None


def is_retryable(status: int) -> bool:
    """Whether a request that ended with `status` is worth another attempt."""
    return status == NETWORK_ERROR or status in RETRY_STATUSES


def with_retries[T](attempt: Callable[[], Attempt[T]], backoff: Backoff) -> tuple[Attempt[T], int]:
    """The result of `attempt`, tried until it succeeds, fails for good, or the retries run out.

    Returns the last attempt and the number of retries made; the caller decides
    what a failed last attempt means.
    """
    retries = 0
    while True:
        tried = attempt()
        if not is_retryable(tried.status) or retries >= backoff.max_retries:
            return tried, retries
        delay = backoff.first_seconds * BACKOFF_FACTOR**retries
        after = tried.retry_after
        if after and after.replace(".", "", 1).isdigit():
            delay = max(delay, float(after))
        backoff.sleep(delay)
        retries += 1
