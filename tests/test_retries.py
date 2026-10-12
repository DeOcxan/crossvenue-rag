"""Tests for `retries`: which statuses are retried, and the backoff between tries."""

from __future__ import annotations

from crossvenue_rag.retries import NETWORK_ERROR, Attempt, Backoff, is_retryable, with_retries


def test_a_timeout_a_rate_limit_a_server_error_and_a_network_error_are_retried_only():
    assert all(is_retryable(s) for s in (408, 429, 500, 503, NETWORK_ERROR))
    assert not any(is_retryable(s) for s in (200, 400, 401, 404, 409, 501))


def test_a_missing_page_is_tried_once_and_a_rate_limit_until_it_passes():
    waits: list[float] = []
    backoff = Backoff(max_retries=5, first_seconds=1.0, sleep=waits.append)
    last, retries = with_retries(lambda: Attempt("gone", 404), backoff)
    assert (last.status, retries, waits) == (404, 0, [])

    answers = iter([Attempt("", 429, "7"), Attempt("", 503), Attempt("ok", 200)])
    last, retries = with_retries(lambda: next(answers), backoff)
    # The Retry-After of 7 s exceeds the first backoff of 1 s; the second wait doubles.
    assert (last.result, retries, waits) == ("ok", 2, [7.0, 2.0])


def test_the_last_failure_is_returned_when_the_retries_run_out():
    backoff = Backoff(max_retries=2, first_seconds=0.5, sleep=lambda _: None)
    last, retries = with_retries(lambda: Attempt("busy", 503), backoff)
    assert (last.status, retries) == (503, 2)
