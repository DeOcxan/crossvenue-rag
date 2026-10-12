"""Tests for `crossvenue_rag.generation.openrouter`: retries and failed calls."""

from __future__ import annotations

import httpx
import pytest

from crossvenue_rag.generation.openrouter import (
    HttpxTransport,
    ModelCallError,
    RetryPolicy,
    post_with_retries,
)
from tests.generation.offline import (
    FakeTransport,
)


def test_retries_429_and_5xx_with_exponential_backoff():
    transport = FakeTransport((429, {}), (503, {}), (200, {"ok": True}))
    waits: list[float] = []
    payload, retries = post_with_retries(
        transport, "u", {}, {"model": "m"}, RetryPolicy(5, 3, 1.5, waits.append)
    )
    assert payload == {"ok": True} and retries == 2
    assert waits == [1.5, 3.0]


def test_a_4xx_other_than_429_fails_at_once_and_exhaustion_fails_too():
    with pytest.raises(ModelCallError, match="HTTP 400 after 0 retries"):
        post_with_retries(
            FakeTransport((400, {})),
            "u",
            {},
            {"model": "m"},
            RetryPolicy(5, 3, 1.0, lambda s: None),
        )
    with pytest.raises(ModelCallError, match="HTTP 429 after 2 retries"):
        post_with_retries(
            FakeTransport((429, {}), (429, {}), (429, {})),
            "u",
            {},
            {"model": "m"},
            RetryPolicy(5, 2, 1.0, lambda s: None),
        )


def test_a_failed_call_is_a_failure_outside_the_program_whose_hint_says_what_to_try():
    from crossvenue_rag.errors import ExternalFailure, UserError

    with pytest.raises(ModelCallError) as caught:
        post_with_retries(
            FakeTransport((404, {})), "u", {}, {"model": "m"}, RetryPolicy(5, 0, 1.0, print)
        )
    assert isinstance(caught.value, ExternalFailure)
    assert not isinstance(caught.value, UserError)
    assert "re-run" in caught.value.hint and "OPENROUTER_API_KEY" in caught.value.hint


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (400, {"error": "Bad request"}, "HTTP 400 after 0 retries: Bad request"),
        (502, ["upstream"], "HTTP 502 after 0 retries: the reply is a JSON value other than an"),
        (
            400,
            {"detail": "model not found"},
            "HTTP 400 after 0 retries: no error message in the reply; its fields are `detail`$",
        ),
    ],
)
def test_a_body_that_is_not_the_expected_object_fails_as_a_model_call_error(
    monkeypatch, status, body, message
):
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: httpx.Response(status, json=body))
    with pytest.raises(ModelCallError, match=message):
        post_with_retries(
            HttpxTransport(), "u", {}, {"model": "m"}, RetryPolicy(5, 0, 1.0, lambda s: None)
        )


def test_a_body_without_a_response_format_type_is_a_value_error():
    from crossvenue_rag.generation.openrouter import sent_params

    body = {"model": "m", "messages": [], "response_format": {"type": "json_schema"}}
    assert sent_params(body) == {"model": "m", "response_format": "json_schema"}
    for broken in ({"model": "m"}, {"model": "m", "response_format": {}}):
        with pytest.raises(ValueError, match="no response_format type"):
            sent_params(broken)
