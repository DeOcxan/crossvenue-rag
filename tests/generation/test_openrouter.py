"""Tests for `crossvenue_rag.generation.openrouter`."""

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
        transport, {}, {"model": "m"}, RetryPolicy(5, 3, 1.5, waits.append)
    )
    assert payload == {"ok": True} and retries == 2
    assert waits == [1.5, 3.0]


def test_the_post_goes_to_the_openrouter_chat_completions_endpoint():
    from crossvenue_rag.generation.openrouter import OPENROUTER_URL

    transport = FakeTransport((200, {"ok": True}))
    post_with_retries(transport, {}, {"model": "m"}, RetryPolicy(5, 0, 1.0, lambda s: None))
    assert [url for url, _headers, _body in transport.calls] == [OPENROUTER_URL]


def test_a_4xx_other_than_429_fails_at_once_and_exhaustion_fails_too():
    with pytest.raises(ModelCallError, match="HTTP 400 after 0 retries"):
        post_with_retries(
            FakeTransport((400, {})),
            {},
            {"model": "m"},
            RetryPolicy(5, 3, 1.0, lambda s: None),
        )
    with pytest.raises(ModelCallError, match="HTTP 429 after 2 retries"):
        post_with_retries(
            FakeTransport((429, {}), (429, {}), (429, {})),
            {},
            {"model": "m"},
            RetryPolicy(5, 2, 1.0, lambda s: None),
        )


def test_a_failed_call_is_a_failure_outside_the_program_whose_hint_says_what_to_try():
    from crossvenue_rag.errors import ExternalFailure, UserError

    with pytest.raises(ModelCallError) as caught:
        post_with_retries(
            FakeTransport((404, {})), {}, {"model": "m"}, RetryPolicy(5, 0, 1.0, print)
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
            HttpxTransport(), {}, {"model": "m"}, RetryPolicy(5, 0, 1.0, lambda s: None)
        )


def test_a_network_error_on_every_attempt_names_the_network_and_no_http_status(monkeypatch):
    def refuse(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", refuse)
    with pytest.raises(ModelCallError) as caught:
        post_with_retries(
            HttpxTransport(), {}, {"model": "m"}, RetryPolicy(5, 1, 1.0, lambda s: None)
        )
    assert str(caught.value) == "m: a network error after 1 retry: ConnectError"
    assert "HTTP" not in str(caught.value) and "model id" not in caught.value.hint
    assert "network connection" in caught.value.hint


def test_a_body_without_a_response_format_type_is_a_value_error():
    from crossvenue_rag.generation.openrouter import sent_params

    body = {"model": "m", "messages": [], "response_format": {"type": "json_schema"}}
    assert sent_params(body) == {"model": "m", "response_format": "json_schema"}
    for broken in ({"model": "m"}, {"model": "m", "response_format": {}}):
        with pytest.raises(ValueError, match="no response_format type"):
            sent_params(broken)
