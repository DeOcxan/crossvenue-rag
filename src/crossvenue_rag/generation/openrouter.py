"""OpenRouter's protocol: the request body, the POST with its retries, and the response models.

What goes over the wire and back, with no client state: the `LLM` client in
`llm.py` builds on it. The HTTP layer is a `Transport`, so tests run against a
fake with no network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue

from .. import formatting, listing
from ..config_generation import ModelConfig
from ..errors import ExternalFailure
from ..retries import NETWORK_ERROR, Attempt, Backoff, with_retries
from .api_key import KEY_NAME

# The HTTP statuses of a successful request.
HTTP_SUCCESS = range(200, 300)


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# The characters of an error message a ModelCallError keeps.
ERROR_EXCERPT = 300


@dataclass(frozen=True)
class HttpResponse:
    """An HTTP response as the client reads it."""

    status: int  # `NETWORK_ERROR` for a network error, which is retried
    payload: dict[str, JsonValue]  # the JSON body
    headers: dict[str, str]


class Transport(Protocol):
    """What the client needs to send a request; tests supply a fake."""

    def post(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> HttpResponse:
        """Sends `body` as JSON in a POST and returns the response."""
        ...


class HttpxTransport:
    """The real transport. A network error comes back as status `NETWORK_ERROR`, retried."""

    def post(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> HttpResponse:
        """Sends `body` in a POST with httpx; a non-JSON reply comes back as an error payload."""
        try:
            response = httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.TransportError as exc:
            return HttpResponse(NETWORK_ERROR, {"error": {"message": type(exc).__name__}}, {})
        try:
            payload: JsonValue = response.json()
        except ValueError:
            return HttpResponse(response.status_code, _error(response.text), dict(response.headers))
        return HttpResponse(response.status_code, response_payload(payload), dict(response.headers))


def _error(message: str) -> dict[str, JsonValue]:
    return {"error": {"message": message[:ERROR_EXCERPT]}}


def response_payload(payload: JsonValue) -> dict[str, JsonValue]:
    """A decoded JSON body in the shape `post_with_retries` reads: an object whose `error` is one.

    A body that is not an object (`["upstream"]`) becomes an error payload saying so,
    and an `error` that is not an object (`"Bad request"`) becomes one holding it
    as the message, so every failed call ends as a `ModelCallError`.
    """
    if not isinstance(payload, dict):
        return _error("the reply is a JSON value other than an object")
    error = payload.get("error")
    if error is not None and not isinstance(error, dict):
        return {**payload, "error": {"message": str(error)}}
    return payload


def failure_message(payload: dict[str, JsonValue]) -> str:
    """What a failed reply says: its error message, or the fields it has in place of one."""
    error = payload.get("error")
    message = error.get("message") if isinstance(error, dict) else None
    if message is not None:
        return str(message)[:ERROR_EXCERPT]
    fields = listing.prose([f"`{name}`" for name in sorted(payload)])
    return f"no error message in the reply{f'; its fields are {fields}' if fields else ''}"


class ModelCallError(ExternalFailure):
    """A call that failed for good: a failure outside the program.

    It got a status `retries.RETRY_STATUSES` does not retry, exhausted its retries, or
    received an invalid response on every attempt.
    """


@dataclass(frozen=True)
class RetryPolicy:
    """How a POST is retried.

    Holds the request timeout, the retries allowed, the first backoff (doubled
    per retry) and the function that waits.
    """

    timeout: float
    max_retries: int
    backoff_seconds: float
    sleep: Callable[[float], None]


def post_with_retries(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    policy: RetryPolicy,
) -> tuple[dict[str, JsonValue], int]:
    """Sends a POST, retrying what `retries.is_retryable` allows, with exponential backoff.

    Returns (payload, retries). A `Retry-After` header in seconds is honored
    when it asks for longer than the backoff.

    Raises:
        ModelCallError: The last attempt did not succeed.
    """

    def attempt() -> Attempt[dict[str, JsonValue]]:
        response = transport.post(url, headers, body, policy.timeout)
        after = response.headers.get("retry-after") or response.headers.get("Retry-After")
        return Attempt(response.payload, response.status, after)

    backoff = Backoff(policy.max_retries, policy.backoff_seconds, policy.sleep)
    last, retries = with_retries(attempt, backoff)
    if last.status in HTTP_SUCCESS:
        return last.result, retries
    message = failure_message(last.result)
    tries = formatting.plural(retries, "retry", "retries")
    raise ModelCallError(
        f"{body.get('model')}: HTTP {formatting.own_form(last.status)} after {tries}: {message}",
        hint="re-run to retry; if the same status repeats, check the model id in the config "
        f"and {KEY_NAME}",
    )


class _Lenient(BaseModel):
    """An OpenRouter object: a field this model does not declare is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class TokenDetails(_Lenient):
    """`usage.completion_tokens_details`: the reasoning share of the completion."""

    reasoning_tokens: int | None = None


class Usage(_Lenient):
    """`usage` of a response: tokens and the cost OpenRouter charged."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    completion_tokens_details: TokenDetails | None = None

    @property
    def reasoning(self) -> int:
        """Reasoning tokens, counted inside `completion_tokens`."""
        details = self.completion_tokens_details
        return (details.reasoning_tokens or 0) if details else 0


class Message(_Lenient):
    """A choice's message; `content` is the model's answer text."""

    content: str | None = None


class Choice(_Lenient):
    """One completion choice."""

    message: Message | None = None


class OpenRouterResponse(_Lenient):
    """A chat completion as OpenRouter returns it, parsed where it enters the client.

    Every field has a default, so a reply missing its answer parses and then
    counts as an invalid response, as a reply with an empty answer does.
    """

    model: str | None = None
    provider: str | None = None
    usage: Usage | None = None
    choices: list[Choice] = []

    def text(self) -> str:
        """The first choice's content.

        Raises:
            ValueError: The reply has no choice, or no message content.
        """
        message = self.choices[0].message if self.choices else None
        if message is None or message.content is None:
            raise ValueError("the response has no message content")
        return message.content


def parse_json(text: str) -> dict[str, JsonValue]:
    """The model's JSON object; a code fence around it is accepted.

    Raises:
        ValueError: The JSON is not an object.
    """
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
    if fenced:
        stripped = fenced.group(1)
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


@dataclass(frozen=True)
class SamplingParams:
    """The sampling parameters of a call.

    `effort` is the reasoning effort (`reasoning_effort` of a generator, `effort` of a
    judge), sent when set; `temperature` is sent when set and the model's endpoint
    accepts it.
    """

    max_tokens: int
    effort: str | None
    temperature: float | None = None


@dataclass(frozen=True)
class ModelRequest:
    """One structured call.

    Holds the model, the system and user messages, the JSON schema the reply
    must follow and its name, and the sampling parameters.
    """

    model: ModelConfig
    system: str
    user: str
    schema: dict[str, Any]
    schema_name: str
    params: SamplingParams


def request_body(request: ModelRequest) -> dict[str, Any]:
    """The OpenRouter request body for `request`, with its `SamplingParams`."""
    model, params = request.model, request.params
    body: dict[str, Any] = {
        "model": model.model,
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "strict": True,
                "schema": request.schema,
            },
        },
        "max_tokens": params.max_tokens,
        "usage": {"include": True},
        # Only providers that honor every parameter sent, the JSON schema
        # included, may serve the request.
        "provider": {"require_parameters": True},
    }
    if params.effort:
        body["reasoning"] = {"effort": params.effort, "exclude": True}
    if params.temperature is not None and model.temperature_supported:
        body["temperature"] = params.temperature
    if model.provider_order:
        body["provider"] = {
            **body["provider"],
            "order": list(model.provider_order),
            "allow_fallbacks": False,
        }
    return body


def sent_params(body: dict[str, Any]) -> dict[str, Any]:
    """What the body asked of the model, without the messages.

    Raises:
        ValueError: `body` has no `response_format` type, as a `request_body` has.
    """
    response_format = body.get("response_format", {})
    if isinstance(response_format, dict) and "type" not in response_format:
        raise ValueError("the request body has no response_format type")
    return {k: v for k, v in body.items() if k not in ("messages", "response_format")} | {
        "response_format": response_format["type"]
    }
