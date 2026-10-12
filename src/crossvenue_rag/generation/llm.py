"""The model client: calls through OpenRouter with the cache, the budget and the call records.

One thin client over OpenRouter's OpenAI-compatible API with no SDK
(docs/adr/0001-no-framework.md); the wire protocol it speaks is in `openrouter.py`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import JsonValue, ValidationError

from .. import formatting
from ..config_generation import GenerationConfig
from ..errors import UserError, validation_problems
from ..models import MS_PER_SECOND
from ..stats import mean
from .api_key import load_api_key
from .cache import DiskCache, cache_key, read_entry
from .openrouter import (
    OPENROUTER_URL,
    HttpxTransport,
    ModelCallError,
    ModelRequest,
    OpenRouterResponse,
    RetryPolicy,
    Transport,
    Usage,
    parse_json,
    post_with_retries,
    request_body,
    sent_params,
)
from .records import CallRecord
from .schemas import InvalidResponse

log = logging.getLogger(__name__)

# Decimals a call's cost in USD is saved with: a millionth of a cent, below any price.
COST_DECIMALS = 8
# The decimals of a call's latency in its record.
LATENCY_DECIMALS = 1


@dataclass(slots=True)
class LLMResult:
    """A validated model reply and the record of the call that produced it."""

    content: dict[str, JsonValue]
    record: CallRecord


class BudgetExceeded(UserError):
    """The next call could take the run past `generation.budget_usd`."""


@dataclass(frozen=True)
class Timers:
    """How the client waits between retries and measures latency; tests replace both."""

    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.perf_counter


class LLM:
    """One entry point for every model call.

    It handles the request, retries, validation, cache, budget and record.

    Two kinds of retry, both bounded by the config:

    - HTTP: `retries.RETRY_STATUSES` and network errors, with exponential backoff
      (`max_retries`, `backoff_seconds`).
    - content: a response that is not valid JSON or fails its schema check is
      asked for again, up to `max_invalid_retries` times, the same rule for
      every model.

    `budget_usd` caps what one run may spend. Before each uncached call the
    client adds the most expensive call of that model on record, from this run
    or from the cache, to what the run has spent. If the sum would pass the
    budget, it raises `BudgetExceeded` instead of calling. The first call of a
    model with no cached call has no known cost and is not bounded. A response that
    reports no cost is recorded with none (`cost_usd` None, never 0), and every later
    uncached call of that model is refused, since what it would cost is unknown. One
    `LLM` is one run: the service (`make serve`) holds one for the life of the process
    and lets one request at a time use it. Its budget covers every uncached call until a
    restart, and a cached reply is returned whatever has been spent.
    """

    def __init__(
        self,
        cfg: GenerationConfig,
        transport: Transport | None = None,
        cache: DiskCache | None = None,
        api_key: str | None = None,
        timers: Timers = Timers(),
    ) -> None:
        """A client for `cfg`; an argument left out defaults from `cfg` or the environment."""
        self.cfg = cfg
        self.transport = transport or HttpxTransport()
        self.cache = cache or DiskCache(cfg.cache_dir)
        self._api_key = api_key
        self._clock = timers.clock
        self._policy = RetryPolicy(
            cfg.timeout_seconds, cfg.max_retries, cfg.backoff_seconds, timers.sleep
        )
        self.budget_usd = cfg.budget_usd
        self.spent_usd = 0.0
        self._max_cost: dict[str, float] | None = None

    def _key(self) -> str:
        if self._api_key is None:
            self._api_key = load_api_key()
        return self._api_key

    def _known_costs(self) -> dict[str, float]:
        if self._max_cost is None:
            self._max_cost = max_observed_costs(self.cache)
        return self._max_cost

    def _check_budget(self, model: str) -> None:
        expected = self._known_costs().get(model, 0.0)
        if math.isinf(expected):
            raise BudgetExceeded(
                f"a {model} response reported no cost, so what the next call costs is unknown",
                hint="re-run: cached calls cost nothing; an uncached call needs a reported cost",
            )
        if self.spent_usd + expected > self.budget_usd:
            raise BudgetExceeded(
                f"run budget {formatting.cost(self.budget_usd)} reached: "
                f"{formatting.cost(self.spent_usd)} spent, next {model} call expected up to "
                f"{formatting.cost(expected)}",
                hint="raise generation.budget_usd, or re-run: cached calls cost nothing",
            )

    def complete(
        self,
        request: ModelRequest,
        validate: Callable[[dict[str, JsonValue]], object],
        replicate: int = 0,
    ) -> LLMResult:
        """The validated reply to `request`, from the cache or the API.

        `replicate` > 0 is a deliberate repeat: a separate cache entry. The
        result's `CallRecord` holds the model id requested, the id and provider
        the response reports, input, reasoning and answer tokens separately, the
        cost OpenRouter reports, latency, HTTP retries, invalid responses, and
        the parameters sent.
        """
        body = request_body(request)
        key = cache_key({"body": body, "replicate": replicate})
        hit = self.cache.get(key)
        if hit is not None:
            _validate_cached(validate, hit.content, self.cache.path(key))
            return LLMResult(hit.content, hit.record)

        model = request.model.model
        totals = _CallTotals()
        while True:
            self._check_budget(model)
            payload = self._attempt(body, totals)
            try:
                response = OpenRouterResponse.model_validate(payload)
            except ValidationError as exc:
                # A body that is not a response still reports what it cost, if it can.
                self._accumulate(model, _usage_of(payload), totals)
                reason = f"not an OpenRouter response: {validation_problems(exc)}"
                self._invalid(model, InvalidResponse(reason), totals)
                continue
            self._accumulate(model, response.usage or Usage(), totals)
            try:
                content = _parsed(response, validate)
                break
            # What an unusable reply raises: no text or not JSON (ValueError), or a
            # validator's refusal (InvalidResponse).
            except (ValueError, InvalidResponse) as exc:
                self._invalid(model, exc, totals)
        record = totals.record(model, response, key, sent_params(body))
        self.cache.put(key, {"request": body, "content": content, "record": record.to_json()})
        return LLMResult(content, record)

    def _invalid(self, model: str, exc: Exception, totals: _CallTotals) -> None:
        """Counts an unusable reply; past `max_invalid_retries` the call fails for good."""
        totals.invalid += 1
        log.info("%s: invalid response (%s)", model, type(exc).__name__)
        if totals.invalid > self.cfg.max_invalid_retries:
            invalid = formatting.plural(totals.invalid, "invalid response")
            raise ModelCallError(
                f"{model}: {invalid} in a row ({exc})",
                hint="re-run to ask again, or raise generation.max_invalid_retries",
            ) from exc

    def _attempt(self, body: dict[str, Any], totals: _CallTotals) -> dict[str, JsonValue]:
        """One request, with its HTTP retries; its latency and retries go into `totals`."""
        headers = {
            "Authorization": f"Bearer {self._key()}",
            "Content-Type": "application/json",
            "X-Title": "crossvenue-rag",
        }
        start = self._clock()
        payload, http_retries = post_with_retries(
            self.transport, OPENROUTER_URL, headers, body, self._policy
        )
        totals.latency_ms += (self._clock() - start) * MS_PER_SECOND
        totals.retries += http_retries
        return payload

    def _accumulate(self, model: str, usage: Usage, totals: _CallTotals) -> None:
        """Adds one response's tokens and cost to the call and its cost to the run's budget."""
        known = self._known_costs()
        if usage.cost is None:
            totals.cost_unknown = True
            known[model] = math.inf
        else:
            totals.cost += usage.cost
            self.spent_usd += usage.cost
            known[model] = max(known.get(model, 0.0), usage.cost)
        totals.tokens_in += usage.prompt_tokens
        totals.tokens_reasoning += usage.reasoning
        totals.tokens_out += usage.completion_tokens - usage.reasoning

    def is_cached(self, request: ModelRequest) -> bool:
        """Whether this call, not a replicate, would be served from the cache. Makes no call."""
        body = request_body(request)
        return self.cache.get(cache_key({"body": body, "replicate": 0})) is not None


# Builds the model client a job or a question calls from the generation config: `LLM`.
LLMFactory = Callable[[GenerationConfig], LLM]


@dataclass(slots=True)
class _CallTotals:
    """What the attempts of one call add up to, for its `CallRecord`."""

    cost: float = 0.0
    cost_unknown: bool = False
    latency_ms: float = 0.0
    retries: int = 0
    invalid: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_reasoning: int = 0

    def record(
        self, model: str, response: OpenRouterResponse, key: str, params: dict[str, Any]
    ) -> CallRecord:
        """The call's record: the last response's model and provider, the totals, `params`."""
        return CallRecord(
            requested_model=model,
            actual_model=response.model or "",
            provider=response.provider or "",
            input_tokens=self.tokens_in,
            output_tokens=self.tokens_out,
            reasoning_tokens=self.tokens_reasoning,
            cost_usd=None if self.cost_unknown else round(self.cost, COST_DECIMALS),
            latency_ms=round(self.latency_ms, LATENCY_DECIMALS),
            retries=self.retries,
            invalid_responses=self.invalid,
            cache_key=key,
            params=params,
            cached=False,
        )


def _usage_of(payload: dict[str, JsonValue]) -> Usage:
    """The `usage` of a body that is not a valid response; empty when it has none."""
    try:
        return Usage.model_validate(payload.get("usage") or {})
    except ValidationError:
        return Usage()


def _validate_cached(
    validate: Callable[[dict[str, JsonValue]], object], content: dict[str, JsonValue], path: Path
) -> None:
    """Checks a cached reply; a refused one names its entry, which a re-run would read again."""
    try:
        validate(content)
    except InvalidResponse as exc:
        raise InvalidResponse(str(exc), hint=f"delete {path} to ask the model again") from exc


def _parsed(
    response: OpenRouterResponse, validate: Callable[[dict[str, JsonValue]], object]
) -> dict[str, JsonValue]:
    """The reply's JSON object, validated; an exception `LLM.complete` counts as invalid."""
    content = parse_json(response.text())
    validate(content)
    return content


def _cached_costs(cache: DiskCache) -> dict[str, list[float]]:
    totals: dict[str, list[float]] = {}
    if not cache.directory.exists():
        return {}
    for path in cache.directory.rglob("*.json"):
        record = read_entry(path).record
        if record.cost_usd is not None:
            totals.setdefault(record.requested_model, []).append(record.cost_usd)
    return totals


def observed_costs(cache: DiskCache) -> dict[str, float]:
    """Mean cost per call by requested model, over every cached response."""
    return {model: mean(costs) for model, costs in _cached_costs(cache).items()}


def max_observed_costs(cache: DiskCache) -> dict[str, float]:
    """Most expensive cached call by requested model: the budget's expected cost."""
    return {model: max(costs) for model, costs in _cached_costs(cache).items()}
