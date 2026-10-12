"""Tests for `generation.llm`: the OpenRouter client, its use of the cache and the budget."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.cache import DiskCache
from crossvenue_rag.generation.llm import LLM, BudgetExceeded, Timers, observed_costs
from crossvenue_rag.generation.openrouter import (
    ModelCallError,
    ModelRequest,
    SamplingParams,
)
from crossvenue_rag.generation.schemas import InvalidResponse
from tests.generation.offline import (
    SECRET,
    FakeTransport,
    gen_config,
    openrouter_payload,
)

SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
PARAMS = SamplingParams(100, "minimal", 0.0)


def accept(content):
    return content


def llm_for(tmp_path, transport, **cfg_patch) -> tuple:
    cfg = gen_config(tmp_path, **cfg_patch)
    return cfg, LLM(cfg, transport, api_key=SECRET, timers=Timers(sleep=lambda s: None))


def test_record_carries_models_provider_tokens_cost_and_sent_params(tmp_path):
    transport = FakeTransport((200, openrouter_payload({"answer": "x"}, reasoning=30)))
    cfg, llm = llm_for(tmp_path, transport)
    result = llm.complete(
        ModelRequest(cfg.generators["gemini"], "sys", "user", SCHEMA, "answer", PARAMS),
        validate=accept,
    )
    record = result.record
    assert result.content == {"answer": "x"}
    assert record.requested_model == "google/gemini-3.8-flash"
    assert record.actual_model == "google/gemini-3.8-flash-20260901"
    assert record.provider == "Google"
    assert (record.input_tokens, record.reasoning_tokens, record.output_tokens) == (1000, 30, 170)
    assert record.cost_usd == pytest.approx(0.0012)
    assert record.retries == 0 and record.invalid_responses == 0 and not record.cached
    # Gemini's endpoint takes no temperature: none is sent, and the record says so.
    assert "temperature" not in record.params
    assert record.params["reasoning"] == {"effort": "minimal", "exclude": True}
    assert record.params["provider"]["order"] == ["google-vertex/global"]


def test_body_pins_the_provider_disables_fallback_and_requires_every_parameter(tmp_path):
    transport = FakeTransport((200, openrouter_payload({})))
    cfg, llm = llm_for(tmp_path, transport)
    llm.complete(
        ModelRequest(cfg.generators["qwen"], "s", "u", SCHEMA, "answer", PARAMS), validate=accept
    )
    body = transport.calls[0][2]
    assert body["provider"] == {
        "require_parameters": True,
        "order": ["deepinfra/bf16"],
        "allow_fallbacks": False,
    }
    assert body["temperature"] == 0.0
    assert body["response_format"]["json_schema"]["strict"] is True


def test_judge_body_sends_effort_and_no_temperature(tmp_path):
    transport = FakeTransport((200, openrouter_payload({}, provider="Anthropic")))
    cfg, llm = llm_for(tmp_path, transport)
    judge_params = SamplingParams(8000, "medium")
    llm.complete(
        ModelRequest(cfg.judge_model("opus"), "s", "u", SCHEMA, "judgement", judge_params),
        validate=accept,
    )
    body = transport.calls[0][2]
    assert body["model"] == "anthropic/claude-opus-5.5"
    assert "temperature" not in body
    assert body["reasoning"]["effort"] == "medium"
    assert body["provider"]["order"] == ["anthropic"] and not body["provider"]["allow_fallbacks"]


def test_an_invalid_response_is_retried_and_counted(tmp_path):
    transport = FakeTransport(
        (200, openrouter_payload("not json", cost=0.001)),
        (200, openrouter_payload({"wrong": 1}, cost=0.001)),
        (200, openrouter_payload({"answer": "ok"}, cost=0.001)),
    )
    cfg, llm = llm_for(tmp_path, transport)

    def validate(content):
        if "answer" not in content:
            raise InvalidResponse("schema")

    result = llm.complete(
        ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS), validate=validate
    )
    assert result.content == {"answer": "ok"}
    assert result.record.invalid_responses == 2
    assert result.record.cost_usd == pytest.approx(0.003)  # every attempt is paid for


def test_an_error_that_is_no_unusable_reply_is_not_counted_as_one(tmp_path):
    transport = FakeTransport(*[(200, openrouter_payload({"wrong": 1}, cost=0.001))] * 2)
    cfg, llm = llm_for(tmp_path, transport)

    def validate(content):
        raise UserError("a problem of the run, which a retry does not fix")

    request = ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS)
    with pytest.raises(UserError, match="a problem of the run"):
        llm.complete(request, validate=validate)
    with pytest.raises(KeyError):
        llm.complete(request, validate=lambda content: content["answer"])


def test_one_invalid_response_is_written_in_the_singular(tmp_path):
    transport = FakeTransport((200, openrouter_payload("not json", cost=0.001)))
    cfg, llm = llm_for(tmp_path, transport, max_invalid_retries=0)
    request = ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS)
    with pytest.raises(ModelCallError, match="1 invalid response in a row"):
        llm.complete(request, validate=accept)


def test_a_reply_without_choices_is_an_invalid_response_whose_usage_is_still_counted(tmp_path):
    empty = {"model": "m", "provider": "p", "usage": {"prompt_tokens": 7, "cost": 0.002}}
    transport = FakeTransport((200, empty), (200, openrouter_payload({"answer": "ok"}, cost=0.001)))
    cfg, llm = llm_for(tmp_path, transport)
    result = llm.complete(
        ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS), validate=accept
    )
    assert result.content == {"answer": "ok"}
    assert result.record.invalid_responses == 1
    assert result.record.cost_usd == pytest.approx(0.003)
    assert llm.spent_usd == pytest.approx(0.003)


def test_more_invalid_responses_than_allowed_is_an_error(tmp_path):
    transport = FakeTransport(*[(200, openrouter_payload("nope"))] * 3)
    cfg, llm = llm_for(tmp_path, transport, max_invalid_retries=2)
    with pytest.raises(ModelCallError, match="3 invalid responses"):
        llm.complete(
            ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS), validate=accept
        )


def test_the_run_stops_before_a_call_that_could_pass_the_budget(tmp_path):
    transport = FakeTransport(respond=lambda url, body: openrouter_payload({"n": 1}, cost=0.4))
    cfg, llm = llm_for(tmp_path, transport, budget_usd=1.0)
    model = cfg.generators["gemini"]
    llm.complete(ModelRequest(model, "s", "u1", SCHEMA, "a", PARAMS), validate=accept)
    llm.complete(ModelRequest(model, "s", "u2", SCHEMA, "a", PARAMS), validate=accept)
    assert llm.spent_usd == pytest.approx(0.8)
    with pytest.raises(BudgetExceeded):
        llm.complete(
            ModelRequest(model, "s", "u3", SCHEMA, "a", PARAMS), validate=accept
        )  # 0.8 + 0.4 > 1.0
    assert len(transport.calls) == 2
    # A cached call costs nothing and is never stopped.
    llm.complete(ModelRequest(model, "s", "u1", SCHEMA, "a", PARAMS), validate=accept)


def test_second_identical_call_reads_the_cache_and_a_replicate_does_not(tmp_path):
    transport = FakeTransport(respond=lambda url, body: openrouter_payload({"n": 1}))
    cfg, llm = llm_for(tmp_path, transport)
    model = cfg.generators["gemini"]
    first = llm.complete(ModelRequest(model, "s", "u", SCHEMA, "answer", PARAMS), validate=accept)
    assert llm.is_cached(ModelRequest(model, "s", "u", SCHEMA, "answer", PARAMS))
    again = llm.complete(ModelRequest(model, "s", "u", SCHEMA, "answer", PARAMS), validate=accept)
    assert len(transport.calls) == 1
    assert again.record.cached and again.record.cache_key == first.record.cache_key
    other = llm.complete(
        ModelRequest(model, "s", "u", SCHEMA, "answer", PARAMS), validate=accept, replicate=1
    )
    assert len(transport.calls) == 2 and other.record.cache_key != first.record.cache_key
    assert observed_costs(llm.cache) == {"google/gemini-3.8-flash": pytest.approx(0.0012)}


def test_the_api_key_never_reaches_the_cache_or_the_record(tmp_path):
    transport = FakeTransport((200, openrouter_payload({"answer": "x"})))
    cfg = gen_config(tmp_path)
    llm = LLM(cfg, transport, cache=DiskCache(tmp_path / "c"), api_key=SECRET)
    result = llm.complete(
        ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "answer", PARAMS), validate=accept
    )
    assert transport.calls[0][1]["Authorization"] == f"Bearer {SECRET}"
    stored = "".join(p.read_text() for p in (tmp_path / "c").rglob("*.json"))
    assert SECRET not in stored
    assert SECRET not in json.dumps(result.record.model_dump())


def test_a_cached_call_bounds_the_first_new_call_of_its_model(tmp_path):
    """The expected cost of a model's next call starts at its dearest cached call."""
    from crossvenue_rag.generation.llm import max_observed_costs

    expensive = openrouter_payload({"ok": True}, cost=2.5)
    cfg, llm = llm_for(tmp_path, FakeTransport(respond=lambda url, body: expensive))
    model = cfg.generators["gemini"]
    llm.complete(ModelRequest(model, "s", "first", SCHEMA, "x", PARAMS), validate=accept)
    assert max_observed_costs(llm.cache) == {model.model: 2.5}
    _, fresh = llm_for(tmp_path, FakeTransport(respond=lambda url, body: expensive), budget_usd=1.0)
    with pytest.raises(BudgetExceeded, match=r"expected up to \$2\.500$"):
        fresh.complete(ModelRequest(model, "s", "second", SCHEMA, "x", PARAMS), validate=accept)


def test_a_response_with_no_cost_is_recorded_without_one_and_its_model_refused_after(tmp_path):
    transport = FakeTransport((200, openrouter_payload({"a": 1}, cost=None)))
    cfg, llm = llm_for(tmp_path, transport)
    request = ModelRequest(cfg.generators["gemini"], "s", "u1", SCHEMA, "answer", PARAMS)
    record = llm.complete(request, validate=accept).record
    assert record.cost_usd is None and llm.spent_usd == 0.0
    later = ModelRequest(cfg.generators["gemini"], "s", "u2", SCHEMA, "answer", PARAMS)
    with pytest.raises(BudgetExceeded, match="reported no cost"):
        llm.complete(later, validate=accept)
    assert len(transport.calls) == 1


def test_a_body_that_is_not_a_response_is_an_invalid_attempt_whose_cost_counts(tmp_path):
    transport = FakeTransport(
        (200, {"choices": "oops", "usage": {"cost": 0.05}}),
        (200, openrouter_payload({"a": 1}, cost=0.001)),
    )
    cfg, llm = llm_for(tmp_path, transport)
    result = llm.complete(
        ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "answer", PARAMS), validate=accept
    )
    assert result.content == {"a": 1}
    assert result.record.invalid_responses == 1
    assert llm.spent_usd == pytest.approx(0.051)


def test_a_body_that_is_never_a_response_fails_naming_the_field_with_no_raw_value(tmp_path):
    body = {"choices": {"message": "x"}, "usage": {"cost": 0.001}}
    transport = FakeTransport((200, body))
    cfg, llm = llm_for(tmp_path, transport, max_invalid_retries=0)
    with pytest.raises(ModelCallError, match="not an OpenRouter response: choices: ") as failed:
        llm.complete(
            ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "answer", PARAMS),
            validate=accept,
        )
    assert "input_value" not in str(failed.value) and "{'message'" not in str(failed.value)


def test_a_cached_reply_the_check_refuses_names_the_entry_to_delete(tmp_path):
    """A re-run reads the same entry again; the model is asked anew once the entry is deleted."""
    transport = FakeTransport(respond=lambda url, body: openrouter_payload({"n": 1}))
    cfg, llm = llm_for(tmp_path, transport)
    request = ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "answer", PARAMS)
    first = llm.complete(request, validate=accept)

    def refuse(content):
        raise InvalidResponse("generator response lacks partial")

    with pytest.raises(InvalidResponse, match="lacks partial") as caught:
        llm.complete(request, validate=refuse)
    assert str(llm.cache.path(first.record.cache_key)) in caught.value.hint


def test_too_many_invalid_responses_say_how_to_allow_more(tmp_path):
    transport = FakeTransport((200, openrouter_payload("not json", cost=0.001)))
    cfg, llm = llm_for(tmp_path, transport, max_invalid_retries=0)
    request = ModelRequest(cfg.generators["gemini"], "s", "u", SCHEMA, "a", PARAMS)
    with pytest.raises(ModelCallError) as caught:
        llm.complete(request, validate=accept)
    assert "generation.max_invalid_retries" in caught.value.hint
