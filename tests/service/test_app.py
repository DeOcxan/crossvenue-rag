"""Tests for `service.app`: POST /answer and GET /health, with fakes and no network."""

from __future__ import annotations

from fastapi.testclient import TestClient

from crossvenue_rag.config import Config
from crossvenue_rag.generation.llm import LLM, Timers
from crossvenue_rag.service.app import create_app
from tests.generation.offline import (
    ASK_ANSWER,
    SECRET,
    DetectingRetriever,
    FakeTransport,
    gen_config,
    openrouter_payload,
)


def client(tmp_path, respond, *, raise_server_exceptions: bool = True) -> TestClient:
    cfg = Config(generation=gen_config(tmp_path, generator_prompt="prompts/generator.md"))
    # A clock that never advances: the byte-exact bodies below record latency_ms 0.0.
    timers = Timers(sleep=lambda _: None, clock=lambda: 0.0)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET, timers=timers)
    app = create_app(cfg, DetectingRetriever(), llm)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def test_health_reports_ok_and_the_generators(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        response = c.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok" and "gemini" in response.json()["generators"]


def test_answer_returns_what_ask_prints_as_json(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        response = c.post("/answer", json={"question": "binance testnet ip ban?"})
    assert response.status_code == 200
    body = response.json()
    assert body["claims"][0]["citations"][0]["doc_path"] == "testnet/rest-api.md"
    assert body["refused"] is False and body["generator"] == "gemini"


def test_bad_requests_are_422_and_model_failures_502(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        assert c.post("/answer", json={"question": ""}).status_code == 422
        unknown = c.post("/answer", json={"question": "q", "generator": "nope"})
        detail = unknown.json()["detail"]
        assert unknown.status_code == 422 and "'nope' is not a generator" in detail
        assert "(pass --generator one of: " in detail
    with client(tmp_path / "bad", lambda url, body: openrouter_payload("not json")) as c:
        assert c.post("/answer", json={"question": "binance ban?"}).status_code == 502


def test_an_unknown_retriever_is_422_not_500(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        response = c.post("/answer", json={"question": "q", "retriever": "venue_split_oracle"})
    assert response.status_code == 422


def test_without_a_key_the_service_starts_and_answering_is_503(tmp_path, monkeypatch):
    import crossvenue_rag.service.app as app_module
    from crossvenue_rag.errors import UserError

    def no_key():
        raise UserError("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(app_module, "load_api_key", no_key)
    cfg = Config(generation=gen_config(tmp_path, generator_prompt="prompts/generator.md"))
    with TestClient(create_app(cfg, DetectingRetriever())) as c:
        health = c.get("/health").json()
        response = c.post("/answer", json={"question": "binance ban?"})
    assert health["answering"] == "OPENROUTER_API_KEY is not set"
    assert response.status_code == 503 and "not set" in response.json()["detail"]


def test_a_spent_budget_is_503_for_a_request_that_needs_a_paid_call(tmp_path):
    cfg = Config(
        generation=gen_config(tmp_path, generator_prompt="prompts/generator.md", budget_usd=0.01)
    )
    transport = FakeTransport(respond=lambda url, body: openrouter_payload(ASK_ANSWER, cost=0.02))
    llm = LLM(cfg.generation, transport, api_key=SECRET)
    with TestClient(create_app(cfg, DetectingRetriever(), llm)) as c:
        first = c.post("/answer", json={"question": "binance ban?"})
        second = c.post("/answer", json={"question": "binance ban, again?"})
        cached = c.post("/answer", json={"question": "binance ban?"})
        spent = c.get("/health").json()["spent_usd"]
    assert first.status_code == 200
    assert second.status_code == 503 and "budget" in second.json()["detail"]
    # The first question's answer is in the cache, which the budget does not bound.
    assert cached.status_code == 200
    assert spent == 0.02


def test_the_openapi_schema_describes_both_response_bodies(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        spec = c.get("/openapi.json").json()
    ok = {path: next(iter(ops.values()))["responses"]["200"] for path, ops in spec["paths"].items()}
    schemas = spec["components"]["schemas"]
    for path, model, field in (
        ("/health", "HealthResponse", "spent_usd"),
        ("/answer", "AskResult", "claims"),
    ):
        body = ok[path]["content"]["application/json"]["schema"]
        assert body == {"$ref": f"#/components/schemas/{model}"}
        assert field in schemas[model]["properties"]


# The bodies the endpoints returned for the stubbed answer before they declared
# response models; declaring them changed the schema and left the JSON byte-identical.
HEALTH_JSON = (
    b'{"status":"ok","answering":"available","generators":["gemini","qwen"],"top_k":8,'
    b'"budget_usd":3.0,"spent_usd":0.0}'
)
ANSWER_JSON = (
    b'{"question":"binance testnet ip ban?","venues_detected":["binance"],'
    b'"retriever":"venue_split","generator":"gemini",'
    b'"answer":"On testnet, Binance bans scale from 2 minutes to 3 days [1].",'
    b'"claims":[{"text":"bans scale 2m to 3d","venue":"binance","citations":[{"chunk":1,'
    b'"venue":"binance","doc_path":"testnet/rest-api.md","url":"https://t",'
    b'"environment":"testnet","api_version":null,"product":null}]}],"outcome":"full",'
    b'"refused":false,"asked_clarification":false,"partial":false,"missing":[],'
    b'"model":"google/gemini-3.8-flash-20260901","provider":"Google","cost_usd":0.0012,'
    b'"latency_ms":0.0,"cached":false}'
)


def test_the_response_bodies_are_the_json_the_endpoints_always_returned(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        health = c.get("/health").content
        answer = c.post("/answer", json={"question": "binance testnet ip ban?"}).content
    assert health == HEALTH_JSON
    assert answer == ANSWER_JSON


def test_an_invalid_model_reply_is_a_502_and_a_defect_is_not_a_422(tmp_path, monkeypatch):
    from crossvenue_rag.generation.schemas import InvalidResponse

    def invalid(*_args, **_kwargs):
        raise InvalidResponse("generator response does not match the schema: x")

    monkeypatch.setattr("crossvenue_rag.service.app.answer_question", invalid)
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        assert c.post("/answer", json={"question": "q"}).status_code == 502

    def defect(*_args, **_kwargs):
        raise ValueError("a bug inside answering")

    monkeypatch.setattr("crossvenue_rag.service.app.answer_question", defect)
    respond = lambda url, body: openrouter_payload(ASK_ANSWER)  # noqa: E731
    with client(tmp_path, respond, raise_server_exceptions=False) as c:
        assert c.post("/answer", json={"question": "q"}).status_code == 500


def test_a_field_the_request_does_not_have_is_422_not_a_default(tmp_path):
    with client(tmp_path, lambda url, body: openrouter_payload(ASK_ANSWER)) as c:
        response = c.post("/answer", json={"question": "binance ban?", "generatr": "qwen"})
    assert response.status_code == 422


def test_concurrent_requests_use_the_model_client_one_at_a_time(tmp_path, monkeypatch):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    inside, most = 0, 0
    guard = threading.Lock()

    def slow(*_args, **_kwargs):
        nonlocal inside, most
        with guard:
            inside += 1
            most = max(most, inside)
        time.sleep(0.05)
        with guard:
            inside -= 1
        raise ValueError("stop here")

    monkeypatch.setattr("crossvenue_rag.service.app.answer_question", slow)
    respond = lambda url, body: openrouter_payload(ASK_ANSWER)  # noqa: E731
    with (
        client(tmp_path, respond, raise_server_exceptions=False) as c,
        ThreadPoolExecutor(4) as pool,
    ):
        codes = list(pool.map(lambda _: c.post("/answer", json={"question": "q"}), range(4)))
    assert [r.status_code for r in codes] == [500] * 4
    assert most == 1
