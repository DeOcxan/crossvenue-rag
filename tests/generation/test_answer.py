"""Tests for `generation.answer`: generating and judging one answer."""

from __future__ import annotations

from crossvenue_rag.generation.llm import LLM
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    openrouter_payload,
    respond,
)


def test_the_judge_sees_the_models_clarification_flag_on_a_refusal():
    """The outcome rule changes `refused` only; an answer that both refused and
    asked keeps `asked_clarification: true` in the judge's message.
    """
    from crossvenue_rag.generation.answer import judge_message
    from crossvenue_rag.generation.schemas import GeneratedAnswer
    from crossvenue_rag.models import Question

    both = GeneratedAnswer(answer="Which venue?", refused=True, asked_clarification=True)
    q = Question("cv-0001", "swap fail error?", "gold", [], "fact")
    assert "refused: true, asked_clarification: true" in judge_message(q, both, {}, {})


def test_generate_records_the_answer_its_citations_and_what_produced_it(tmp_path):
    import hashlib
    from pathlib import Path

    from crossvenue_rag.generation.answer import generate
    from tests.generation.offline import (
        FakeTransport,
        config,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    results = TwoChunkRetriever().retrieve(QUESTIONS[0].question, "venue_split", k=2)
    generated = generate(
        llm, cfg.generation, cfg.generation.generators["gemini"], QUESTIONS[0], results
    )
    assert generated.chunk_ids == ["b1", "y1"]
    assert generated.answer.claims[0].venue == "binance"
    [[citation]] = generated.citations
    assert citation.chunk_id == "b1" and citation.venue == CHUNKS["b1"].venue
    prompt = Path("prompts/generator.md").read_bytes()
    assert generated.prompt_sha256 == hashlib.sha256(prompt).hexdigest()
    assert generated.answer_schema_version == 2
    assert {"partial", "missing"} <= set(generated.to_json()["answer"])


def test_judge_derives_the_metrics_from_its_verdicts(tmp_path):
    from crossvenue_rag.generation.answer import Submission, generate, judge_answer
    from tests.generation.offline import (
        FakeTransport,
        config,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    question = QUESTIONS[0]
    results = TwoChunkRetriever().retrieve(question.question, "venue_split", k=2)
    generated = generate(
        llm, cfg.generation, cfg.generation.generators["gemini"], question, results
    )
    graded = judge_answer(llm, cfg.generation, Submission(question, generated, CHUNKS), "sonnet")
    assert graded.judge == "sonnet" and graded.replicate == 0
    assert graded.metrics.model_dump() == {
        "faithfulness": 1.0,
        "venue_attribution_error": 0.0,
        "coverage": 1.0,
        "behaviour_correct": 1.0,
    }


def test_a_new_record_saves_every_field_of_its_schema_even_one_left_to_its_default(tmp_path):
    """A schema v1 generator response and a judge response without rationales keep every field.

    Every field of the v1 answer and of the judgment is saved, and the v1 answer has no
    `partial`, `missing` or `answer_schema_version`, as every committed v1 run's records.
    """
    from pathlib import Path

    from crossvenue_rag.generation.answer import Submission, generate, judge_answer
    from crossvenue_rag.models import Question
    from tests.generation.offline import (
        FakeTransport,
        config,
    )

    def terse(url, body):
        if body["model"].startswith("anthropic/"):
            verdict = {"index": 1, "supported": True, "venue_correct": True}
            return openrouter_payload({"claims": [verdict], "behaviour_correct": True})
        claim = {"text": "max 60000", "venue": "binance", "chunks": [1]}
        return openrouter_payload(
            {"answer": "a", "claims": [claim], "refused": False, "asked_clarification": False}
        )

    cfg = config(tmp_path)
    cfg.generation.generator_prompt = Path("prompts/generator.v1.md")
    cfg.generation.answer_schema_version = 1
    llm = LLM(cfg.generation, FakeTransport(respond=terse), api_key=SECRET)
    # No gold chunks: the terse judgment needs no coverage entry.
    question = Question("cv-0003", "binance recvWindow max?", "gold", ["binance"], "fact")
    results = TwoChunkRetriever().retrieve(question.question, "venue_split", k=2)
    generated = generate(
        llm, cfg.generation, cfg.generation.generators["gemini"], question, results
    )
    saved = generated.to_json()
    assert list(saved) == [
        "answer", "citations", "chunk_ids", "prompt_sha256", "call"
    ]  # fmt: skip
    assert list(saved["answer"]) == [
        "answer", "claims", "refused", "asked_clarification"
    ]  # fmt: skip
    assert generated.answer.stated_partial() is None
    assert list(saved["call"]) == [
        "requested_model", "actual_model", "provider", "input_tokens", "output_tokens",
        "reasoning_tokens", "cost_usd", "latency_ms", "retries", "invalid_responses",
        "cache_key", "params", "cached",
    ]  # fmt: skip
    graded = judge_answer(llm, cfg.generation, Submission(question, generated, CHUNKS), "sonnet")
    saved = graded.to_json()
    assert list(saved) == ["judge", "judgement", "metrics", "prompt_sha256", "replicate", "call"]
    assert list(saved["judgement"]) == [
        "claims", "coverage", "behaviour_correct", "behaviour_rationale"
    ]  # fmt: skip
    assert saved["judgement"]["claims"][0]["rationale"] == ""


def test_temperature_is_not_sent_where_the_endpoint_rejects_it(tmp_path):
    from crossvenue_rag.config_generation import ModelConfig
    from crossvenue_rag.generation.answer import generator_params

    cfg = config(tmp_path).generation
    no_temp = ModelConfig(model="m/x", temperature_supported=False)
    assert generator_params(cfg, no_temp).temperature is None
    assert generator_params(cfg, cfg.generators["qwen"]).temperature == 0.0


def test_a_chunk_the_judge_cannot_be_shown_is_a_value_error_naming_it(tmp_path):
    import pytest

    from crossvenue_rag.generation.answer import Submission, generate, judge_message, judge_request
    from crossvenue_rag.generation.schemas import Claim, GeneratedAnswer
    from tests.generation.offline import FakeTransport

    cited = GeneratedAnswer(
        answer="60000", claims=[Claim(text="60000", venue="binance", chunks=[3])]
    )
    with pytest.raises(ValueError, match="cites chunk 3"):
        judge_message(QUESTIONS[0], cited, {}, {})
    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    question = QUESTIONS[0]
    results = TwoChunkRetriever().retrieve(question.question, "venue_split", k=2)
    generated = generate(
        llm, cfg.generation, cfg.generation.generators["gemini"], question, results
    )
    partial = {"b1": CHUNKS["b1"]}
    with pytest.raises(ValueError, match="generated from y1, which the chunks lack"):
        judge_request(cfg.generation, Submission(question, generated, partial))


def test_the_gold_chunks_are_grouped_by_venue_in_gold_order_without_unknown_ids():
    from crossvenue_rag.generation.answer import gold_by_venue
    from tests.generation.offline import chunk

    chunks = {**CHUNKS, "b2": chunk("b2", "binance", "Binance: weight 1200.")}
    gold = gold_by_venue(["b2", "zz", "y1", "b1"], chunks)
    assert gold == {"binance": [chunks["b2"], chunks["b1"]], "bybit": [chunks["y1"]]}
