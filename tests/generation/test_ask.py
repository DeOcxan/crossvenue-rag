"""Tests for `generation.ask`: one ad-hoc question, offline."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.ask import answer_question
from tests.generation.offline import (
    ASK_ANSWER,
    ask_setup,
)


def test_ask_returns_claims_with_citations_resolved_to_the_matching_occurrence(tmp_path):
    cfg, llm, retriever, transport = ask_setup(tmp_path)
    result = answer_question(
        cfg,
        retriever,
        llm,
        "binance testnet ip ban length?",
        "venue_split",
        generator_name="gemini",
    )
    assert retriever.calls == [("venue_split", cfg.generation.top_k)]
    assert result.venues_detected == ["binance"]
    assert result.refused is False and result.asked_clarification is False
    [claim] = result.claims
    [citation] = claim.citations
    assert citation.doc_path == "testnet/rest-api.md" and citation.url == "https://t"
    assert citation.environment == "testnet"
    assert transport.calls[0][2]["model"] == "google/gemini-3.8-flash"


def test_ask_rejects_an_unknown_generator_and_the_oracle(tmp_path):
    cfg, llm, retriever, _ = ask_setup(tmp_path)
    with pytest.raises(UserError, match="--generator 'nope' is not a generator"):
        answer_question(cfg, retriever, llm, "q", "venue_split", "nope")
    with pytest.raises(UserError, match="oracle"):
        answer_question(cfg, retriever, llm, "q", "venue_split_oracle", generator_name="gemini")
    with pytest.raises(UserError, match="empty"):
        answer_question(
            cfg, retriever, llm, "   ", retriever_name="venue_split", generator_name="gemini"
        )


def test_a_partial_answer_without_claims_is_a_refusal_with_its_missing_venues(tmp_path):
    missing = [{"venue": "bybit", "missing": "the ban length"}]
    partial = {**ASK_ANSWER, "claims": [], "partial": True, "missing": missing}
    cfg, llm, retriever, _ = ask_setup(tmp_path, partial)
    result = answer_question(
        cfg,
        retriever,
        llm,
        "binance and bybit ban length?",
        retriever_name="venue_split",
        generator_name="gemini",
    )
    assert result.outcome == "refused" and result.refused is True
    assert result.partial is True and [m.model_dump() for m in result.missing] == missing


def test_a_partial_answer_with_claims_names_what_is_missing(tmp_path):
    partial = {**ASK_ANSWER, "partial": True, "missing": [{"venue": "bybit", "missing": "bans"}]}
    cfg, llm, retriever, _ = ask_setup(tmp_path, partial)
    result = answer_question(
        cfg,
        retriever,
        llm,
        "binance and bybit ban length?",
        retriever_name="venue_split",
        generator_name="gemini",
    )
    assert result.outcome == "partial" and result.refused is False


def test_a_citation_carries_its_chunks_venue_not_the_claims(tmp_path):
    misattributed = {**ASK_ANSWER, "claims": [{"text": "bans", "venue": "bybit", "chunks": [1]}]}
    cfg, llm, retriever, _ = ask_setup(tmp_path, misattributed)
    result = answer_question(
        cfg,
        retriever,
        llm,
        "binance ban length?",
        retriever_name="venue_split",
        generator_name="gemini",
    )
    claim = result.claims[0]
    assert claim.venue == "bybit" and claim.citations[0].venue == "binance"


def test_ask_checks_the_options_before_it_opens_the_retriever(tmp_path):
    from crossvenue_rag.generation import ask as ask_module

    cfg, _, _, _ = ask_setup(tmp_path)

    def unopenable(_cfg):
        raise AssertionError("the retriever was opened")

    with pytest.raises(UserError, match="--generator 'nope' is not a generator"):
        ask_module.ask(cfg, "q", "venue_split", "nope", open_retriever=unopenable)


def test_ask_answers_on_the_current_index_and_closes_the_retriever(tmp_path):
    from crossvenue_rag.generation import ask as ask_module
    from tests.generation.offline import DetectingRetriever

    cfg, llm, _, _ = ask_setup(tmp_path)
    events: list[str] = []

    class OpenedRetriever(DetectingRetriever):
        def __init__(self, _cfg):
            super().__init__()

        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, *exc):
            events.append("close")

        def require_current_index(self):
            events.append("index checked")
            return "v"

    result = ask_module.ask(
        cfg,
        "binance testnet ip ban length?",
        open_retriever=OpenedRetriever,
        make_llm=lambda _gen: llm,
        retriever_name="venue_split",
        generator_name="gemini",
    )
    assert result.claims[0].citations[0].doc_path == "testnet/rest-api.md"
    assert events == ["open", "index checked", "close"]
