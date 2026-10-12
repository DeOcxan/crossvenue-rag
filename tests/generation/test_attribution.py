"""Tests for `generation.attribution`: the kind of a misattributed claim."""

from __future__ import annotations

from crossvenue_rag.generation.attribution import (
    attribution_kind,
    attribution_summary,
    documented_exchange,
)
from tests.generation.factories import (
    answer_json,
    generation_json,
    judge_json,
    judgement_json,
    record,
)

VENUES = {"binance", "bybit", "bingx", "ccxt"}


def test_kinds_follow_the_cited_chunks_and_the_named_venue():
    assert attribution_kind("bybit", {"binance"}, VENUES, documented=()) == "cross_exchange"
    assert attribution_kind("BingX", {"ccxt"}, VENUES, {"bingx"}) == "wrapper"
    # Cites a chunk of the venue it names: the judge's verdict has another reason.
    assert attribution_kind("binance", {"binance", "bybit"}, VENUES, documented=()) == "other"
    # Names no venue of the corpus.
    assert attribution_kind("General", {"binance"}, VENUES, documented=()) == "other"
    # Names the wrapper, a library and no exchange.
    assert attribution_kind("ccxt", {"binance"}, VENUES, documented=()) == "other"


def test_a_ccxt_page_about_another_exchange_is_a_cross_exchange_error():
    assert attribution_kind("bingx", {"ccxt"}, VENUES, {"bybit"}) == "cross_exchange"
    # A CCXT page about no one exchange: not the wrapper kind either.
    assert attribution_kind("bingx", {"ccxt"}, VENUES, {None}) == "other"
    assert documented_exchange("ccxt", "wiki/exchanges/bybit.md") == "bybit"
    assert documented_exchange("ccxt", "wiki/Manual.md") is None
    assert documented_exchange("binance", "wiki/exchanges/bybit.md") is None


def _record(config, claims, verdicts):
    answer = {**answer_json(), "claims": claims}
    judgement = judgement_json(claims=0)
    judgement["claims"] = [
        {"index": i, "supported": True, "venue_correct": v, "rationale": ""}
        for i, v in enumerate(verdicts, start=1)
    ]
    generation = generation_json(answer=answer, chunk_ids=("c-bin", "c-ccxt"))
    return record(config, generation=generation, judge=judge_json(judgement=judgement))


def test_summary_counts_kinds_and_averages_shares_over_answers_with_claims():
    venue_of = {"c-bin": "binance", "c-ccxt": "ccxt", "c-byb": "bybit", "c-bgx": "bingx"}
    doc_path_of = {c: f"{v}/doc.md" for c, v in venue_of.items()}
    doc_path_of["c-ccxt"] = "wiki/exchanges/bingx.md"
    right = {"text": "t", "venue": "binance", "chunks": [1]}
    wrapped = {"text": "t", "venue": "bingx", "chunks": [2]}
    crossed = {"text": "t", "venue": "bybit", "chunks": [1]}
    records = [
        _record("a", [right, wrapped], [True, False]),
        _record("a", [crossed], [False]),
        _record("a", [], []),  # no claims: not an answer the metric applies to
    ]
    summary = attribution_summary(records, venue_of, doc_path_of, ["a", "b"])
    a = summary["a"]
    assert (a.answers, a.claims) == (2, 3)
    assert a.misattributed == {"cross_exchange": 1, "wrapper": 1, "other": 0}
    assert a.error == {"cross_exchange": 0.5, "wrapper": 0.25, "other": 0.0}
    assert a.total_error == 0.75
    # A configuration with no answer that has claims: the error does not apply, never 0.
    assert summary["b"].error == {"cross_exchange": None, "wrapper": None, "other": None}
    assert summary["b"].total_error is None


def test_a_misattributed_claim_on_a_chunk_of_no_known_venue_or_page_is_a_value_error():
    import pytest

    wrapped = {"text": "t", "venue": "bingx", "chunks": [2]}
    records = [_record("a", [wrapped], [False])]
    venue_of = {"c-bin": "binance", "c-ccxt": "ccxt"}
    for venues, pages in (({"c-bin": "binance"}, {}), (venue_of, {"c-bin": "binance/doc.md"})):
        with pytest.raises(ValueError, match="cites c-ccxt, of no known venue or page"):
            attribution_summary(records, venues, pages, ["a"])
