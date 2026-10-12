"""Tests for `crossvenue_rag.evaluation.metrics`."""

from __future__ import annotations

import pytest

from crossvenue_rag.evaluation.metrics import (
    evaluate,
    ndcg_at_k,
    off_venue_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    venue_coverage_at_k,
)

RETRIEVED = ["c1", "c2", "c3", "c4", "c5"]


GOLD = ["c2", "c5", "c9"]


def test_binary_relevance_metrics_match_a_hand_checked_ranking():
    scores = evaluate(RETRIEVED, GOLD, [1, 5])
    assert scores["recall@1"] == 0.0
    assert scores["recall@5"] == pytest.approx(2 / 3)
    assert scores["mrr"] == pytest.approx(0.5)  # first hit is c2, at rank 2
    assert set(scores) == {
        "mrr",
        "recall@1",
        "precision@1",
        "ndcg@1",
        "recall@5",
        "precision@5",
        "ndcg@5",
    }

    assert ndcg_at_k(["g1", "g2", "x"], ["g1", "g2"], 3) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["x", "y", "g1"], ["g1"], 3) < 1.0
    # Five gold chunks and k=2 must still allow nDCG@2 == 1.0.
    assert ndcg_at_k(["g1", "g2"], ["g1", "g2", "g3", "g4", "g5"], 2) == pytest.approx(1.0)


def test_precision_denominator_is_k_not_the_result_count():
    """Returning three results where ten were asked for is not precision 1.0."""
    assert precision_at_k(["g1"], ["g1"], 10) == pytest.approx(0.1)
    assert precision_at_k(["g1"], ["g1"], 1) == 1.0


def test_duplicate_results_do_not_inflate_scores():
    assert recall_at_k(["g1", "g1", "g1"], ["g1", "g2"], 3) == pytest.approx(0.5)
    assert precision_at_k(["g1", "g1"], ["g1"], 2) == pytest.approx(0.5)


def test_metrics_undefined_without_gold_raise():
    """Scoring an unlabeled question as 0.0 would raise or lower methods' means at random."""
    for fn in (recall_at_k, ndcg_at_k):
        with pytest.raises(ValueError, match="empty gold set"):
            fn(["a"], [], 5)


def test_venue_coverage_needs_every_named_venue_in_the_top_k():
    assert venue_coverage_at_k(["binance", "bybit", "ccxt"], ["bybit", "binance"], 2) == 1.0
    assert venue_coverage_at_k(["binance", "ccxt", "bybit"], ["bybit", "binance"], 2) == 0.0
    with pytest.raises(ValueError, match="naming 2 or more venues"):
        venue_coverage_at_k(["binance"], ["binance"], 1)


def test_condensed_ndcg_drops_unjudged_documents_before_scoring():
    from math import log2

    from crossvenue_rag.evaluation.metrics import ndcg_at_k, ndcg_condensed_at_k

    retrieved = ["a", "x", "g"]  # x was never judged
    gold = ["g"]
    judged = {"a", "g"}
    # Plain nDCG: g at rank 3, x counted as irrelevant.
    assert ndcg_at_k(retrieved, gold, 3) == pytest.approx(1 / log2(4))
    # Condensed list [a, g]: g moves up to rank 2.
    assert ndcg_condensed_at_k(retrieved, gold, judged, 3) == pytest.approx(1 / log2(3))
    # Fully judged: identical to nDCG.
    assert ndcg_condensed_at_k(retrieved, gold, {"a", "x", "g"}, 3) == ndcg_at_k(retrieved, gold, 3)
    # A gold document one rank below the cut-off enters it once the unjudged one
    # above it is dropped: rank 4 of the list, rank 3 of the condensed list.
    assert ndcg_condensed_at_k(["a", "x", "b", "g"], gold, {"a", "b", "g"}, 3) == pytest.approx(
        1 / log2(4)
    )


def test_condensed_ndcg_keeps_a_gold_document_the_judgments_do_not_cover():
    from crossvenue_rag.evaluation.metrics import ndcg_condensed_at_k

    # g is gold but absent from the judged set: it stays in the condensed list at rank 1.
    assert ndcg_condensed_at_k(["g", "a"], ["g"], {"a"}, 2) == pytest.approx(1.0)


def test_off_venue_share_counts_unnamed_venues_and_keeps_ccxt_apart():
    venues = ["bybit", "binance", "ccxt", "okx", "binance"]
    off, via = off_venue_at_k(venues, ["bybit", "binance"], 5)
    assert (off, via) == (pytest.approx(1 / 5), pytest.approx(1 / 5))
    # A question that names ccxt: its chunks are on-venue.
    assert off_venue_at_k(venues, ["bybit", "binance", "ccxt"], 5) == (pytest.approx(1 / 5), 0.0)


def test_off_venue_at_1_is_the_top1_rate():
    assert off_venue_at_k(["okx", "bybit"], ["bybit"], 1) == (1.0, 0.0)
    assert off_venue_at_k(["ccxt", "okx"], ["bybit"], 1) == (0.0, 1.0)
    assert off_venue_at_k(["bybit", "okx"], ["bybit"], 1) == (0.0, 0.0)


def test_off_venue_share_is_over_the_chunks_retrieved():
    assert off_venue_at_k(["okx", "bybit"], ["bybit"], 10) == (0.5, 0.0)
    assert off_venue_at_k([], ["bybit"], 10) == (0.0, 0.0)


def test_off_venue_needs_a_named_venue():
    with pytest.raises(ValueError, match="naming a venue"):
        off_venue_at_k(["bybit"], [], 10)


@pytest.mark.parametrize(
    "metric",
    [
        lambda: recall_at_k(["a"], [], 1),
        lambda: precision_at_k(["a"], [], 1),
        lambda: ndcg_at_k(["a"], [], 1),
        lambda: reciprocal_rank(["a"], []),
    ],
)
def test_every_ranking_metric_is_undefined_on_an_empty_gold_set(metric):
    with pytest.raises(ValueError, match="empty gold set"):
        metric()
