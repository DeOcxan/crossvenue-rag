"""Tests for `crossvenue_rag.metric_direction`."""

from __future__ import annotations

import pytest

from crossvenue_rag.metric_direction import HIGHER_IS_BETTER, higher_is_better


@pytest.mark.parametrize(
    ("metric", "up"),
    [
        ("recall@10", True),
        ("ndcg_prime@10", True),
        ("venue_coverage@10", True),
        ("mrr", True),
        ("unjudged@10", False),
        ("off_venue@10", False),
        ("off_venue@1", False),
        ("via_ccxt@10", False),
    ],
)
def test_every_metric_has_a_direction(metric: str, *, up: bool):
    assert higher_is_better(metric) is up


def test_a_metric_without_a_direction_is_refused():
    with pytest.raises(ValueError, match="no direction recorded for metric 'latency@10'"):
        higher_is_better("latency@10")


def test_every_reported_metric_family_has_a_direction():
    from crossvenue_rag.evaluation.evaluator import (
        judged_metric_names_for,
        metric_names_for,
        off_venue_metric_names_for,
        venue_metric_names_for,
    )

    k = [1, 5, 10]
    names = [
        *metric_names_for(k),
        *venue_metric_names_for(k),
        *judged_metric_names_for(k),
        *off_venue_metric_names_for(k),
    ]
    assert {n.split("@")[0] for n in names} <= set(HIGHER_IS_BETTER)


def test_every_answer_metric_has_a_direction():
    from crossvenue_rag.generation.metrics import METRICS

    assert {m: higher_is_better(m) for m in (*METRICS, "wrong_refusal")} == {
        "faithfulness": True,  # share of claims supported
        "venue_attribution_error": False,
        "coverage": True,
        "behaviour_correct": True,
        "wrong_refusal": False,
    }
    with pytest.raises(ValueError, match="no direction recorded"):
        higher_is_better("latency")
