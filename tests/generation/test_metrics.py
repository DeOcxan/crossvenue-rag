"""Tests for `generation.metrics`: the answer metrics a judgment gives."""

from __future__ import annotations

from crossvenue_rag.generation.metrics import answer_metrics
from tests.generation.factories import judgement, verdict


def test_answer_metrics_follow_their_definitions():
    j = judgement(
        [
            verdict(1, supported=True, venue_ok=True),
            verdict(2, supported=True, venue_ok=False),
            verdict(3, supported=False, venue_ok=True),
            verdict(4, supported=False, venue_ok=False),
        ],
        [
            {"venue": "binance", "gold_facts": 2, "covered_facts": 2, "rationale": ""},
            {"venue": "bybit", "gold_facts": 2, "covered_facts": 0, "rationale": ""},
        ],
    )
    m = answer_metrics(j)
    assert m.faithfulness == 0.5
    assert m.venue_attribution_error == 0.5
    assert m.coverage == 0.5  # one venue fully covered, the other not at all
    assert m.behaviour_correct == 1.0


def test_a_refusal_has_no_faithfulness_and_no_gold_means_no_coverage():
    m = answer_metrics(judgement([], [], behaviour=False))
    assert m.faithfulness is None and m.venue_attribution_error is None
    assert m.coverage is None and m.behaviour_correct == 0.0


def test_a_name_that_is_no_answer_metric_is_a_value_error():
    import pytest

    with pytest.raises(ValueError, match="'recall' is not an answer metric"):
        answer_metrics(judgement([], [])).value("recall")
