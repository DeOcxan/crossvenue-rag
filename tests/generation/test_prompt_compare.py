"""Tests for `generation.prompt_compare`: H-v2 and the side-effect checks."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.prompt_compare import (
    compare_prompts,
    side_effects,
)
from tests.conftest import assert_round_trip, committed_of_kind
from tests.generation.factories import prompt_record


def test_wrong_refusals_counted_only_where_no_refusal_is_expected_and_holm_across_configs():
    v1 = [
        prompt_record("a", "q1", "answer", refused=True),
        prompt_record("a", "q2", "answer", refused=True),
        prompt_record("a", "q3", "refuse", refused=True),
        prompt_record("b", "q1", "answer", vae=0.0),
    ]
    v2 = [
        prompt_record("a", "q1", "answer", vae=0.0),
        prompt_record("a", "q2", "answer", refused=True),
        prompt_record("a", "q3", "refuse", vae=0.0),
        prompt_record("b", "q1", "answer", vae=0.5),
    ]
    rows = {(r.config, r.metric): r for r in compare_prompts(v1, v2, ["a", "b"], 200, 1)}
    wrong = rows[("a", "wrong_refusal")]
    assert (wrong.n, wrong.v1, wrong.v2) == (2, 1.0, 0.5)  # q3 expects a refusal
    assert wrong.better == 1 and wrong.worse == 0 and wrong.test == "exact McNemar"
    assert rows[("b", "venue_attribution_error")].diff == 0.5
    assert rows[("b", "venue_attribution_error")].worse == 1
    assert all(r.holm_p is not None and r.holm_p >= r.p for r in rows.values())
    assert not any(r.metric == "faithfulness" for r in rows.values())


def test_side_effects_find_claims_citing_only_another_venue():
    venue_of = {"c-bin": "binance", "c-byb": "bybit"}
    borrowed = {"text": "bybit allows 60000", "venue": "bybit", "chunks": [1]}  # cites binance
    own = {"text": "binance allows 60000", "venue": "binance", "chunks": [1]}
    records = [
        prompt_record("a", "q1", "answer", partial=True, claims=[own, borrowed], missing=["bybit"]),
        prompt_record("a", "q2", "answer", claims=[own]),  # full answer: not checked
        prompt_record("a", "q3", "answer", partial=True, claims=[], missing=["bybit"]),  # a refusal
    ]
    [effect] = side_effects(records, venue_of)
    assert effect.answer_id == "a:q1"
    assert effect.claims_on_missing_venue == [2]
    assert effect.claims_citing_only_other_venues == [2]


def test_prompt_comparisons_count_a_decrease_as_better():
    v1 = [
        prompt_record("a", "q1", "answer", vae=0.5),
        prompt_record("a", "q2", "answer", refused=True),
    ]
    v2 = [prompt_record("a", "q1", "answer", vae=0.0), prompt_record("a", "q2", "answer", vae=0.0)]
    rows = {r.metric: r for r in compare_prompts(v1, v2, ["a"], 200, 1)}
    assert not rows["wrong_refusal"].higher_is_better
    assert (rows["wrong_refusal"].better, rows["wrong_refusal"].worse) == (1, 0)
    assert not rows["venue_attribution_error"].higher_is_better
    assert (rows["venue_attribution_error"].better, rows["venue_attribution_error"].worse) == (1, 0)


def test_a_configuration_with_no_shared_answer_is_refused_by_name():
    v1 = [prompt_record("a", "q1", "answer", vae=0.0), prompt_record("b", "q1", "answer", vae=0.0)]
    v2 = [prompt_record("a", "q1", "answer", vae=0.0)]  # a partial run: b never answered
    with pytest.raises(UserError, match="b: v1 and v2 share no answered question"):
        compare_prompts(v1, v2, ["a", "b"], 100, 0)


def test_a_configuration_without_claims_in_both_has_no_attribution_row():
    v1 = [
        prompt_record("a", "q1", "answer", vae=0.0),
        prompt_record("b", "q1", "answer", refused=True),
    ]
    v2 = [prompt_record("a", "q1", "answer", vae=0.5), prompt_record("b", "q1", "answer", vae=0.0)]
    rows = compare_prompts(v1, v2, ["a", "b"], 100, 0)
    assert [(r.config, r.metric) for r in rows] == [
        ("a", "wrong_refusal"),
        ("a", "venue_attribution_error"),
        ("b", "wrong_refusal"),
    ]


@pytest.mark.parametrize(
    "run", committed_of_kind("comparison.csv", "_prompts"), ids=lambda p: p.name
)
def test_every_committed_comparison_table_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.generation.prompt_compare import read_comparison

    assert_round_trip(run, "comparison.csv", read_comparison(run), tmp_path)


def test_a_partial_answer_citing_a_chunk_of_no_known_venue_is_a_value_error():
    own = {"text": "binance allows 60000", "venue": "binance", "chunks": [1]}
    records = [prompt_record("a", "q1", "answer", partial=True, claims=[own], missing=["bybit"])]
    with pytest.raises(ValueError, match="a:q1 cites c-bin, which venue_of lacks"):
        side_effects(records, {"c-byb": "bybit"})
