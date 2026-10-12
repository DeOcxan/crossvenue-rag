"""Tests for `generation.grader_agreement`: two graders compared per dimension."""

from __future__ import annotations

import pytest

from crossvenue_rag.generation.grader_agreement import (
    agreement_from_pairs,
    agreement_intervals,
    compare_gradings,
    paired_verdicts,
)
from crossvenue_rag.generation.schemas import Judgement
from tests.generation.factories import judgement, verdict


def test_each_dimension_is_compared_over_the_answers_both_gradings_share():
    first = {
        "a": judgement(
            [verdict(1, supported=True, venue_ok=True), verdict(2, supported=False, venue_ok=True)],
            [{"venue": "x", "gold_facts": 4, "covered_facts": 1, "rationale": ""}],
        ),
        "b": judgement(
            [verdict(1, supported=True, venue_ok=False)],
            [{"venue": "x", "gold_facts": 4, "covered_facts": 4, "rationale": ""}],
            behaviour=False,
        ),
        "c": judgement([], [{"venue": "x", "gold_facts": 2, "covered_facts": 1, "rationale": ""}]),
    }
    second = {
        "a": judgement(
            [verdict(1, supported=True, venue_ok=True), verdict(2, supported=True, venue_ok=True)],
            [{"venue": "x", "gold_facts": 4, "covered_facts": 1, "rationale": ""}],
        ),
        "b": judgement(
            [verdict(1, supported=True, venue_ok=False)],
            [{"venue": "x", "gold_facts": 4, "covered_facts": 3, "rationale": ""}],
            behaviour=False,
        ),
        "c": judgement([], [{"venue": "x", "gold_facts": 2, "covered_facts": 1, "rationale": ""}]),
    }
    # coverage: first 0.25, 1.0, 0.5; second 0.25, 0.75, 0.5: same order
    rows = {r.dimension: r for r in compare_gradings(first, second)}
    assert rows["supported"].n == 3 and rows["supported"].agreement == pytest.approx(2 / 3)
    assert rows["venue_correct"].agreement == 1.0 and rows["venue_correct"].kappa == 1.0
    assert rows["behaviour_correct"].n == 3
    assert rows["coverage"].n == 3 and rows["coverage"].spearman == pytest.approx(1.0)


def _grading(supported, *, behaviour=True, covered=1):
    """A grading whose claims are all attributed to the right venue."""
    return Judgement.model_validate(
        {
            "claims": [
                {"index": i, "supported": s, "venue_correct": True}
                for i, s in enumerate(supported, start=1)
            ],
            "coverage": [{"venue": "binance", "gold_facts": 2, "covered_facts": covered}],
            "behaviour_correct": behaviour,
        }
    )


def test_intervals_resample_answers_and_count_undefined_kappa():

    first = {f"a{i}": _grading([True, i % 3 == 0], covered=i % 3) for i in range(12)}
    second = {f"a{i}": _grading([True, i % 4 == 0], covered=i % 2) for i in range(12)}
    pairs = paired_verdicts(first, second)
    assert len(pairs) == 12 and len(pairs[0].supported) == 2
    ci = agreement_intervals(pairs, resamples=500, seed=1)
    low, high = ci["supported"]["agreement"].low, ci["supported"]["agreement"].high
    assert low is not None and high is not None
    assert 0.0 <= low <= high <= 1.0
    # Every venue verdict is True for both graders: kappa is undefined in every resample.
    assert ci["venue_correct"]["kappa"].undefined == 500
    assert ci["venue_correct"]["kappa"].low is None
    assert ci["coverage"]["spearman"].low is not None


def test_an_excluded_answer_leaves_only_behaviour_correct():

    first = {"a": _grading([True], behaviour=True), "b": _grading([False], behaviour=True)}
    second = {"a": _grading([True], behaviour=False), "b": _grading([False], behaviour=True)}
    dims = {d.dimension: d for d in agreement_from_pairs(paired_verdicts(first, second, ["a"]))}
    assert dims["behaviour_correct"].n == 1 and dims["behaviour_correct"].agreement == 1.0
    assert dims["supported"].n == 2  # its claims still count


def test_a_dimension_one_category_throughout_has_no_kappa_and_no_interval():
    from crossvenue_rag.generation.calibration import agreement_record

    first = {f"a{i}": _grading([True], covered=i % 3) for i in range(12)}
    second = {f"a{i}": _grading([True], covered=i % 2) for i in range(12)}
    record = agreement_record(first, second, resamples=50, seed=1)
    supported = record["supported"]
    assert supported.agreement == 1.0
    assert supported.kappa is None and supported.kappa_ci is None


def test_an_excluded_answer_leaves_behaviour_correct_but_not_the_claims():
    first = {"a": _grading([True], behaviour=True), "b": _grading([True], behaviour=False)}
    second = {"a": _grading([True], behaviour=False), "b": _grading([True], behaviour=False)}
    rows = {r.dimension: r for r in compare_gradings(first, second, exclude_behaviour=["a"])}
    assert rows["behaviour_correct"].n == 1 and rows["behaviour_correct"].agreement == 1.0
    assert rows["supported"].n == 2


def test_dimension_verdicts_pair_claims_by_index_and_refuse_a_non_binary_dimension():
    from crossvenue_rag.generation.grader_agreement import dimension_verdicts

    first = {"a": judgement([verdict(1, supported=True, venue_ok=False)], [])}
    second = {
        "a": judgement(
            [verdict(2, supported=True, venue_ok=True), verdict(1, supported=False, venue_ok=True)],
            [],
            behaviour=False,
        )
    }
    pairs = paired_verdicts(first, second)
    assert dimension_verdicts(pairs, "supported") == [(True, False)]
    assert dimension_verdicts(pairs, "venue_correct") == [(False, True)]
    assert dimension_verdicts(pairs, "behaviour_correct") == [(True, False)]
    with pytest.raises(ValueError, match="not a binary dimension"):
        dimension_verdicts(pairs, "coverage")
