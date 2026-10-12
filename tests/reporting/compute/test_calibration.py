"""Tests for `reporting.compute.calibration`, on ratings written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.generation.calibration_file import calibration_json
from crossvenue_rag.reporting.compute import calibration
from crossvenue_rag.reporting.compute.estimates import Estimate
from crossvenue_rag.reporting.run_ids import ANSWERS_V1, CALIBRATION_SAMPLE
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import calibration_of
from tests.generation.factories import (
    answer_json,
    generation_json,
    judge_json,
    judgement_json,
    record_json,
)
from tests.reporting.conftest import write_file, write_jsonl

REPORT = {
    "answers_rated": 40,
    "thresholds": {"supported": 0.6, "venue_correct": 0.6, "behaviour_correct": 0.6},
    "candidates": {"opus": {"passes": ["supported", "venue_correct"]}, "sonnet": {"passes": []}},
    "comparisons": {"opus vs human": {"supported": {"kappa": 0.25}}},
}


def claim(index, *, supported, venue_correct):
    return {"index": index, "supported": supported, "venue_correct": venue_correct}


def graded(claims=(), *, behaviour=True):
    """A judgment or a human rating with `claims` and no coverage."""
    return {**judgement_json(claims=0, behaviour_correct=behaviour), "claims": list(claims)}


def rating_row(answer_id, rating):
    return {"answer_id": answer_id, "rater": "r", "at": "2026-09-25T00:00:00Z", "rating": rating}


def judge_row(answer_id, judgement):
    return {"answer_id": answer_id, "judge": judge_json(judgement=judgement)}


def test_passes_dimensions_and_kappa_are_read_from_calibration_json(tmp_path):
    write_file(
        tmp_path,
        f"results/{ANSWERS_V1}/calibration.json",
        calibration_json(calibration_of(**REPORT)),
    )
    src = Sources(tmp_path)
    assert calibration.passes(src, "opus") == 2
    assert calibration.passed_dimensions(src, "opus") == {"supported", "venue_correct"}
    assert calibration.passes(src, "sonnet") == 0
    assert calibration.dimension_count(src) == 3
    assert calibration.answers_rated(src) == 40
    assert calibration.comparison_kappa(src, "opus vs human", "supported") == 0.25


def ratings(root):
    """The human rates two answers; each judge grades the first, Opus also the second."""
    human = [
        rating_row(
            "a1",
            graded(
                [
                    claim(0, supported=False, venue_correct=True),
                    claim(1, supported=True, venue_correct=False),
                ]
            ),
        ),
        rating_row("a2", graded([claim(0, supported=False, venue_correct=False)])),
    ]
    write_jsonl(root / "evals/annotations/judge_ratings.jsonl", human)
    opus = [
        judge_row(
            "a1",
            graded(
                [
                    claim(0, supported=False, venue_correct=False),
                    claim(1, supported=True, venue_correct=True),
                ]
            ),
        ),
        judge_row("a2", graded([claim(0, supported=True, venue_correct=False)])),
        {"answer_id": "a3", "error": "no judgement"},
    ]
    sonnet = [judge_row("a1", graded([claim(0, supported=True, venue_correct=True)]))]
    write_jsonl(root / f"results/{ANSWERS_V1}/judgements_opus.jsonl", opus)
    write_jsonl(root / f"results/{ANSWERS_V1}/judgements_sonnet.jsonl", sonnet)
    return Sources(root)


def test_verdict_pairs_are_over_the_claims_both_graded(tmp_path):
    src = ratings(tmp_path)
    assert calibration.verdict_pairs(src, "opus", "supported") == {
        (False, False): 1,
        (True, True): 1,
        (True, False): 1,
    }
    assert calibration.verdict_pairs(src, "sonnet", "supported") == {(True, False): 1}
    assert calibration.verdict_pairs(src, "opus", "supported")[(False, False)] == 1
    with pytest.raises(ValueError, match="not a binary dimension"):
        calibration.verdict_pairs(src, "opus", "coverage")


def test_the_human_negatives_count_every_rated_claim(tmp_path):
    src = ratings(tmp_path)
    assert calibration.human_negatives(src) == calibration.HumanClaims(3, 2, 2)


def test_venue_disagreement_counts_either_and_both(tmp_path):
    src = ratings(tmp_path)
    # a1 claim 0: Opus only; a1 claim 1: human only; a2 claim 0: both.
    assert calibration.venue_disagreement(src, "opus") == (3, 1)


FULL_REPORT = {
    "resamples": 1000,
    "seed": 7,
    "candidates": {"opus": {}, "sonnet": {}},
    "thresholds": {"supported": 0.6, "coverage": 0.7},
    "statistic": {"supported": "kappa", "venue_correct": "kappa", "coverage": "spearman"},
    "comparisons": {
        "opus vs human": {
            "supported": {"n": 91, "kappa": 0.85, "kappa_ci": [-0.1, 1.0]},
            "venue_correct": {
                "n": 91,
                "kappa": 0.85,
                "kappa_ci": [-0.1, 1.0],
                "kappa_undefined_resamples": 12,
            },
            "coverage": {"n": 34, "spearman": 0.9, "spearman_ci": [0.78, 0.96], "kappa": 0.1},
        }
    },
}


def test_thresholds_statistics_and_intervals_are_read_from_calibration_json(tmp_path):
    write_file(
        tmp_path,
        f"results/{ANSWERS_V1}/calibration.json",
        calibration_json(calibration_of(**FULL_REPORT)),
    )
    src = Sources(tmp_path)
    assert calibration.judge_candidate_count(src) == 2
    assert calibration.resamples(src) == 1000
    assert calibration.bootstrap_seed(src) == 7
    assert calibration.threshold(src, "coverage") == 0.7
    assert calibration.statistic(src, "coverage") == "spearman"
    assert calibration.compared_count(src, "opus vs human", "coverage") == 34
    # Each dimension's interval is that of its own statistic.
    assert calibration.agreement_interval(src, "opus vs human", "coverage") == Estimate(
        0.9, 0.78, 0.96
    )
    assert calibration.agreement_interval(src, "opus vs human", "supported") == Estimate(
        0.85, -0.1, 1.0
    )
    assert calibration.same_agreement(src, "opus vs human", "supported", "venue_correct")
    assert not calibration.same_agreement(src, "opus vs human", "supported", "coverage")
    assert calibration.kappa_undefined(src, "opus vs human", "venue_correct") == 12


def test_an_interval_with_undefined_resamples_establishes_no_threshold(tmp_path):
    thresholds = {"supported": 0.6, "venue_correct": 0.6, "coverage": 0.7}
    venue_correct = {
        "n": 91,
        "kappa": 0.66,
        "kappa_ci": [0.65, 0.9],
        "kappa_undefined_resamples": 12,
    }
    coverage = {"n": 34, "spearman": 0.9, "spearman_ci": [0.78, 0.96]}
    report = {
        **FULL_REPORT,
        "thresholds": thresholds,
        "candidates": {"opus": {"passes": ["venue_correct", "coverage"]}, "sonnet": {}},
        "comparisons": {"opus vs human": {"venue_correct": venue_correct, "coverage": coverage}},
    }
    write_file(
        tmp_path,
        f"results/{ANSWERS_V1}/calibration.json",
        calibration_json(calibration_of(**report)),
    )
    src = Sources(tmp_path)
    # venue_correct's interval lies above its threshold, but 12 resamples leave kappa undefined.
    assert not calibration.interval_informative(src, "opus vs human", "venue_correct")
    assert calibration.interval_informative(src, "opus vs human", "coverage")
    assert calibration.passes_on_interval(src, "opus") == ("coverage",)
    assert calibration.passes_by_reach(src, "opus") == (
        frozenset({"coverage"}),
        frozenset({"venue_correct"}),
    )


def test_the_behaviour_kappa_is_over_the_answers_both_graded(tmp_path):
    human = [
        rating_row(f"a{i}", graded(behaviour=ok))
        for i, ok in enumerate([True, True, False, False, True])
    ]
    write_jsonl(tmp_path / "evals/annotations/judge_ratings.jsonl", human)
    # The judge agrees on a0 to a3 and never graded a4.
    grades = [
        judge_row(f"a{i}", graded(behaviour=ok)) for i, ok in enumerate([True, True, False, False])
    ]
    write_jsonl(tmp_path / f"results/{ANSWERS_V1}/judgements_opus.jsonl", grades)
    assert calibration.behaviour_kappa(Sources(tmp_path), "opus") == 1.0


def sample_fixture(root, categories):
    """One answer per category in `categories`, all in the sample but the last."""
    records = [
        record_json(
            "gemini@venue_split" if i % 2 else "qwen@venue_split",
            f"cv-{i:04d}",
            category=category,
            generation=generation_json(answer=answer_json(claims=i)),
        )
        for i, category in enumerate(categories)
    ]
    write_jsonl(root / f"results/{ANSWERS_V1}/answers.jsonl", records)
    ids = "".join(f"{records[i]['answer_id']}\n" for i in range(len(categories) - 1))
    write_file(root, CALIBRATION_SAMPLE, f"# Run: results/{ANSWERS_V1}\n" + ids)
    return Sources(root)


def test_the_calibration_sample_is_counted_by_config_and_category(tmp_path):
    src = sample_fixture(tmp_path, ["comparison", "unanswerable", "fact", "noisy", "fact"])
    assert calibration.sample_ids(src) == [
        "qwen@venue_split:cv-0000",
        "gemini@venue_split:cv-0001",
        "qwen@venue_split:cv-0002",
        "gemini@venue_split:cv-0003",
    ]
    assert len(calibration.sample_answers(src)) == 4
    # a0 to a3 hold 0 + 1 + 2 + 3 claims.
    assert calibration.sample_claims(src) == 6
    assert calibration.sample_by(src, "config") == {"qwen@venue_split": 2, "gemini@venue_split": 2}
    assert calibration.sample_by(src, "category")["adversarial"] == 0
    with pytest.raises(ValueError, match="not a text field"):
        calibration.sample_by(src, "judge")
    assert calibration.other_category_size(src) == 1


def test_a_name_that_is_no_field_of_an_answer_record_is_a_value_error(tmp_path):
    src = sample_fixture(tmp_path, ["comparison", "fact"])
    with pytest.raises(ValueError, match="no_such_field is not a text field"):
        calibration.sample_by(src, "no_such_field")


def test_uneven_other_categories_have_no_one_size(tmp_path):
    src = sample_fixture(tmp_path, ["comparison", "fact", "fact", "noisy", "x"])
    assert calibration.other_category_size(src) is None


def test_the_calibration_excludes_exactly_the_configured_questions(tmp_path):
    write_file(
        tmp_path, "configs/default.yaml", "generation:\n  calibration_exclude: [cv-2, cv-1]\n"
    )
    src = Sources(tmp_path)
    assert calibration.excludes_exactly(src, ("cv-1", "cv-2"))
    assert not calibration.excludes_exactly(src, ("cv-1",))


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda s: calibration.passes(s, "haiku"), "candidates has no 'haiku'"),
        (lambda s: calibration.passes_on_interval(s, "haiku"), "candidates has no 'haiku'"),
        (lambda s: calibration.passes_by_reach(s, "haiku"), "candidates has no 'haiku'"),
        (lambda s: calibration.selected_passes_most(s, "haiku"), "candidates has no 'haiku'"),
        (lambda s: calibration.threshold(s, "coverage"), "thresholds has no 'coverage'"),
        (lambda s: calibration.statistic(s, "coverage"), "statistics has no 'coverage'"),
        (
            lambda s: calibration.comparison_kappa(s, "opus vs human", "venue_correct"),
            "opus vs human has no 'venue_correct'",
        ),
        (
            lambda s: calibration.compared_count(s, "sonnet vs human", "supported"),
            "comparisons has no 'sonnet vs human'",
        ),
        (
            lambda s: calibration.kappa_undefined(s, "sonnet vs human", "supported"),
            "comparisons has no 'sonnet vs human'",
        ),
        (
            lambda s: calibration.interval_informative(s, "sonnet vs human", "supported"),
            "comparisons has no 'sonnet vs human'",
        ),
        (
            lambda s: calibration.agreement_interval(s, "sonnet vs human", "supported"),
            "comparisons has no 'sonnet vs human'",
        ),
        (
            lambda s: calibration.agreement_range(s, "sonnet vs human", ["supported"]),
            "comparisons has no 'sonnet vs human'",
        ),
    ],
)
def test_a_candidate_dimension_or_comparison_calibration_json_lacks_is_a_value_error(
    tmp_path, call, message
):
    report = {**REPORT, "selected": "haiku"}
    write_file(
        tmp_path,
        f"results/{ANSWERS_V1}/calibration.json",
        calibration_json(calibration_of(**report)),
    )
    with pytest.raises(ValueError, match=message):
        call(Sources(tmp_path))


def test_a_sample_answer_the_answer_run_lacks_is_a_value_error(tmp_path):
    src = sample_fixture(tmp_path, ["comparison", "fact"])
    write_file(tmp_path, CALIBRATION_SAMPLE, f"# Run: results/{ANSWERS_V1}\nabsent:cv-9999\n")
    with pytest.raises(ValueError, match="answer run has no 'absent:cv-9999'"):
        calibration.sample_by(src, "config")
