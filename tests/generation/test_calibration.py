"""Tests for `generation.calibration`: selecting a judge from its agreements."""

from __future__ import annotations

import pytest

from crossvenue_rag.config_generation import CalibrationDimension
from crossvenue_rag.errors import UserError
from tests.conftest import dimension_of
from tests.generation.factories import (
    rated_in,
    write_judgements,
    write_ratings,
)
from tests.generation.offline import (
    calibration_config,
)


def as_dimensions(dims):
    """Each candidate's dimensions, from partial dicts."""
    return {j: {d: dimension_of(**row) for d, row in v.items()} for j, v in dims.items()}


def test_the_judge_meeting_more_thresholds_is_selected():
    from crossvenue_rag.generation.calibration import select_judge

    thresholds: dict[CalibrationDimension, float] = {
        "supported": 0.61, "venue_correct": 0.61, "behaviour_correct": 0.61, "coverage": 0.70
    }  # fmt: skip
    dims = {
        "a": {"supported": {"kappa": 0.2}, "venue_correct": {"kappa": 0.5},
              "behaviour_correct": {"kappa": 0.8}, "coverage": {"spearman": 0.8}},
        "b": {"supported": {"kappa": 0.3}, "venue_correct": {"kappa": 0.7},
              "behaviour_correct": {"kappa": 0.7}, "coverage": {"spearman": 0.9}},
    }  # fmt: skip
    chosen = select_judge(as_dimensions(dims), thresholds, {"a": 0.01, "b": 0.03})
    assert chosen.selected == "b"
    assert chosen.candidates["a"].fails == ["supported", "venue_correct"]
    assert "more than any other" in chosen.reason


def test_a_tie_goes_to_venue_correct_then_coverage_then_cost():
    from crossvenue_rag.generation.calibration import select_judge

    thresholds: dict[CalibrationDimension, float] = {
        "supported": 0.61, "venue_correct": 0.61, "behaviour_correct": 0.61, "coverage": 0.70
    }  # fmt: skip
    same = {"supported": {"kappa": 0.7}, "venue_correct": {"kappa": 0.7},
            "behaviour_correct": {"kappa": 0.7}, "coverage": {"spearman": 0.8}}  # fmt: skip
    better_venue = {**same, "venue_correct": {"kappa": 0.75}}
    dims = as_dimensions({"a": same, "b": better_venue})
    assert select_judge(dims, thresholds, {"a": 0.01, "b": 0.03}).selected == "b"
    dims = as_dimensions({"a": same, "b": dict(same)})
    # The cheaper judge wins only when everything else ties.
    assert select_judge(dims, thresholds, {"a": 0.03, "b": 0.01}).selected == "b"


def test_select_judge_refuses_a_candidate_that_was_never_sampled(tmp_path, monkeypatch):
    from crossvenue_rag.generation.calibration import calibration_report

    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)  # opus, the other candidate, has no file
    rated_in(monkeypatch, write_ratings(tmp_path / "ratings.jsonl"))
    with pytest.raises(UserError, match="no judgments by opus") as caught:
        calibration_report(calibration_config(tmp_path), run)
    assert "judge-sample --judge opus" in caught.value.hint


def test_an_excluded_answer_the_run_does_not_have_is_refused(tmp_path, monkeypatch):
    from crossvenue_rag.generation.calibration import calibration_report

    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)
    write_judgements(run, "opus", cost=0.03)
    cfg = calibration_config(tmp_path)
    cfg.generation.calibration_exclude_behaviour = ["qwen@venue_split:cv-9999"]
    rated_in(monkeypatch, write_ratings(tmp_path / "ratings.jsonl"))
    with pytest.raises(UserError, match="names answers the run does not have"):
        calibration_report(cfg, run)


def test_a_candidate_that_graded_none_of_the_rated_answers_is_refused(tmp_path, monkeypatch):
    from crossvenue_rag import jsonl
    from crossvenue_rag.generation.calibration import calibration_report
    from tests.generation.factories import graded_judgement

    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)
    write_judgements(run, "opus", cost=0.03)
    ratings = tmp_path / "ratings.jsonl"
    rating = {"rater": "tester", "at": "2026-09-25T00:00:00Z", "rating": graded_judgement(0)}
    jsonl.write(ratings, [{"answer_id": "qwen@venue_split:cv-9999", **rating}])
    rated_in(monkeypatch, ratings)
    with pytest.raises(UserError, match="sonnet graded no answer rated in") as caught:
        calibration_report(calibration_config(tmp_path), run)
    assert "judge-sample --judge sonnet" in caught.value.hint


def test_a_judge_passing_one_threshold_is_said_to_pass_one_dimension():
    from crossvenue_rag.generation.calibration import select_judge

    thresholds: dict[CalibrationDimension, float] = {"supported": 0.61, "coverage": 0.70}
    low = {"supported": {"kappa": 0.1}, "venue_correct": {"kappa": 0.1},
           "coverage": {"spearman": 0.1}}  # fmt: skip
    dims = {"a": {**low, "supported": {"kappa": 0.7}}, "b": low}
    chosen = select_judge(as_dimensions(dims), thresholds, {"a": 0.01, "b": 0.01})
    assert chosen.reason == "a meets the threshold on 1 dimension, more than any other candidate"


def test_a_candidate_without_a_statistic_or_a_cost_is_a_value_error_naming_it():
    from crossvenue_rag.generation.calibration import select_judge

    thresholds: dict[CalibrationDimension, float] = {"supported": 0.61}
    full = {"supported": {"kappa": 0.7}, "venue_correct": {"kappa": 0.7},
            "coverage": {"spearman": 0.8}}  # fmt: skip
    without_coverage = {k: v for k, v in full.items() if k != "coverage"}
    with pytest.raises(ValueError, match="a has no statistics on coverage"):
        select_judge(as_dimensions({"a": without_coverage}), thresholds, {"a": 0.01})
    with pytest.raises(ValueError, match="a has no cost per answer"):
        select_judge(as_dimensions({"a": full}), thresholds, {})
