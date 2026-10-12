"""Tests for `generation.calibration_report`: calibration.md."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag import jsonl
from crossvenue_rag.generation import calibration_report
from tests.conftest import calibration_of, output_defects
from tests.generation.factories import (
    ANSWER_IDS,
    graded_judgement,
    judge_json,
    rated_in,
    write_judgements,
    write_ratings,
)
from tests.generation.offline import (
    calibration_config,
)


def test_calibration_markdown_has_no_output_defects():
    from crossvenue_rag.generation.calibration_report import calibration_markdown

    row = {
        "n": 91,
        "agreement": 0.912,
        "agreement_ci": [0.84, 0.96],
        "kappa": 0.238,
        "kappa_ci": [-0.0512, 0.5581],
        "kappa_undefined_resamples": 3634,
    }
    report = {
        "run": "results/r",
        "answers_rated": 40,
        "raters": ["a", "b"],
        "resamples": 10000,
        "seed": 1,
        "excluded_from_behaviour_correct": ["x@y:q1"],
        "comparisons": {"opus vs human": {"supported": row}},
        "thresholds": {"supported": 0.61},
        "statistic": {"supported": "kappa"},
        "candidates": {
            "opus": {"passes": [], "fails": ["supported"], "cost_per_answer_usd": 0.0301}
        },
        "selected": "opus",
        "reason": "most thresholds",
    }
    text = calibration_markdown(calibration_of(**report))
    assert "undefined in 3,634 resamples" in text and "$0.030" in text
    assert text.startswith("# Judge calibration: `results/r`\n")
    assert output_defects(text) == []


def test_a_threshold_is_written_at_two_decimals_and_a_missing_one_as_a_dash():
    from crossvenue_rag.generation.calibration_report import calibration_markdown

    report = calibration_of(
        thresholds={"supported": 0.6},
        statistic={"supported": "kappa", "coverage": "spearman"},
        comparisons={"sonnet vs human": {"supported": {"n": 3}, "coverage": {"n": 3}}},
    )
    lines = calibration_markdown(report).splitlines()
    bar = "| kappa \u2265 0.60 |"
    assert any(line.startswith("| `supported` |") and line.endswith(bar) for line in lines)
    assert any(line.startswith("| `coverage` |") and line.endswith("| - |") for line in lines)


def test_the_seed_is_written_through_the_formatting_module_as_an_identifier(monkeypatch):
    from crossvenue_rag.generation.calibration_report import calibration_markdown

    report = calibration_of(seed=20260925)
    assert "seed 20260925." in calibration_markdown(report)
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<seed {value}>")
    assert "seed <seed 20260925>." in calibration_markdown(report)


def test_select_judge_files_writes_the_report_and_selects_the_agreeing_judge(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)
    write_judgements(run, "opus", flip=True, cost=0.03)
    jsonl.write(
        run / "judgements_sonnet_r1.jsonl",
        [
            {"answer_id": a, "judge": judge_json(judgement=graded_judgement(i))}
            for i, a in enumerate(ANSWER_IDS)
        ],
    )
    rated_in(monkeypatch, write_ratings(tmp_path / "ratings.jsonl"))
    text = calibration_report.select_judge_files(calibration_config(tmp_path), run)
    report = json.loads((run / "calibration.json").read_text())
    assert (run / "calibration.md").read_text(encoding="utf-8") == text
    assert report["selected"] == "sonnet"
    assert report["raters"] == ["tester"] and report["answers_rated"] == 12
    assert set(report["comparisons"]) == {
        "sonnet vs human",
        "opus vs human",
        "sonnet vs opus",
        "sonnet vs sonnet replicate 1",
    }
    assert report["candidates"]["sonnet"]["cost_per_answer_usd"] == pytest.approx(0.01)
    assert report["comparisons"]["sonnet vs human"]["supported"]["kappa"] == 1.0
    assert "Selected: sonnet." in text and "## opus vs human" in text


def test_select_judge_files_without_human_ratings_stops_and_writes_nothing(tmp_path, monkeypatch):
    from crossvenue_rag.errors import UserError

    run = tmp_path / "run"
    run.mkdir()
    write_judgements(run, "sonnet", cost=0.01)
    write_judgements(run, "opus", flip=True, cost=0.03)
    rated_in(monkeypatch, tmp_path / "no_ratings.jsonl")
    with pytest.raises(UserError, match="no human rating in") as caught:
        calibration_report.select_judge_files(calibration_config(tmp_path), run)
    assert "crossvenue rate" in caught.value.hint
    assert sorted(p.name for p in run.iterdir()) == [
        "answers.jsonl",
        "judgements_opus.jsonl",
        "judgements_sonnet.jsonl",
    ]


def test_a_threshold_without_its_statistic_is_a_value_error_naming_the_dimension():
    from crossvenue_rag.generation.calibration_report import calibration_markdown

    report = calibration_of(
        thresholds={"supported": 0.6},
        comparisons={"sonnet vs human": {"supported": {"n": 3}}},
    )
    with pytest.raises(ValueError, match="threshold on supported but no statistic"):
        calibration_markdown(report)
