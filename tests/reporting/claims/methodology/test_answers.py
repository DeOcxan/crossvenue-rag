"""Tests for `reporting.claims.methodology.answers`: the calibration table's cells, the counts."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.methodology import answers
from crossvenue_rag.reporting.claims.methodology.answers import dimension_label, threshold_cell
from crossvenue_rag.reporting.compute import calibration
from crossvenue_rag.reporting.sources import Sources

METHODOLOGY = "evals/methodology.md"


def test_a_statistic_or_dimension_the_table_does_not_name_is_a_value_error():
    assert threshold_cell("kappa", 0.61) == "\u03ba >= 0.61"
    assert dimension_label("coverage", "answers rated") == "coverage (answers rated answers)"
    with pytest.raises(ValueError, match="STATISTIC_LABELS has no 'pearson'"):
        threshold_cell("pearson", 0.61)
    with pytest.raises(ValueError, match="UNITS has no 'faithfulness'"):
        dimension_label("faithfulness", "answers rated")


def test_the_section_9_counts_are_the_candidates_and_trial_questions_of_the_run(monkeypatch):
    """With another candidate count and trial set, the counts section 9 states do not match."""
    monkeypatch.setattr(calibration, "judge_candidate_count", lambda src: 3)
    src = Sources(".")
    trial = list(src.default_config.generation.calibration_exclude)
    monkeypatch.setattr(src.default_config.generation, "calibration_exclude", [*trial, "cv-9999"])
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    answers.generator_protocol_claims(doc, src)
    for have, want, template in (("Two", "3", "{} candidates"), ("five", "6", "The {}-question")):
        assert any(
            p.message.startswith(f"{have} in the text, {want} recomputed (hole 1 of '{template}")
            for p in doc.problems
        )
