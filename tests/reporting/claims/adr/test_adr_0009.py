"""Tests for `reporting.claims.adr.adr_0009`: the trial size and the OpenRouter price entry."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.adr import adr_0009
from crossvenue_rag.reporting.claims.reasons import OPENROUTER
from crossvenue_rag.reporting.sources import Sources

RECORD = "docs/adr/0009-generators.md"


def test_the_trial_size_is_the_number_of_trial_questions_the_config_excludes(monkeypatch):
    """With another trial set, the trial size the record states does not match."""
    src = Sources(".")
    trial = list(src.default_config.generation.calibration_exclude)
    monkeypatch.setattr(src.default_config.generation, "calibration_exclude", [*trial, "cv-9999"])
    doc = Doc.from_text(RECORD, Path(RECORD).read_text(encoding="utf-8"))
    adr_0009.generators_record(doc, src)
    assert any(
        p.message.startswith("five in the text, 6 recomputed (hole 1 of 'In a {}-question")
        for p in doc.problems
    )


def test_the_generator_count_is_not_in_the_openrouter_price_entry():
    """The generator count beside the prices comes from `generation.matrix` and is compared."""
    src = Sources(".")
    doc = Doc.from_text(RECORD, Path(RECORD).read_text(encoding="utf-8"))
    adr_0009.generators_record(doc, src)
    listed = [g.text for g in doc.given_numbers if g.reason == OPENROUTER]
    assert "0.04 0.15" in listed
