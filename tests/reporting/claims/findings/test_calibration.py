"""Tests for `reporting.claims.findings.calibration`: Opus's agreement with its repeat."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.findings import calibration as claims
from crossvenue_rag.reporting.compute import calibration
from crossvenue_rag.reporting.sources import Sources

FINDINGS = "docs/results/findings.md"


def test_one_repeat_kappa_for_two_dimensions_holds_only_when_they_agree(monkeypatch):
    """With `behaviour_correct`'s repeat agreement apart from `venue_correct`'s, the claim fails."""
    monkeypatch.setattr(calibration, "same_agreement", lambda *args: False)
    doc = Doc.from_text(FINDINGS, Path(FINDINGS).read_text(encoding="utf-8"))
    claims.repeat(doc, Sources("."))
    assert any(
        p.message == "does not hold: behaviour_correct's repeat agreement equals venue_correct's"
        for p in doc.problems
    )
