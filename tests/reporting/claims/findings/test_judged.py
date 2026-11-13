"""Tests for `reporting.claims.findings.judged`: the venue-attribution claim."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.findings import judged as claims
from crossvenue_rag.reporting.compute import answers_judged
from crossvenue_rag.reporting.sources import Sources

FINDINGS = "docs/results/findings.md"


def test_no_misattributed_comparison_claim_fails_when_a_pair_has_a_mean_above_zero(monkeypatch):
    """An interval that includes 0 with a mean above 0 does not show that no pair misattributes."""
    monkeypatch.setattr(answers_judged, "means_zero", lambda *args: False)
    doc = Doc.from_text(FINDINGS, Path(FINDINGS).read_text(encoding="utf-8"))
    claims.generator_comparison_claims(doc, Sources("."))
    assert any(
        p.message.startswith("does not hold") and "venue_attribution_error" in p.message
        for p in doc.problems
    )
