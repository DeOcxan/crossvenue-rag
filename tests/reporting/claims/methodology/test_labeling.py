"""Tests for `reporting.claims.methodology.labeling`: the retest gap the section states."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.methodology import labeling
from crossvenue_rag.reporting.sources import Sources

METHODOLOGY = "evals/methodology.md"


def test_the_gap_rules_state_the_minimum_retest_gap(monkeypatch):
    """With another minimum gap, the gap the section 5 rules state does not match."""
    monkeypatch.setattr(labeling, "MIN_RETEST_DAYS", 15)
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    labeling.agreement(doc, Sources("."))
    for hole, template in ((1, "Round 2 met the {}-day"), (2, "Fallback: a {}-day gap")):
        assert any(
            p.message.startswith(f"14 in the text, 15 recomputed (hole {hole} of '{template}")
            for p in doc.problems
        )
