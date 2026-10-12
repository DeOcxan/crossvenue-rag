"""Tests for `reporting.claims.methodology.deviations`: the registered sizes the rows state."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.dataset.selection import TARGETS
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.methodology import deviations, protocol
from crossvenue_rag.reporting.sources import Sources

METHODOLOGY = "evals/methodology.md"


def test_the_d2_and_d3_rows_state_the_set_size_and_the_venue_floor(monkeypatch):
    """With another fact target and venue floor, the D2 and D3 rows do not match them."""
    monkeypatch.setitem(TARGETS, "fact", TARGETS["fact"] + 1)
    monkeypatch.setattr(protocol, "VENUE_FLOOR", "9")
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    src = Sources(".")
    deviations.selection(doc, src)
    deviations.venues(doc, src)
    assert any(
        p.message.startswith("105 in the text, 106 recomputed (hole 3 of '| D2 |")
        for p in doc.problems
    )
    assert any(
        p.message.startswith("8 in the text, 9 recomputed (hole 1 of '| D3 |") for p in doc.problems
    )


def test_the_d26_row_holds_only_while_the_three_constraints_fit_in_100(monkeypatch):
    """The committed targets pass; with comparison at 60 the constraints sum to 105."""
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    deviations.size_deviation(doc)
    assert doc.problems == []
    monkeypatch.setitem(TARGETS, "comparison", 60)
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    deviations.size_deviation(doc)
    assert any(p.message.startswith("95 in the text, 105 recomputed") for p in doc.problems)
    assert any(p.message.startswith("does not hold: the three constraints") for p in doc.problems)
