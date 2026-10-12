"""Tests for `reporting.claims.methodology.decisions`: the registered targets the text states."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.dataset.selection import TARGETS
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.methodology import decisions
from crossvenue_rag.reporting.sources import Sources

METHODOLOGY = "evals/methodology.md"


def test_the_decisions_and_section_3_state_the_category_targets(monkeypatch):
    """With another unanswerable target, the decision table and section 3 do not match it."""
    monkeypatch.setitem(TARGETS, "unanswerable", TARGETS["unanswerable"] + 1)
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    decisions.decision_table(doc, Sources("."))
    decisions.unanswerable(doc)
    for hole, template in (
        (4, "Labeled set size | {} questions"),
        (2, "Category split | "),
        (1, "At least {} questions are unanswerable"),
    ):
        assert any(
            p.message.startswith(f"15 in the text, 16 recomputed (hole {hole} of '{template}")
            for p in doc.problems
        )


def test_the_pooling_sentence_states_the_pool_methods_and_their_depth(monkeypatch):
    """With another pool, the count of systems the pooling section states does not match."""
    monkeypatch.setattr(decisions, "POOL_METHODS", ("bm25", "dense", "hybrid", "hybrid_rerank"))
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    decisions.pooling(doc, Sources("."))
    assert any(
        p.message.startswith('Five in the text, 4 recomputed (hole 1 of "{} systems is still')
        for p in doc.problems
    )
