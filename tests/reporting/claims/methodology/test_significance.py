"""Tests for `reporting.claims.methodology.significance`: the holes of the power-model sentences."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.dataset.selection import TARGETS
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims import comparisons
from crossvenue_rag.reporting.claims.methodology import significance
from crossvenue_rag.reporting.sources import Sources

METHODOLOGY = "evals/methodology.md"


def test_the_power_sentences_delta_is_the_delta_its_powers_are_read_at(monkeypatch):
    """With the powers read at another delta, the stated delta does not match."""
    monkeypatch.setattr(significance, "DELTA", 0.21)
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    significance.observed_power(doc, Sources("."))
    assert any(
        p.message.startswith("0.20 in the text, 0.21 recomputed (hole 1 of 'At \u03b4 = {}")
        for p in doc.problems
    )


def test_the_binary_outcomes_cutoff_is_the_retrieval_runs_top_k(monkeypatch):
    """With another top k, the cutoff of the binary-outcome sentence does not match."""
    monkeypatch.setattr(comparisons, "top_k_text", lambda src: "20")
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    significance.observed_power(doc, Sources("."))
    assert any(
        p.message.startswith("10 in the text, 20 recomputed (hole 1 of 'the gold chunk is in")
        for p in doc.problems
    )


def test_the_sample_size_rules_state_the_registered_targets(monkeypatch):
    """With another fact target, the set size the section 7 decision states does not match."""
    monkeypatch.setitem(TARGETS, "fact", TARGETS["fact"] + 1)
    doc = Doc.from_text(METHODOLOGY, Path(METHODOLOGY).read_text(encoding="utf-8"))
    significance.observed_power(doc, Sources("."))
    for template in ("{} questions, hard floor", "- {} total gives"):
        assert any(
            p.message.startswith(f"105 in the text, 106 recomputed (hole 1 of '{template}")
            for p in doc.problems
        )
