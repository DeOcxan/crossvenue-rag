"""Tests for `reporting.claims.datasheet`: the augmentation's top ranks and the display size."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims import datasheet
from crossvenue_rag.reporting.claims.methodology import protocol
from crossvenue_rag.reporting.compute import datasets
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import needs_local_corpus

DATASHEET = "evals/datasheet.md"

# The datasheet's claims run on the local corpus and index.
pytestmark = needs_local_corpus


def test_the_never_shown_chunks_are_counted_in_the_top_ranks_the_compute_functions_use(
    monkeypatch,
):
    """With another top of the pool run, the top the never-shown chunks count does not match."""
    monkeypatch.setattr(datasets, "TOP_RANKS", 20)
    doc = Doc.from_text(DATASHEET, Path(DATASHEET).read_text(encoding="utf-8"))
    datasheet.datasheet(doc, Sources("."))
    assert any(
        p.message.startswith("10 in the text, 20 recomputed (hole 3 of 'never showed")
        for p in doc.problems
    )


def test_the_candidates_per_question_are_the_display_size_of_the_pool(monkeypatch):
    """With another display size, the candidates per question the timings use do not match."""
    monkeypatch.setattr(protocol, "pool_display_text", lambda src: "30")
    doc = Doc.from_text(DATASHEET, Path(DATASHEET).read_text(encoding="utf-8"))
    datasheet.datasheet(doc, Sources("."))
    assert any(
        p.message.startswith("25 in the text, 30 recomputed (hole 3 of 'median")
        for p in doc.problems
    )
