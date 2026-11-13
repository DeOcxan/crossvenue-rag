"""Tests for `reporting.claims.datasheet`: the augmentation, the display size and round 1's pace."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from crossvenue_rag.dataset.journal import FIRST_ROUND
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims import datasheet
from crossvenue_rag.reporting.claims.methodology import protocol
from crossvenue_rag.reporting.compute import datasets, round3
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
        p.message.startswith("25 in the text, 30 recomputed (hole 2 of 'median")
        for p in doc.problems
    )


def test_the_seconds_per_candidate_divide_by_the_display_size_of_the_pool(monkeypatch):
    """With another display size, the seconds per candidate of round 1 do not match."""
    monkeypatch.setattr(datasets, "pool_display", lambda src: 50)
    doc = Doc.from_text(DATASHEET, Path(DATASHEET).read_text(encoding="utf-8"))
    datasheet.datasheet(doc, Sources("."))
    assert any(
        p.message.startswith("0.7 in the text, 0.3 recomputed (hole 3 of 'median")
        for p in doc.problems
    )


def test_round_one_s_median_per_question_is_the_pace_agreement_states(monkeypatch):
    """With another round-1 pace, the datasheet's median per question does not match."""
    pace = round3.pace

    def slower(src, round_number):
        found = pace(src, round_number)
        return (
            dataclasses.replace(found, per_question=40.0) if round_number == FIRST_ROUND else found
        )

    monkeypatch.setattr(round3, "pace", slower)
    doc = Doc.from_text(DATASHEET, Path(DATASHEET).read_text(encoding="utf-8"))
    datasheet.datasheet(doc, Sources("."))
    assert any(
        p.message.startswith("17 in the text, 40 recomputed (hole 1 of 'median")
        for p in doc.problems
    )


def test_the_augmentation_s_relevant_chunks_are_those_its_journal_marks(monkeypatch):
    """With other augmentation marks, the relevant chunks the datasheet states do not match."""
    counts = datasets.augmentation

    def more(src):
        found = counts(src)
        return dataclasses.replace(found, relevant=found.relevant + 5)

    monkeypatch.setattr(datasets, "augmentation", more)
    doc = Doc.from_text(DATASHEET, Path(DATASHEET).read_text(encoding="utf-8"))
    datasheet.datasheet(doc, Sources("."))
    assert any(
        p.message.startswith("17 in the text, 22 recomputed (hole 1 of '{} were relevant")
        for p in doc.problems
    )
