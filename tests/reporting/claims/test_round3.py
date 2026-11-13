"""Tests for `reporting.claims.round3`: the README sentence on the retrieval primary."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims import round3
from crossvenue_rag.reporting.compute import sensitivity
from crossvenue_rag.reporting.run_ids import RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources

README = "README.md"


def test_an_unchanged_primary_needs_the_difference_and_interval_of_the_retrieval_run(
    monkeypatch,
):
    """With another difference in the retrieval run, the primary is not unchanged."""
    row = sensitivity.paired_row

    def moved(src, run, comparison):
        found = row(src, run, comparison)
        return (
            found.model_copy(update={"diff": found.diff + 0.1}) if run == RETRIEVAL_RUN else found
        )

    monkeypatch.setattr(sensitivity, "paired_row", moved)
    doc = Doc.from_text(README, Path(README).read_text(encoding="utf-8"))
    round3.results_line(doc, Sources("."))
    assert any(
        p.message.startswith("does not hold: the retrieval primary on v0.3-gold")
        for p in doc.problems
    )
