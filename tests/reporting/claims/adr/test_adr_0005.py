"""Tests for `reporting.claims.adr.adr_0005`: the labeling pool's limits."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.adr import adr_0005
from crossvenue_rag.reporting.compute import datasets_rounds
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import needs_local_corpus

RECORD = "docs/adr/0005-labeling-pool.md"

# The record's claims run on the local corpus and index.
pytestmark = needs_local_corpus


def test_the_depth_multiplier_is_the_one_the_labeling_pool_sets(monkeypatch):
    """With another depth multiplier in the pool's code, the record's does not match."""
    monkeypatch.setattr(datasets_rounds, "pool_depth_multiplier", lambda src: 5)
    doc = Doc.from_text(RECORD, Path(RECORD).read_text(encoding="utf-8"))
    adr_0005.labeling_pool_record(doc, Sources("."))
    assert any(
        p.message.startswith("four in the text, 5 recomputed (hole 2 of 'fewer than {} remain")
        for p in doc.problems
    )
