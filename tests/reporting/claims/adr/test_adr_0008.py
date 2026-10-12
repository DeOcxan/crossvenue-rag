"""Tests for `reporting.claims.adr.adr_0008`: the module size the record decides."""

from __future__ import annotations

from pathlib import Path

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.adr import adr_0008
from crossvenue_rag.reporting.compute import repository
from crossvenue_rag.reporting.sources import Sources

RECORD = "docs/adr/0008-package-structure.md"


def test_the_size_rule_is_the_bound_the_repository_checks(monkeypatch):
    """With another module size bound, the size the record states does not match."""
    monkeypatch.setattr(repository, "MODULE_CODE_LINES_BOUND", 500)
    doc = Doc.from_text(RECORD, Path(RECORD).read_text(encoding="utf-8"))
    adr_0008.package_structure_record(doc, Sources("."))
    assert any(
        p.message.startswith("400 in the text, 500 recomputed (hole 1 of 'No module above")
        for p in doc.problems
    )
