"""Tests for `reporting.compute.references`: the cited literature, as recorded."""

from __future__ import annotations

from crossvenue_rag.reporting.compute import references
from crossvenue_rag.reporting.sources import Sources

REGISTRY = """
references:
- section: Agreement
  entry: '- Landis, J. R. & Koch, G. G. (1977). *Title.* Biometrics 33(1):159-174.'
- section: Formats
  entry: '- [Cohen''s kappa](https://en.wikipedia.org/wiki/Cohen)'
- section: Agreement
  entry: '- Fleiss, J. L. (1971). *Title.*'
"""

DOCUMENT = """# Methods

## References

### Agreement
- A

### Formats
- B

### Not verified / left open
- C

### After
"""


def test_entries_are_grouped_by_section_in_the_registry_order(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/references.yaml").write_text(REGISTRY, encoding="utf-8")
    by_section = references.references(Sources(tmp_path))
    assert list(by_section) == ["Agreement", "Formats"]
    assert [e[:9] for e in by_section["Agreement"]] == ["- Landis,", "- Fleiss,"]


def test_the_written_sections_end_before_the_unverified_notes():
    assert references.written_sections(DOCUMENT) == ["Agreement", "Formats"]
