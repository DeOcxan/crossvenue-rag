"""Tests for `reporting.claims.adr`: the dispatch on the record's number."""

from __future__ import annotations

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.claims.adr import RECORDS, adr, record_number
from crossvenue_rag.reporting.sources import Sources


def test_the_record_number_is_read_from_the_file_name():
    assert record_number("docs/adr/0004-index-version.md") == "0004"
    assert record_number("docs/adr/README.md") is None
    assert sorted(RECORDS) == [f"{i:04d}" for i in range(1, 14)]


def test_a_record_defines_its_number_then_runs_its_claims(tmp_path):
    text = "# 0004. `index_version`\n\n## Consequences\n\nTwo checks cover that gap.\n"
    doc = Doc.from_text("docs/adr/0004-index-version.md", text)
    adr(doc, Sources(tmp_path))
    doc.audit()
    assert doc.problems == []
    assert [g.reason for g in doc.given_numbers] == [
        "the record's number",
        "the list that follows in the sentence",
    ]


def test_a_file_without_a_number_is_not_checked(tmp_path):
    doc = Doc.from_text("docs/adr/README.md", "# Decision records\n\n0001 is the first.\n")
    adr(doc, Sources(tmp_path))
    assert doc.given_numbers == []
    assert doc.checked == 0
