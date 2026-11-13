"""Tests for `crossvenue_rag.reporting.span_sets`: span membership."""

from __future__ import annotations

from crossvenue_rag.reporting.span_sets import Spans


def test_spans_include_the_start_exclude_the_end_and_merge_overlaps():
    spans = Spans([(5, 8), (0, 2), (7, 10)])
    assert [p for p in range(12) if p in spans] == [0, 1, 5, 6, 7, 8, 9]


def test_no_offset_lies_in_an_empty_set_of_spans():
    assert 0 not in Spans([])
