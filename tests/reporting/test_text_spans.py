"""Tests for `crossvenue_rag.reporting.text_spans`: collapsed-text offsets."""

from __future__ import annotations

import re

from crossvenue_rag.reporting.text_spans import Collapsed, Found


def test_collapsed_text_has_each_whitespace_run_as_one_space():
    assert Collapsed("a  b\n\n c").text == "a b c"


def test_an_offset_in_the_collapsed_text_maps_back_to_the_text_as_written():
    written = "a  b\n\n c"
    flat = Collapsed(written)
    assert [flat.at(i) for i in range(len(flat.text))] == [0, 1, 3, 4, 7]
    assert all(written[flat.at(i)] == flat.text[i] for i in range(len(flat.text)) if i % 2 == 0)


def test_offsets_of_a_text_without_whitespace_are_unchanged():
    flat = Collapsed("abc")
    assert [flat.at(i) for i in range(3)] == [0, 1, 2]


def test_a_found_match_reports_its_span_and_group_starts_in_the_text_as_written():
    written = "a  b\n\n c"
    flat = Collapsed(written)
    match = re.search(r"(b) (c)", flat.text)
    assert match is not None
    found = Found(match, flat)
    assert (found.group, found.groups) == ("b c", ("b", "c"))
    assert found.span == (3, 8) and written[slice(*found.span)] == "b\n\n c"
    assert found.group_start(2) == 7
