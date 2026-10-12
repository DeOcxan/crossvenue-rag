"""Tests for `crossvenue_rag.reporting.outcomes`: problems, listed numbers and parameter holes."""

from __future__ import annotations

import dataclasses

from crossvenue_rag.reporting.outcomes import SKIP, Listed, Param, Problem


def test_a_problem_prints_as_path_line_and_message():
    assert str(Problem("docs/a.md", 12, "text not found")) == "docs/a.md:12: text not found"


def test_problems_with_the_same_fields_are_equal():
    assert Problem("a.md", 1, "m") == Problem("a.md", 1, "m")
    assert Problem("a.md", 1, "m") != Problem("a.md", 2, "m")


def test_a_param_is_the_string_it_holds_and_carries_its_reason():
    param = Param("0.5", "the threshold of the config")
    assert param == "0.5" and isinstance(param, str)
    assert param.reason == "the threshold of the config"


def test_a_listed_number_keeps_its_file_line_text_and_reason():
    listed = Listed("README.md", 3, "50", "given")
    assert dataclasses.astuple(listed) == ("README.md", 3, "50", "given")


def test_skip_is_a_sentinel_equal_only_to_itself():
    assert object() != SKIP
    assert SKIP != "" and SKIP is not None
