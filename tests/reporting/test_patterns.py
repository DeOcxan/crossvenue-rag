"""Tests for `crossvenue_rag.reporting.patterns`: typography, numbers, identifiers, references."""

from __future__ import annotations

import pytest

from crossvenue_rag.formatting import MINUS
from crossvenue_rag.reporting.patterns import (
    HYPHEN_MINUS,
    REF,
    TOKEN,
    collapse,
    is_reference,
    table_lines,
    typeset,
)


def test_typeset_replaces_each_ascii_spelling_with_the_documents_character():
    text = typeset("a <-> b, k >= 5, 1--2, x - y, nDCG'")
    assert text == f"a \u2194 b, k \u2265 5, 1\u20132, x {MINUS} y, nDCG\u2032"


def test_a_minus_sign_written_as_a_hyphen_is_typeset_as_u2212():
    assert typeset("0.5 - 0.3") == f"0.5 {MINUS} 0.3" == "0.5 \u2212 0.3"


def test_collapse_turns_every_whitespace_run_line_breaks_included_into_one_space():
    assert collapse("a \n\t b  c\n") == "a b c "


def test_token_finds_standalone_numbers_grouped_numbers_units_and_number_words():
    text = "Recall 0.465 on 1,234 docs in 12ms, two runs"
    assert [m.group() for m in TOKEN.finditer(text)] == ["0.465", "1,234", "12ms", "two"]


def test_token_skips_digits_inside_versions_dates_and_identifiers():
    text = "v1.2.3 on 2024-01-05 for run-7 at x2y"
    assert [m.group() for m in TOKEN.finditer(text)] == []


def test_hyphen_minus_flags_a_hyphen_before_a_digit_but_not_inside_a_word():
    assert [m.group() for m in HYPHEN_MINUS.finditer("a -3, x-3, --3")] == ["-3"]


def test_ref_finds_run_directories_full_hashes_and_elided_hash_prefixes():
    text = "results/20240101T120000Z_bm25 and abcdef0123456789 and deadbeef\u2026 and abc"
    assert [m.group() for m in REF.finditer(text)] == [
        "results/20240101T120000Z_bm25",
        "abcdef0123456789",
        "deadbeef\u2026",
    ]


def test_a_number_inside_an_adr_rule_or_section_reference_is_a_reference_and_a_bare_one_is_not():
    text = "See ADR 0007, rule 5 and \u00a7 3 for 42 runs"
    found = {m.group(): is_reference(text, m) for m in TOKEN.finditer(text)}
    assert found == {"0007": True, "5": True, "3": True, "42": False}


def test_a_table_is_its_first_run_of_rows_with_their_offsets():
    text = "intro\n| a |\n|---|\n| 1 |\nend\n| b |\n"
    assert table_lines(text, 0) == [(6, "| a |"), (12, "|---|"), (18, "| 1 |")]


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("the five-question trial", []),
        ("a two-sided test", []),
        ("a 50-question set", []),
        ("the two runs", ["two"]),
        ("Five of them.", ["Five"]),
    ],
)
def test_a_number_word_joined_to_a_word_by_a_hyphen_is_not_a_number(text: str, tokens: list[str]):
    assert [m.group() for m in TOKEN.finditer(text)] == tokens
