"""Tests for `reporting.check`: templates, tables, identifiers and the audit."""

from __future__ import annotations

import pytest

from crossvenue_rag.formatting import MINUS
from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.outcomes import SKIP, Param


def test_a_matching_template_covers_its_numbers():
    doc = Doc.from_text("d.md", "Recall is 0.465 on 50\nquestions.\n")
    doc.check("Recall is {} on {} questions.", "0.465", "50")
    doc.audit()
    assert doc.problems == [] and doc.checked == 2


def test_a_wrapped_template_matches_a_one_line_paragraph():
    """A template written with a line break and indentation matches the same
    words on one line: a run of whitespace matches any run of whitespace."""
    doc = Doc.from_text("d.md", "- Recall is 0.465 on 50 questions.\n")
    doc.check("Recall is {} on\n  {} questions.", "0.465", "50")
    doc.audit()
    assert doc.problems == [] and doc.checked == 2


def test_a_mismatch_names_the_line_and_both_values():
    doc = Doc.from_text("d.md", "Intro.\n\nRecall is 0.466 on 50 questions.\n")
    doc.check("Recall is {} on {} questions.", "0.465", "50")
    [problem] = doc.problems
    assert (problem.path, problem.line) == ("d.md", 3)
    assert "0.466 in the text, 0.465 recomputed" in problem.message


def test_text_that_is_gone_is_a_problem_not_a_pass():
    doc = Doc.from_text("d.md", "Recall is 0.465.\n")
    doc.check("Precision is {}.", "0.5")
    assert "text not found" in doc.problems[0].message


def test_a_number_no_check_covers_is_a_problem():
    doc = Doc.from_text("d.md", "Recall is 0.465 and precision 0.2.\n")
    doc.check("Recall is {} and", "0.465")
    doc.audit()
    [problem] = doc.problems
    assert "'0.2' is not recomputed" in problem.message


def test_identifiers_code_links_dates_and_references_are_not_numbers():
    text = (
        "recall@10 on v0.2.1-gold, `crossvenue eval`, cv-0045, 2026-09-25, "
        "[ADR 0007](../adr/0007-x.md), \u00a79a, rule 5, round 1, p50 and (Sakai 2007).\n"
        "```\nmake eval 42\n```\n"
    )
    doc = Doc.from_text("d.md", text)
    doc.audit()
    assert doc.problems == []


def test_number_words_headings_hyphen_minus_and_units_are_audited():
    text = (
        "## Three of four pass\n\nThe corpus has eight venues. The gap is -0.5; "
        "it is 10x faster and costs 12ms.\n"
    )
    doc = Doc.from_text("d.md", text)
    doc.check("## {} of {} pass", "3", "4")
    doc.audit()
    messages = [p.message for p in doc.problems]
    assert "number 'eight' is not recomputed by any check" in messages
    assert any("'-0' : a minus sign" in m or "'-0': a minus sign" in m for m in messages)
    assert "number '10x' is not recomputed by any check" in messages
    assert "number '12ms' is not recomputed by any check" in messages
    assert not any("Three" in m or "'4'" in m for m in messages)


def test_a_number_word_in_a_hole_is_compared_by_value():
    doc = Doc.from_text("d.md", "The corpus has eight venues.\n")
    doc.check("The corpus has {} venues.", "9")
    assert "eight in the text, 9 recomputed" in doc.problems[0].message


def test_runs_and_hashes_must_be_checked_by_ref():
    text = "Run `results/20260925T012314Z`, index `2825a7160c4d09bf`, manifest `c419b4be\u2026`.\n"
    unchecked = Doc.from_text("d.md", text)
    unchecked.audit()
    assert len(unchecked.problems) == 3
    doc = Doc.from_text("d.md", text)
    doc.ref(
        "Run `results/{}`, index `{}`, manifest `{}`.",
        "20260925T144223Z",
        "2825a7160c4d09bf",
        "c419b4be\u2026",
    )
    doc.audit()
    [problem] = doc.problems
    assert "20260925T012314Z in the text, 20260925T144223Z recomputed" in problem.message


def test_identifiers_checked_by_ref_are_counted_apart_from_numbers():
    doc = Doc.from_text("d.md", "Run `results/20260925T012314Z` has 3 methods.\n")
    doc.ref("Run `results/{}`", "20260925T012314Z")
    doc.check("has {} methods", "3")
    assert (doc.refs, doc.checked) == (1, 1)


def test_a_definition_is_covered_and_listed_with_its_reason():
    doc = Doc.from_text("d.md", "Two checks follow.\n")
    doc.definition("Two checks follow", reason="the list that follows")
    doc.audit()
    assert doc.problems == [] and doc.given_numbers[0].reason == "the list that follows"


LIST = "### Works\n- Cohen (1960)\n- Fleiss (1971)\n\nAfter.\n"


def test_a_list_equal_to_the_expected_items_is_covered():
    doc = Doc.from_text("d.md", LIST)
    doc.items("### Works", ["- Cohen (1960)", "- Fleiss (1971)"], reason="cited")
    doc.audit()
    assert doc.problems == [] and len(doc.given_numbers) == 2


def test_a_missing_an_unexpected_and_a_moved_item_are_each_a_problem():
    doc = Doc.from_text("d.md", LIST)
    doc.items("### Works", ["- Cohen (1961)", "- Fleiss (1971)"], reason="cited")
    assert [p.message.split(":")[0] for p in doc.problems] == ["missing item", "unexpected item"]
    doc = Doc.from_text("d.md", LIST)
    doc.items("### Works", ["- Fleiss (1971)", "- Cohen (1960)"], reason="cited")
    assert [p.message.split(":")[0] for p in doc.problems] == ["items out of order"]


def test_typography_is_spelled_in_ascii_in_the_spec():
    doc = Doc.from_text(
        "d.md", f"a {MINUS} b is {MINUS}0.062 over 0.03\u20130.09, naming \u2265 2 venues.\n"
    )
    doc.check("a - b is {} over {}--{}, naming >= {} venues.", f"{MINUS}0.062", "0.03", "0.09", "2")
    doc.audit()
    assert doc.problems == []


def test_a_literal_number_in_a_template_is_refused():
    doc = Doc.from_text("d.md", "top 10 of 50\n")
    with pytest.raises(ValueError, match="literal number"):
        doc.check("top 10 of {}", "50")


def test_tables_compare_cell_by_cell_and_every_row_is_accounted_for():
    text = (
        "Results:\n\n| method | recall | note |\n|---|---:|---|\n"
        "| bm25 | 0.465 | text |\n| dense | 0.354 | text |\n| extra | 1.0 | x |\n"
    )
    doc = Doc.from_text("d.md", text)
    doc.table("Results:", {"bm25": ["0.465", SKIP], "dense": ["0.353", SKIP], "gone": ["1", SKIP]})
    messages = [p.message for p in doc.problems]
    assert "row 'dense', column 2: '0.354' in the text, '0.353' recomputed" in messages
    assert "row 'extra' is not recomputed" in messages
    assert "row 'gone' missing from the table" in messages


def test_a_claim_fails_when_the_recomputed_condition_does_not_hold():
    doc = Doc.from_text("d.md", "no chunk above 0.7.\n")
    doc.claim("no chunk above {}.", "0.7", holds=False, what="max cosine < 0.7")
    assert doc.problems[0].message == "does not hold: max cosine < 0.7"


def test_given_and_unsaved_numbers_are_listed_and_covered():
    doc = Doc.from_text("d.md", "seed 20260924; one call took 6,936 ms.\n")
    doc.given("seed {};", "20260924", reason="config")
    doc.unsaved("one call took {} ms", reason="ad hoc timing")
    doc.audit()
    assert doc.problems == []
    assert [g.reason for g in doc.given_numbers] == ["config"]
    assert [u.reason for u in doc.unsaved_numbers] == ["ad hoc timing"]


def test_a_given_number_is_not_counted_as_compared_as_a_param_is_not():
    doc = Doc.from_text("d.md", "seed 20260924; at 95% level; n = 50.\n")
    doc.given("seed {};", "20260924", reason="config")
    doc.check("at {}% level", Param("95", "convention"))
    assert doc.checked == 0 and len(doc.given_numbers) == 2
    doc.check("n = {}.", "50")
    assert doc.checked == 1


def test_each_compared_value_is_listed_with_its_line_and_template():
    doc = Doc.from_text("d.md", "Intro.\n\nn = 50 of 114.\n")
    doc.check("n = {} of {}.", "50", Param("114", "registered"))
    assert [(c.line, c.text, c.reason) for c in doc.compared_numbers] == [
        (3, "50", "n = {} of {}.")
    ]


def test_a_compared_table_cell_counts_as_one_value_whatever_it_holds():
    text = "Table:\n\n| method | outcome | counts |\n|---|---|---|\n| bm25 | answer | 25 to 9 |\n"
    doc = Doc.from_text("d.md", text)
    doc.table("Table:", {"bm25": ["answer", "25 to 9"]})
    assert doc.problems == [] and doc.checked == 2
    assert [c.text for c in doc.compared_numbers] == ["answer", "25 to 9"]
    assert doc.compared_numbers[0].reason == "row 'bm25', column 2"


def test_an_identifier_is_listed_apart_from_the_compared_values():
    doc = Doc.from_text("d.md", "Run `results/20260925T012314Z`.\n")
    doc.ref("Run `results/{}`.", "20260925T012314Z")
    assert doc.checked == 0
    assert [(i.line, i.text) for i in doc.identifiers] == [(1, "20260925T012314Z")]


def test_a_listed_span_or_template_is_listed_in_full_however_long():
    long = "a sentence written long enough that a short excerpt of it would drop the values"
    text = f"First {long} 20 and 50.\n\nSecond {long} 7.\n\nThird {long} 3.\n"
    doc = Doc.from_text("d.md", text)
    doc.unsaved(f"First {long} {{}} and {{}}.", reason="no saved run")
    doc.check(f"Second {long} {{}}.", "7")
    doc.definition(f"Third {long} 3.", reason="a definition")
    assert doc.problems == []
    assert doc.unsaved_numbers[0].text == f"First {long} 20 and 50."
    assert doc.compared_numbers[0].reason == f"Second {long} {{}}."
    assert doc.given_numbers[0].text == f"Third {long} 3."


def test_a_claim_matches_however_its_paragraph_is_wrapped():
    for text in ("Gemini refused 12 (venue_split).\n", "Gemini\n  refused 12\n(venue_split).\n"):
        doc = Doc.from_text("d.md", text)
        doc.check("Gemini refused {} (venue_split).", "12")
        doc.audit()
        assert doc.problems == [], text


def test_a_wrapped_mismatch_is_reported_on_the_line_of_its_number():
    doc = Doc.from_text("d.md", "Intro.\n\nGemini refused\n13 (venue_split).\n")
    doc.check("Gemini refused {} (venue_split).", "12")
    assert [(p.line, p.message[:24]) for p in doc.problems] == [(4, "13 in the text, 12 recom")]


def test_code_spans_and_references_broken_across_lines_are_not_numbers():
    doc = Doc.from_text("d.md", "See `cv-0004 sends 403\ntoo often` and ADR\n0003.\n")
    doc.audit()
    assert doc.problems == []


def test_a_definition_that_ends_in_a_digit_does_not_cover_a_longer_number():
    doc = Doc.from_text("d.md", "Drafted by Opus 5.5 and a 5023 error.\n")
    doc.definition("Opus 5", reason="a model name")
    doc.definition("a 502", reason="a status")
    doc.audit()
    assert any("5.5" in str(p) for p in doc.problems)
    assert any("5023" in str(p) for p in doc.problems)


def test_a_table_row_without_its_closing_pipe_is_compared():
    doc = Doc.from_text("d.md", "Results:\n\n| m | v |\n|---|---|\n| bm25 | 0.465\n")
    doc.table("Results:", {"bm25": ["0.466"]})
    assert any("0.465" in str(p) and "0.466" in str(p) for p in doc.problems)


def test_a_caption_with_more_holes_than_values_is_a_value_error():
    doc = Doc.from_text("d.md", "Recall on 50 questions in `20260925T132613Z`.\n")
    with pytest.raises(ValueError, match="more holes than the 1 values"):
        doc.caption("Recall on {} questions in {} in `{run}`.", ["20260925T132613Z"], "50")
