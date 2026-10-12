"""Tests for `evaluation.labels`: how methods, missing values and tables are shown."""

from __future__ import annotations

from crossvenue_rag.evaluation.labels import POOL_NOTE, fmt, markdown_table, method_label


def test_a_missing_value_is_a_single_dash_not_zero():
    assert fmt(None) == "-"
    assert fmt(0.0) == "0.000"
    assert fmt(0.46549) == "0.465"


def test_a_markdown_table_has_a_header_a_rule_per_column_and_one_line_per_row():
    table = markdown_table(["a", "b"], [["1", "2"], ["3", "-"]])
    assert table == "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | - |"
    assert markdown_table(["a"], []) == "| a |\n|---|"


def test_the_pool_note_shows_a_missing_judged_metric_as_a_dash():
    assert "Both are `-` when the judgment history is not available" in POOL_NOTE
    assert "n/a" not in POOL_NOTE


def test_methods_outside_the_pool_carry_a_dagger_and_oracles_say_so():
    assert method_label("hybrid") == "hybrid"
    assert method_label("venue_split_oracle") == "venue_split_oracle (oracle) \u2020"
    assert "\u2020" in POOL_NOTE


def test_the_pool_note_counts_unjudged_chunks_as_misses_and_claims_no_bound_for_them():
    unjudged = POOL_NOTE.split("`unjudged@k`", 1)[1].split("`ndcg_prime@k`", 1)[0]
    assert "counted as a miss" in unjudged
    assert "bound" not in unjudged


def test_the_pool_note_calls_the_ndcg_prime_gap_the_effect_of_dropping_unjudged_chunks():
    condensed = POOL_NOTE.split("`ndcg_prime@k`", 1)[1].split("Both are", 1)[0]
    assert "dropped" in condensed
    assert "cost" not in condensed


def test_the_pool_note_cites_sakai_2007_by_its_entry_in_the_methodology_references():
    from pathlib import Path

    assert "(Sakai 2007, full entry under References in `evals/methodology.md`)" in POOL_NOTE
    methodology = Path(__file__).resolve().parents[2] / "evals" / "methodology.md"
    references = methodology.read_text(encoding="utf-8").split("\n## References\n", 1)[1]
    assert "- Sakai, T. (2007). *Alternatives to Bpref.*" in references
