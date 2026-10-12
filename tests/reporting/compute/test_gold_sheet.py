"""Tests for `reporting.compute.gold_sheet`, on gold sets and a pool run written under tmp_path."""

from __future__ import annotations

from typing import Any

import pytest

from crossvenue_rag.reporting.compute import gold_sheet
from crossvenue_rag.reporting.run_ids import GOLD, GOLD_BEFORE_AUGMENTATION, POOL_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks, write_gold, write_retrieval_run

BEFORE: list[dict[str, Any]] = [
    {"id": "q1", "gold_chunk_ids": ["a"], "venue_coverage": {"binance": "covered",
                                                            "bybit": "missing_from_pool"}},
    {"id": "q2", "category": "fact", "qtype": "fact", "venues": ["binance"],
     "gold_chunk_ids": ["c"]},
    {"id": "q3", "category": "unanswerable", "qtype": "fact", "venues": ["bybit"],
     "is_answerable": False, "expected_behavior": "refuse"},
]  # fmt: skip


def gold_sets(tmp_path, *, drop=False):
    after = [dict(q) for q in BEFORE]
    after[0] = {**after[0], "gold_chunk_ids": ["a", "b"],
                "venue_coverage": {"binance": "covered", "bybit": "covered"}}  # fmt: skip
    if drop:
        after[1] = {**after[1], "gold_chunk_ids": []}
    write_gold(tmp_path, after, GOLD)
    write_gold(tmp_path, BEFORE, GOLD_BEFORE_AUGMENTATION)
    return Sources(tmp_path)


def test_a_category_row_counts_questions_answerable_labelled_and_gold(tmp_path):
    src = gold_sets(tmp_path)
    assert gold_sheet.category_row(src, "comparison") == gold_sheet.CategoryRow(1, 1, 1, 2)
    assert gold_sheet.category_row(src, "unanswerable") == gold_sheet.CategoryRow(1, 0, 0, 0)
    assert gold_sheet.categories(src) == ["comparison", "fact", "unanswerable"]
    assert gold_sheet.answerable_count(src) == 2


def test_questions_are_counted_by_how_many_venues_they_name(tmp_path):
    assert gold_sheet.named_venue_counts(gold_sets(tmp_path)) == gold_sheet.NamedVenueCounts(
        2, 1, 0
    )


def test_augmentation_is_the_gold_added_and_the_coverage_it_changed(tmp_path):
    src = gold_sets(tmp_path)
    assert gold_sheet.added_gold(src) == {"q1": {"b"}}
    assert gold_sheet.removed_gold(src) == 0
    assert gold_sheet.coverage_changes(src) == {("missing_from_pool", "covered"): 1}
    assert gold_sheet.coverage_questions(src) == gold_sheet.CoverageQuestions(1, 1, 2)
    assert gold_sheet.added_gold_count(src) == 1
    assert gold_sheet.coverage_changes_to(src, "covered") == 1
    assert gold_sheet.coverage_change_results(src) == {"covered"}
    assert gold_sheet.coverage_value_count(src) == 2
    assert gold_sheet.coverage_outside_comparison(src) == []


def test_a_gold_chunk_dropped_by_the_later_set_is_counted_as_removed(tmp_path):
    assert gold_sheet.removed_gold(gold_sets(tmp_path, drop=True)) == 1


def test_an_added_chunk_is_ranked_in_the_pool_run(tmp_path):
    src = gold_sets(tmp_path)
    write_chunks(tmp_path, [make_chunk("b", "bybit", "text")])
    per_query = [
        {"question_id": "q1", "method": "venue_split", "retrieved_ids": "x y b"},
        {"question_id": "q1", "method": "bm25", "retrieved_ids": "b"},
    ]
    write_retrieval_run(tmp_path, POOL_RUN, per_query=per_query)
    assert gold_sheet.changed_venue_rank(src, "bybit", "missing_from_pool", "venue_split") == 3
    assert gold_sheet.changed_venue_rank(src, "bybit", "missing_from_pool", "dense") is None

    # Best rank over the methods: 1 (bm25), so no added chunk is below rank 1.
    assert gold_sheet.added_below(src, 1) == 0
    assert gold_sheet.added_below(src, 0) == 1


def test_a_question_with_added_gold_the_earlier_set_lacks_raises_a_value_error(tmp_path):
    gold_sets(tmp_path)
    write_gold(tmp_path, BEFORE[1:], GOLD_BEFORE_AUGMENTATION)
    write_chunks(tmp_path, [make_chunk("b", "bybit", "text")])
    per_query = [{"question_id": "q1", "method": "bm25", "retrieved_ids": "b"}]
    write_retrieval_run(tmp_path, POOL_RUN, per_query=per_query)
    with pytest.raises(ValueError, match="the set before augmentation has no 'q1'"):
        gold_sheet.changed_venue_rank(Sources(tmp_path), "bybit", "missing_from_pool", "bm25")


def test_the_longest_shared_run_counts_a_run_of_no_word_as_one(tmp_path):
    write_gold(
        tmp_path,
        [
            {"id": "q1", "question": "how is the weight limit set", "gold_chunk_ids": ["a"]},
            {"id": "q2", "question": "unrelated words", "gold_chunk_ids": ["a"]},
            {"id": "q3", "question": "no gold", "gold_chunk_ids": []},
        ],
    )
    write_chunks(tmp_path, [make_chunk("a", "binance", "The weight limit set per IP")])
    # "the weight limit set", case-insensitive, is four words.
    assert list(gold_sheet.shared_run_counts(Sources(tmp_path)).items()) == [(1, 1), (4, 1)]


def test_the_longest_gold_run_cuts_quoted_spans(tmp_path):
    write_chunks(tmp_path, [make_chunk("c1", "binance", "the order rate limit is ten per second")])
    write_gold(
        tmp_path,
        [
            {"id": "cv-0001", "question": "is the order rate limit low?", "gold_chunk_ids": ["c1"]},
            {
                "id": "cv-0002",
                "question": 'is "the order rate limit" ten',
                "gold_chunk_ids": ["c1"],
            },
            {"id": "cv-0003", "question": "no gold here at all"},
        ],
    )
    assert gold_sheet.longest_gold_run(Sources(tmp_path)) == 4
