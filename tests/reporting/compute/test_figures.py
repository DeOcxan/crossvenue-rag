"""Tests for `reporting.compute.figures`, on a retrieval run written under tmp_path."""

from __future__ import annotations

from typing import Any

import pytest

from crossvenue_rag.evaluation.metrics import recall_at_k
from crossvenue_rag.reporting.compute import figures
from crossvenue_rag.reporting.run_ids import RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks, write_gold, write_retrieval_run


def fixture(root):
    questions: list[dict[str, Any]] = [
        {"id": "cv-0001", "gold_chunk_ids": ["a"]},
        {"id": "cv-0002"},
        {"id": "cv-0003", "category": "fact", "gold_chunk_ids": ["x"]},
    ]
    write_gold(root, questions)
    rows = [
        # Top 2: a is found, and one of the two venues is not named.
        ("bm25", "cv-0001", "a b c", "binance okx okx"),
        ("bm25", "cv-0002", "d e", "okx bybit"),
        ("dense", "cv-0003", "x", "binance"),
    ]
    per_query = [
        {"method": m, "question_id": q, "retrieved_ids": ids, "retrieved_venues": venues}
        for m, q, ids, venues in rows
    ]
    config = {"eval": {"k_values": [1, 2], "bootstrap_resamples": 50, "bootstrap_seed": 1}}
    write_retrieval_run(root, RETRIEVAL_RUN, config=config, per_query=per_query)
    return Sources(root)


def test_rows_are_grouped_by_method_for_one_category(tmp_path):
    rows = figures.category_per_query(fixture(tmp_path), "comparison")
    assert [r.question_id for r in rows["bm25"]] == ["cv-0001", "cv-0002"]
    assert rows["dense"] == []


def test_a_metric_is_scored_at_the_top_k_on_questions_with_gold_only(tmp_path):
    src = fixture(tmp_path)
    rows = figures.category_per_query(src, "comparison")["bm25"]
    assert figures.labelled_values(src, rows, recall_at_k) == [1.0]


def test_off_venue_shares_are_averaged_over_the_rows(tmp_path):
    src = fixture(tmp_path)
    rows = figures.category_per_query(src, "comparison")["bm25"]
    off, via = figures.off_venue_shares(src, rows)
    assert off == pytest.approx(0.5)
    assert via == 0.0


def test_a_run_that_saved_no_venues_has_its_venues_read_from_the_corpus(tmp_path):
    write_gold(tmp_path, [{"id": "cv-0001"}])
    write_chunks(tmp_path, [make_chunk("a", "binance", "x"), make_chunk("b", "okx", "y")])
    per_query = [{"method": "bm25", "question_id": "cv-0001", "retrieved_ids": "a b"}]
    config = {"eval": {"k_values": [1, 2]}}
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config=config, per_query=per_query)
    src = Sources(tmp_path)
    rows = figures.category_per_query(src, "comparison")["bm25"]
    assert figures.off_venue_shares(src, rows) == (0.5, 0.0)


def test_the_interval_of_a_mean_holds_the_mean(tmp_path):
    src = fixture(tmp_path)
    interval = figures.bootstrap_mean_interval([0.0, 1.0, 1.0, 0.0], src)
    assert interval.value == 0.5
    assert interval.low <= interval.value <= interval.high


def test_a_question_or_method_the_gold_set_or_methods_lack_raises_a_value_error(tmp_path):
    src = fixture(tmp_path)
    rows = figures.category_per_query(src, "comparison")["bm25"]
    write_gold(tmp_path, [{"id": "cv-0002"}, {"id": "cv-0003"}])
    lacking = Sources(tmp_path)
    with pytest.raises(ValueError, match="the gold set has no 'cv-0001'"):
        figures.category_per_query(lacking, "comparison")
    with pytest.raises(ValueError, match="the gold set has no 'cv-0001'"):
        figures.off_venue_shares(lacking, rows)
    per_query = [{"method": "oracle", "question_id": "cv-0002", "retrieved_ids": "a"}]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, per_query=per_query)
    with pytest.raises(ValueError, match="METHODS has no 'oracle'"):
        figures.category_per_query(Sources(tmp_path), "comparison")
