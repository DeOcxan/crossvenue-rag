"""Tests for `reporting.compute.retrieval_earlier`, on runs written under tmp_path."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.dataset.journal import rounds_path
from crossvenue_rag.reporting.compute import retrieval_earlier
from crossvenue_rag.reporting.run_ids import (
    COVERAGE_FAILURE_RUN,
    DEDUP_IDMAP,
    FIRST_GOLD_RUN,
    GOLD_BEFORE_AUGMENTATION,
    PRE_DEDUP_RUN,
    RETRIEVAL_RUN,
)
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import augmentation, make_chunk
from tests.reporting.conftest import (
    write_chunks,
    write_csv,
    write_file,
    write_gold,
    write_retrieval_run,
)

# o1, o2 and o4 hold one text; the top 3 is read, so o4 counts only at rank 3 or above.
IDMAP = {
    "old_ids": 5,
    "map": {"o1": "n1", "o2": "n1", "o3": "n3", "o4": "n1"},
    "old_chunks_sha256": "",
    "to_index_version": "",
}


def copies_fixture(root):
    write_retrieval_run(root, RETRIEVAL_RUN, config={"eval": {"k_values": [1, 3]}})
    write_file(root, DEDUP_IDMAP, json.dumps(IDMAP))
    rows = [
        ("bm25", "q1", "o1 o2 o3 o4"),
        ("bm25", "q2", "o3 o1 o5"),
        ("dense", "q1", "o1 o3 o2"),
        ("dense", "q2", "o1 o2 o4"),
        ("hybrid", "q1", "o3"),
        ("hybrid", "q2", "o5"),
        ("hybrid_rerank", "q1", "o1 o2"),
        ("hybrid_rerank", "q2", "o2 o1"),
    ]
    per_query = [{"method": m, "question_id": q, "retrieved_ids": ids} for m, q, ids in rows]
    write_retrieval_run(root, PRE_DEDUP_RUN, per_query=per_query)
    return Sources(root)


def test_copies_of_one_text_in_the_top_k_are_counted_per_question(tmp_path):
    src = copies_fixture(tmp_path)
    assert retrieval_earlier.copy_slots(src) == {
        "bm25": [1, 0],
        "dense": [1, 2],
        "hybrid": [0, 0],
        "hybrid_rerank": [1, 1],
    }
    assert retrieval_earlier.copy_slot_questions(src) == 2
    assert retrieval_earlier.copy_slot_mean_range(src) == (0.0, 1.5)
    assert retrieval_earlier.copy_slot_means(src) == [0.5, 1.5, 0.0, 1.0]
    assert retrieval_earlier.copy_question_range(src) == (0, 2)
    assert retrieval_earlier.copy_slot_most(src) == 2


def coverage_fixture(root):
    write_retrieval_run(root, RETRIEVAL_RUN, config={"eval": {"k_values": [1, 2]}})
    write_chunks(
        root,
        [
            make_chunk("b1", "binance", "x"),
            make_chunk("b2", "binance", "y"),
            make_chunk("y1", "bybit", "z"),
        ],
    )
    questions = [{"id": "cv-0001"}, {"id": "cv-0002"}, {"id": "cv-0003", "category": "fact"}]
    write_gold(root, questions, path=GOLD_BEFORE_AUGMENTATION)
    rows = [
        ("bm25", "cv-0001", "b1 y1"),
        ("bm25", "cv-0002", "b1 b2 y1"),
        ("bm25", "cv-0003", "y1"),
        ("hybrid_rerank", "cv-0001", "y1 b2"),
        ("hybrid_rerank", "cv-0002", "b1 b2"),
        ("dense", "cv-0001", "b1 b2"),
        ("dense", "cv-0002", "b1 b2"),
        ("hybrid", "cv-0001", "b1 b2"),
        ("hybrid", "cv-0002", "b1 b2"),
    ]
    per_query = [{"method": m, "question_id": q, "retrieved_ids": ids} for m, q, ids in rows]
    write_retrieval_run(root, FIRST_GOLD_RUN, per_query=per_query)
    return Sources(root)


def test_coverage_on_the_first_gold_set_is_recomputed_at_k(tmp_path):
    src = coverage_fixture(tmp_path)
    # bm25 covers cv-0001 only: its bybit chunk on cv-0002 is below the top 2.
    assert retrieval_earlier.first_gold_coverage(src, "bm25") == 0.5
    assert retrieval_earlier.first_gold_covered(src, "bm25") == 1
    assert retrieval_earlier.first_gold_covered(src, "dense") == 0
    assert retrieval_earlier.first_gold_covered_most(src) == 1
    assert retrieval_earlier.first_gold_coverage_equal(src, "bm25", "hybrid_rerank")
    assert not retrieval_earlier.first_gold_coverage_equal(src, "bm25", "dense")


def test_the_unjudged_share_of_the_first_run_is_recomputed_from_the_journal(tmp_path):
    questions = [{"id": "cv-0001"}, {"id": "cv-0002"}, {"id": "cv-0003", "category": "fact"}]
    write_gold(tmp_path, questions, path=GOLD_BEFORE_AUGMENTATION)
    journal = {
        "cv-0001": {"1": {"pool": ["a", "b"], "augmentations": [augmentation(["c"])]}},
        "cv-0003": {"1": {"pool": []}},
    }
    write_file(tmp_path, str(rounds_path(GOLD_BEFORE_AUGMENTATION)), json.dumps(journal))
    rows = [
        # a, b and c were judged, d was not: 1 of 4.
        ("venue_split", "cv-0001", "a b c d"),
        # Not in the journal, so not counted; nor are a fact question and another method.
        ("venue_split", "cv-0002", "x y"),
        ("venue_split", "cv-0003", "x y"),
        ("bm25", "cv-0001", "x y"),
    ]
    per_query = [{"method": m, "question_id": q, "retrieved_ids": ids} for m, q, ids in rows]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config={"eval": {"k_values": [1, 10]}})
    write_retrieval_run(tmp_path, COVERAGE_FAILURE_RUN, per_query=per_query)
    src = Sources(tmp_path)
    assert retrieval_earlier.first_run_unjudged(src, "venue_split") == 0.25


def test_the_gap_on_the_earlier_gold_is_the_paired_row_of_the_run_on_it(tmp_path):
    config = {"eval": {"k_values": [1, 2]}}
    row = {
        "a": "venue_split",
        "b": "hybrid_rerank",
        "metric": "recall@2",
        "scope": "labelled comparison",
        "diff": -0.1,
        "low": -0.2,
        "high": -0.01,
    }
    per_query = [{"method": "bm25", "question_id": "q1", "retrieved_ids": "a b"}]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config=config, per_query=per_query)
    write_retrieval_run(tmp_path, COVERAGE_FAILURE_RUN, paired=[row], per_query=per_query)
    src = Sources(tmp_path)
    gap = retrieval_earlier.earlier_gold_gap(src)
    assert (gap.value, gap.low, gap.high) == (-0.1, -0.2, -0.01)
    assert retrieval_earlier.earlier_gold_run_matches(src)
    other = [{"method": "bm25", "question_id": "q1", "retrieved_ids": "a c"}]
    write_retrieval_run(tmp_path, COVERAGE_FAILURE_RUN, per_query=other)
    assert not retrieval_earlier.earlier_gold_run_matches(Sources(tmp_path))


def test_a_method_question_or_row_the_earlier_runs_lack_raises_a_value_error(tmp_path):
    src = copies_fixture(tmp_path)
    per_query = [{"method": "dense", "question_id": "q1", "retrieved_ids": "o1"}]
    write_retrieval_run(tmp_path, PRE_DEDUP_RUN, per_query=per_query)
    with pytest.raises(ValueError, match=r"per_query\.csv has no 'bm25'"):
        retrieval_earlier.copy_slot_means(Sources(tmp_path))
    write_csv(tmp_path / "results" / PRE_DEDUP_RUN / "per_query.csv", [])
    with pytest.raises(ValueError, match="has no row"):
        retrieval_earlier.copy_slot_questions(Sources(tmp_path))
    other = {"a": "dense", "b": "bm25", "metric": "recall@3", "scope": "comparison"}
    write_retrieval_run(tmp_path, COVERAGE_FAILURE_RUN, paired=[other])
    with pytest.raises(ValueError, match=r"paired\.csv has no"):
        retrieval_earlier.earlier_gold_gap(src)
    src = coverage_fixture(tmp_path)
    write_gold(tmp_path, [{"id": "cv-0002"}], path=GOLD_BEFORE_AUGMENTATION)
    with pytest.raises(ValueError, match=r"v0\.2-gold has no 'cv-0001'"):
        retrieval_earlier.first_gold_coverage(Sources(tmp_path), "bm25")
