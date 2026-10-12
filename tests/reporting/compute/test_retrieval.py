"""Tests for `reporting.compute.retrieval`, on runs written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.check import Doc
from crossvenue_rag.reporting.compute import retrieval
from crossvenue_rag.reporting.compute.estimates import Estimate
from crossvenue_rag.reporting.run_ids import GOLD_BEFORE_AUGMENTATION, PVALUE_RUN, RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import (
    complete,
    write_csv,
    write_gold,
    write_retrieval_run,
    write_sources,
)


def test_methods_are_the_rows_of_the_summary(tmp_path):
    rows = [{"method": m, "recall@10": "0.5"} for m in ("bm25", "dense", "hybrid")]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, summary=rows)
    assert retrieval.method_count(Sources(tmp_path)) == 3


def test_category_metric_reads_by_category(tmp_path):
    rows = [
        {"category": "comparison", "method": "bm25", "recall@10": "0.465"},
        {"category": "fact", "method": "bm25", "recall@10": "0.9"},
    ]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, by_category=rows)
    src = Sources(tmp_path)
    assert retrieval.category_metric(src, "comparison", "bm25", "recall@10") == 0.465


def test_runs_agree_on_metrics_whatever_their_timing(tmp_path):
    write_retrieval_run(
        tmp_path, "a", summary=[{"method": "bm25", "recall@10": "0.5", "p50_ms": "1"}]
    )
    write_retrieval_run(
        tmp_path, "b", summary=[{"method": "bm25", "recall@10": "0.5", "p50_ms": "2"}]
    )
    write_retrieval_run(
        tmp_path, "c", summary=[{"method": "bm25", "recall@10": "0.6", "p50_ms": "1"}]
    )
    src = Sources(tmp_path)
    [row] = retrieval.metric_rows(src, "a")
    assert row["method"] == "bm25" and row["recall@10"] == "0.5" and "p50_ms" not in row
    assert retrieval.runs_agree_on_metrics(src, "a", "b")
    assert not retrieval.runs_agree_on_metrics(src, "a", "c")


def test_k_and_the_pool_depth_are_read_from_the_run_config(tmp_path):
    config = {"retrieval": {"k": 10}, "eval": {"pool_depth_per_method": 20}}
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config=config)
    src = Sources(tmp_path)
    assert retrieval.top_k(src) == 10
    assert retrieval.pool_depth(src) == 20
    assert retrieval.first_rank_below_top_k(src) == 11
    assert retrieval.at_top_k(src, "off_venue") == "off_venue@10"


def test_the_deepest_cutoff_of_another_run_is_read_from_its_own_config(tmp_path):
    write_retrieval_run(tmp_path, RETRIEVAL_RUN)
    write_retrieval_run(tmp_path, "other", config={"eval": {"k_values": [1, 5]}})
    src = Sources(tmp_path)
    assert (retrieval.top_k(src), retrieval.top_k(src, "other")) == (10, 5)


# A comparison question naming both venues, one naming neither and a fact
# question; bm25 and venue_split retrieve differently on the first.
QUESTIONS = [
    {"id": "cv-0001", "question": "binance or bybit?", "gold_chunk_ids": ["c1", "c9"]},
    {"id": "cv-0002", "question": "which is faster?", "gold_chunk_ids": ["c2"]},
    {"id": "cv-0003", "question": "a binance ban", "venues": ["binance"], "category": "fact"},
]


def per_query_rows():
    rows = []
    for m in retrieval.METHODS:
        for qid, retrieved, covered, total in (
            ("cv-0001", "c1 c5", "0.0", "200"),
            ("cv-0002", "c2", "1.0", "100"),
            ("cv-0003", "c3", "", "50"),
        ):
            if m == "venue_split" and qid == "cv-0001":
                covered, total = "1.0", "300"
            if m == "bm25" and qid == "cv-0001":
                retrieved = "c1 c9"
            rows.append(
                {
                    "question_id": qid,
                    "method": m,
                    "retrieved_ids": retrieved,
                    "venue_coverage@10": covered,
                    "total_ms": total,
                    "rerank_ms": str(int(total) // 2),
                }
            )
    return rows


def retrieval_fixture(root):
    write_gold(root, QUESTIONS)
    old = [{**QUESTIONS[0], "gold_chunk_ids": ["c1"]}, *QUESTIONS[1:]]
    write_gold(root, old, path=GOLD_BEFORE_AUGMENTATION)
    write_sources(root, ["binance", "bybit"])
    steps = {f"{step}_p50_ms": "50" for step in ("encode", "search", "fuse", "rerank")}
    summary = [{"method": m, "p50_ms": "2.5", **steps} for m in retrieval.METHODS]
    write_retrieval_run(root, RETRIEVAL_RUN, summary=summary, per_query=per_query_rows())
    return Sources(root)


def test_summary_metric_and_step_p50_read_the_run(tmp_path):
    src = retrieval_fixture(tmp_path)
    assert retrieval.summary_metric(src, "bm25", "p50_ms") == 2.5
    assert retrieval.step_p50(src, "venue_split", "rerank") == 50


def test_coverage_is_counted_on_the_comparison_questions(tmp_path):
    src = retrieval_fixture(tmp_path)
    assert retrieval.coverage_counts(src, "venue_split") == (2, 2)
    assert retrieval.coverage_counts(src, "hybrid_rerank") == (1, 2)
    assert retrieval.covered_share(src, "hybrid_rerank") == 0.5
    # cv-0001 names both venues in its text, cv-0002 neither.
    assert retrieval.coverage_by_detection(src) == (1, 1)


def test_split_p50_is_over_the_questions_venue_split_splits(tmp_path):
    src = retrieval_fixture(tmp_path)
    assert retrieval.split_p50(src, "venue_split") == 300
    assert retrieval.split_p50(src, "hybrid_rerank") == 200


def test_recall_changes_from_the_gold_set_before_augmentation(tmp_path):
    src = retrieval_fixture(tmp_path)
    assert retrieval.recall_against(src, "dense", src.questions) == 0.75
    assert retrieval.recall_against(src, "dense", src.questions_before_augmentation) == 1.0
    dense = next(r for r in retrieval.recall_before_after(src) if r.method == "dense")
    assert (dense.before, dense.after, dense.change) == (1.0, 0.75, -0.25)
    drops = retrieval.recall_drops(src)
    assert drops == [0.0, 0.25, 0.25, 0.25, 0.25, 0.25]
    assert retrieval.recall_change_range(src) == (0.0, -0.25)
    assert not retrieval.recall_lower_for_every_method(src)


def test_a_question_the_scored_gold_set_does_not_label_is_left_out(tmp_path):
    src = retrieval_fixture(tmp_path)
    # Before augmentation cv-0002 had no gold: recall_against leaves it out of the mean.
    old = [QUESTIONS[0], {**QUESTIONS[1], "gold_chunk_ids": []}, QUESTIONS[2]]
    write_gold(tmp_path, old, path=GOLD_BEFORE_AUGMENTATION)
    assert retrieval.recall_against(src, "dense", src.questions_before_augmentation) == 0.5


def test_paired_comparisons_are_read_from_the_p_value_run(tmp_path):
    row = {
        "a": "venue_split",
        "b": "hybrid_rerank",
        "metric": "recall@10",
        "scope": "comparison",
        "diff": "0.1",
        "low": "-0.05",
        "high": "0.2",
        "mcnemar_p": "0.0004",
    }
    write_retrieval_run(tmp_path, PVALUE_RUN, paired=[row])
    src = Sources(tmp_path)
    cell = ("venue_split", "hybrid_rerank", "recall@10", "comparison")
    assert retrieval.paired_interval(src, *cell) == Estimate(0.1, -0.05, 0.2)
    assert retrieval.paired_includes_zero(src, *cell)
    assert retrieval.paired_p_below(src, *cell, "mcnemar_p")
    assert retrieval.excluded_loss(src, *cell) == 0.05


def test_metric_range_spans_the_methods_on_one_category(tmp_path):
    rows = [
        {"category": "comparison", "method": "bm25", "unjudged@10": "0.2"},
        {"category": "comparison", "method": "dense", "unjudged@10": "0.05"},
        {"category": "comparison", "method": "hybrid", "unjudged@10": "0.9"},
        {"category": "fact", "method": "bm25", "unjudged@10": "0.0"},
    ]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, by_category=rows)
    src = Sources(tmp_path)
    assert retrieval.metric_range(src, "comparison", ("bm25", "dense"), "unjudged@10") == (
        0.05,
        0.2,
    )


def test_the_round_1_pool_count_is_checked_in_the_text():
    doc = Doc.from_text("README.md", "| A four-method labelling pool inflates recall |\n")
    doc.check("A {}-method labelling pool", str(len(retrieval.ROUND1_POOL)))
    doc.audit()
    assert doc.problems == []


def test_a_row_or_question_the_saved_tables_lack_raises_a_value_error(tmp_path):
    src = retrieval_fixture(tmp_path)
    by_category = complete("by_category", [{"category": "fact", "method": "bm25"}])
    write_csv(tmp_path / "results" / RETRIEVAL_RUN / "by_category.csv", by_category)
    other = {"a": "dense", "b": "bm25", "metric": "recall@10", "scope": "comparison"}
    write_retrieval_run(tmp_path, PVALUE_RUN, paired=[other])
    cell = ("venue_split", "hybrid_rerank", "recall@10", "comparison")
    with pytest.raises(ValueError, match=r"by_category\.csv has no"):
        retrieval.category_metric(src, "comparison", "bm25", "recall@10")
    with pytest.raises(ValueError, match=r"summary\.csv has no 'oracle'"):
        retrieval.summary_metric(src, "oracle", "p50_ms")
    with pytest.raises(ValueError, match="has no 'oracle'"):
        retrieval.step_p50(src, "oracle", "rerank")
    with pytest.raises(ValueError, match="has no 'embed'"):
        retrieval.step_p50(src, "bm25", "embed")
    with pytest.raises(ValueError, match=r"paired\.csv has no"):
        retrieval.paired_interval(src, *cell)
    write_gold(tmp_path, QUESTIONS[1:])
    with pytest.raises(ValueError, match="the gold set has no 'cv-0001'"):
        retrieval.coverage_counts(Sources(tmp_path), "bm25")
