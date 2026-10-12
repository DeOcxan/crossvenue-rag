"""Tests for `reporting.compute.retrieval_tables`, on retrieval runs written under tmp_path."""

from __future__ import annotations

from typing import Any

import pytest

from crossvenue_rag.config_eval import COMPARISON_SCOPE, LABELLED_COMPARISON_SCOPE
from crossvenue_rag.models import RETRIEVAL_STEPS
from crossvenue_rag.reporting.compute import retrieval_tables as tables
from crossvenue_rag.reporting.compute.estimates import Estimate
from crossvenue_rag.reporting.compute.retrieval import FIRST_STAGE, METHODS
from crossvenue_rag.reporting.run_ids import (
    GOLD_BEFORE_AUGMENTATION,
    NDCG_PRIME_RUN,
    PVALUE_RUN,
    RETRIEVAL_RUN,
)
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_gold, write_retrieval_run, write_sources

# Three comparison questions naming both venues, in the text two, none and one of
# them, and a fact question. Augmentation adds c9 to the first one's gold.
QUESTIONS: list[dict[str, Any]] = [
    {"id": "cv-0001", "question": "binance or bybit?", "gold_chunk_ids": ["c1", "c9"]},
    {"id": "cv-0002", "question": "which is faster?", "gold_chunk_ids": ["c2"]},
    {"id": "cv-0004", "question": "a bybit question"},
    {"id": "cv-0003", "question": "a binance ban", "venues": ["binance"], "category": "fact"},
]


def retrieved(method, qid):
    """venue_split finds both gold chunks of cv-0001, hybrid_rerank one of them and
    cv-0002's, the other methods only c1."""
    if qid == "cv-0001":
        return "c1 c9" if method == "venue_split" else "c1 c5"
    if qid == "cv-0002" and method in ("venue_split", "hybrid_rerank"):
        return "c2"
    return "c7"


def covered(method, qid):
    """The oracle covers every comparison question, venue_split only cv-0001."""
    if qid == "cv-0003":
        return ""
    if method == "venue_split_oracle" or (method == "venue_split" and qid == "cv-0001"):
        return "1.0"
    return "0.0"


def per_query():
    return [
        {
            "question_id": q["id"],
            "method": m,
            "retrieved_ids": retrieved(m, q["id"]),
            "venue_coverage@10": covered(m, q["id"]),
            **{f"{s}_ms": j + 1 for j, s in enumerate(RETRIEVAL_STEPS)},
        }
        for m in METHODS
        for q in QUESTIONS
    ]


def by_category():
    rows = []
    for i, m in enumerate(METHODS):
        for category, labelled in (("comparison", 2), ("fact", 1), ("multihop", 7)):
            rows.append(
                {
                    "category": category,
                    "method": m,
                    "recall@10": 0.1 * i,
                    "ndcg@10": 0.2,
                    "venue_coverage@10": 0.3,
                    "labelled_questions": labelled,
                }
            )
    return rows


@pytest.fixture
def src(tmp_path):
    write_gold(tmp_path, QUESTIONS)
    old = [{**QUESTIONS[0], "gold_chunk_ids": ["c1"]}, *QUESTIONS[1:]]
    write_gold(tmp_path, old, path=GOLD_BEFORE_AUGMENTATION)
    write_sources(tmp_path, ["binance", "bybit"])
    summary = [
        {
            "method": m,
            "p50_ms": 10 * (i + 1),
            "p95_ms": 20 * (i + 1),
            **{f"{s}_p50_ms": j + 1 for j, s in enumerate(RETRIEVAL_STEPS)},
        }
        for i, m in enumerate(METHODS)
    ]
    write_retrieval_run(
        tmp_path, RETRIEVAL_RUN, summary=summary, per_query=per_query(), by_category=by_category()
    )
    ndcg_by_category = [
        {"category": "comparison", "method": m, "ndcg_prime@10": 0.25} for m in METHODS
    ]
    ndcg_summary = [{"method": m, "ndcg@10": 0.5004, "ndcg_prime@10": 0.5096} for m in METHODS]
    write_retrieval_run(
        tmp_path, NDCG_PRIME_RUN, summary=ndcg_summary, by_category=ndcg_by_category
    )
    return Sources(tmp_path)


def test_a_comparison_row_joins_both_runs_and_the_coverage_counts(src):
    rows = tables.comparison_rows(src)
    assert [r.method for r in rows] == list(METHODS)
    by_method = {r.method: r for r in rows}
    dense = by_method["dense"]
    assert (dense.recall, dense.ndcg, dense.ndcg_prime, dense.coverage) == pytest.approx(
        (0.1, 0.2, 0.25, 0.3)
    )
    assert (dense.covered, dense.questions) == (0, 3)
    assert (by_method["venue_split"].covered, by_method["venue_split"].questions) == (1, 3)
    oracle = by_method["venue_split_oracle"]
    assert (oracle.covered, oracle.questions) == (3, 3)


def test_oracle_gains_are_split_by_the_venues_the_text_names(src):
    # cv-0002 names no venue, cv-0004 one; the fact question names a single venue.
    assert tables.oracle_gains(src) == tables.OracleGains(2, 1, 1)


def test_the_recall_gap_widens_with_the_augmented_gold(src):
    assert tables.recall_gap(src, augmented=False) == 0.0
    assert tables.recall_gap(src, augmented=True) == 0.25


def write_gap(root, diff):
    row = {
        "a": "venue_split",
        "b": "hybrid_rerank",
        "metric": "recall@10",
        "scope": LABELLED_COMPARISON_SCOPE,
        "diff": diff,
        "low": 0.1,
        "high": 0.4,
    }
    write_retrieval_run(root, PVALUE_RUN, paired=[row])


def test_the_rescored_gap_agrees_with_the_p_value_run_at_three_decimals(src, tmp_path):
    write_gap(tmp_path, 0.2504)
    assert tables.rescored_gap_agrees(Sources(tmp_path))
    write_gap(tmp_path, 0.2)
    assert not tables.rescored_gap_agrees(Sources(tmp_path))


def test_reranker_rows_compare_hybrid_rerank_with_each_first_stage_method(tmp_path):
    rows = []
    for i, b in enumerate(reversed(FIRST_STAGE)):
        for metric, scope in (
            ("recall@10", LABELLED_COMPARISON_SCOPE),
            ("venue_coverage@10", COMPARISON_SCOPE),
        ):
            rows.append(
                {
                    "a": "hybrid_rerank",
                    "b": b,
                    "metric": metric,
                    "scope": scope,
                    "diff": i,
                    "low": i - 1,
                    "high": i + 1,
                    "mcnemar_p": 0.01,
                    "holm_p": 0.02,
                }
            )
    write_retrieval_run(tmp_path, PVALUE_RUN, paired=rows)
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config={"eval": {"k_values": [1, 10]}})
    result = tables.reranker_rows(Sources(tmp_path))
    assert [r.b for r in result] == list(reversed(FIRST_STAGE))
    assert result[1] == tables.RerankerRow(
        "dense", Estimate(1.0, 0.0, 2.0), 0.02, Estimate(1.0, 0.0, 2.0), 0.01, 0.02
    )


def test_condensed_ndcg_rounds_before_it_subtracts(src):
    row = tables.condensed_ndcg(src)[0]
    assert row.method == "bm25"
    # 0.510 - 0.500, where the unrounded gain is 0.0092.
    assert row.difference == pytest.approx(0.010)
    assert (row.plain, row.condensed) == (0.5004, 0.5096)


def test_condensed_gains_stay_below_the_bound_unless_one_method_exceeds_it(src, tmp_path):
    assert tables.condensed_gains_below_bound(src)
    summary = [{"method": m, "ndcg@10": 0.5, "ndcg_prime@10": 0.5} for m in METHODS]
    summary[2]["ndcg_prime@10"] = 0.5 + tables.NDCG_PRIME_GAIN_BOUND
    write_retrieval_run(tmp_path, NDCG_PRIME_RUN, summary=summary)
    assert not tables.condensed_gains_below_bound(Sources(tmp_path))


def test_a_latency_row_has_each_step_then_the_whole(src):
    rows = tables.latency_rows(src)
    assert [m for m, _ in rows] == list(METHODS)
    assert rows[1] == ("dense", [1.0, 2.0, 3.0, 4.0, 20.0, 40.0])


def test_a_latency_step_is_the_value_the_runs_own_table_gives(tmp_path, src):
    # summary.csv holds 27.5 for encode, as report.md's one decimal would; per_query 27.47.
    rows = per_query()
    for row in rows:
        row["encode_ms"] = 27.47
    summary = [
        {"method": m, "p50_ms": 1, "p95_ms": 2, **{f"{s}_p50_ms": 27.5 for s in RETRIEVAL_STEPS}}
        for m in METHODS
    ]
    write_retrieval_run(
        tmp_path, RETRIEVAL_RUN, summary=summary, per_query=rows, by_category=by_category()
    )
    assert tables.latency_rows(Sources(tmp_path))[0][1][0] == 27.5


def test_labelled_sizes_follow_the_categories_given_and_span_their_range(src):
    categories = ("comparison", "fact", "multihop")
    assert tables.labelled_sizes(src, categories) == [2, 1, 7]
    assert tables.labelled_size_range(src, categories) == (1, 7)


def test_a_method_or_category_a_run_lacks_raises_a_value_error(src, tmp_path):
    with pytest.raises(ValueError, match=r"by_category.csv has no \('multi', 'bm25'\)"):
        tables.labelled_sizes(src, ["multi"])
    ndcg = {"method": "bm25", "ndcg@10": 0.5, "ndcg_prime@10": 0.5}
    ndcg_by_category = {"category": "comparison", "method": "bm25", "ndcg_prime@10": 0.5}
    write_retrieval_run(tmp_path, NDCG_PRIME_RUN, summary=[ndcg], by_category=[ndcg_by_category])
    rows = [r for r in per_query() if r["method"] != "venue_split"]
    summary = {"method": "bm25", **{f"{s}_p50_ms": 1 for s in RETRIEVAL_STEPS}}
    write_retrieval_run(
        tmp_path, RETRIEVAL_RUN, summary=[summary], per_query=rows, by_category=by_category()
    )
    lacking = Sources(tmp_path)
    with pytest.raises(ValueError, match=r"summary\.csv has no 'dense'"):
        tables.condensed_ndcg(lacking)
    with pytest.raises(ValueError, match=r"summary\.csv has no 'dense'"):
        tables.condensed_gains_below_bound(lacking)
    with pytest.raises(ValueError, match=r"summary\.csv has no 'dense'"):
        tables.latency_rows(lacking)
    with pytest.raises(ValueError, match=r"by_category.csv has no \('comparison', 'dense'\)"):
        tables.comparison_rows(lacking)
    with pytest.raises(ValueError, match=r"per_query.csv has no \('venue_split', 'cv-0001'\)"):
        tables.oracle_gains(lacking)
