"""Tests for `crossvenue_rag.reporting.compute.retrieval_off_venue`: off-venue measures."""

from __future__ import annotations

from typing import Any

import pytest

from crossvenue_rag.reporting.compute import retrieval_off_venue as off_venue
from crossvenue_rag.reporting.compute.retrieval import METHODS, QuestionSplit
from crossvenue_rag.reporting.run_ids import RETRIEVAL_RUN
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_gold, write_retrieval_run

COMPARISON = [f"cv-{i:04d}" for i in range(1, 51)]
ADVERSARIAL = ["cv-0101", "cv-0102"]
# The comparison questions on which bm25's first result is off venue.
BM25_OFF = 20
# The comparison questions on which dense's first result is a CCXT page.
DENSE_VIA_CCXT = 5


def first_venue(method, qid):
    """The venue of `method`'s first result on `qid`; the second is always binance."""
    if qid in ADVERSARIAL:
        # hybrid_rerank and venue_split are off venue on one adversarial question.
        off = qid == ADVERSARIAL[0] and method in ("hybrid_rerank", "venue_split")
        return "okx" if off else "binance"
    i = COMPARISON.index(qid)
    if method == "bm25" and i < BM25_OFF:
        return "okx"
    if method == "dense" and i < DENSE_VIA_CCXT:
        return "ccxt"
    return "bybit"


@pytest.fixture
def src(tmp_path):
    """50 comparison and 2 adversarial questions, every method's top 2 saved."""
    questions: list[dict[str, Any]] = [{"id": q} for q in COMPARISON]
    questions += [
        {"id": q, "venues": ["binance"], "category": "adversarial", "qtype": "fact"}
        for q in ADVERSARIAL
    ]
    write_gold(tmp_path, questions)
    rows = [
        {
            "question_id": q,
            "method": m,
            "retrieved_ids": "x z" if m == "venue_split_oracle" else "x y",
            "retrieved_venues": f"{first_venue(m, q)} binance",
        }
        for m in METHODS
        for q in COMPARISON + ADVERSARIAL
    ]
    config = {
        "retrieval": {"k": 10},
        "eval": {"k_values": [1, 10], "bootstrap_resamples": 20000, "bootstrap_seed": 1},
    }
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config=config, per_query=rows)
    return Sources(tmp_path)


def test_measure_rows_give_each_method_its_shares_and_first_rank_counts(src):
    rows = {r.method: r for r in off_venue.measure_rows(src, "comparison")}
    assert list(rows) == list(METHODS)

    def values(r: off_venue.MeasureRow) -> tuple[float, float, int, int, int]:
        return (r.share, r.via, r.off_first, r.via_first, r.questions)

    # Two results each: an off-venue first result is half the top k.
    assert values(rows["bm25"]) == pytest.approx((0.2, 0.0, BM25_OFF, 0, 50))
    assert values(rows["dense"]) == pytest.approx((0.0, 0.05, 0, DENSE_VIA_CCXT, 50))
    assert off_venue.measure_names(src) == [
        "off_venue@10",
        "via_ccxt@10",
        "off_venue@1",
        "via_ccxt@1",
    ]


def test_same_top_k_counts_questions_retrieving_the_baseline_ids(src):
    assert off_venue.same_top_k(src, "venue_split", "comparison") == 50
    assert off_venue.same_top_k(src, "venue_split_oracle", "comparison") == 0


def test_venue_split_and_the_oracle_equal_the_baseline_only_with_equal_means(src):
    assert off_venue.equal_to_baseline(src, "comparison")
    # The oracle's first result is on venue where hybrid_rerank's is not.
    assert not off_venue.equal_to_baseline(src, "adversarial")


def test_discordant_counts_questions_the_baseline_wins_loses_and_ties(src):
    assert off_venue.discordant(src, "bm25", "off_venue@1", "comparison") == QuestionSplit(
        BM25_OFF, 0, 30
    )
    assert off_venue.discordant(src, "bm25", "off_venue@1", "adversarial") == QuestionSplit(0, 1, 1)


def test_a_comparison_the_baseline_wins_everywhere_is_significant_and_never_worse(src):
    assert off_venue.significant_against(src, "bm25", "comparison")
    assert off_venue.off_venue_share_below_bound(src, "bm25", "comparison")
    assert off_venue.never_worse(src, "bm25", "comparison")
    assert not off_venue.significant_against(src, "dense", "comparison")
    assert not off_venue.never_worse(src, "bm25", "adversarial")


def test_only_intervals_against_other_methods_must_include_zero(src):
    assert off_venue.others_include_zero(src, "bm25", "comparison")
    assert not off_venue.others_include_zero(src, "dense", "comparison")


def test_a_scope_below_the_tested_size_is_descriptive_without_p_values(src):
    assert off_venue.scope_questions(src, "comparison") == 50
    assert off_venue.scope_questions(src, "adversarial") == 2
    assert off_venue.scope_descriptive(src, "adversarial")
    assert not off_venue.scope_descriptive(src, "comparison")
    with pytest.raises(ValueError, match="expected a p-value"):
        off_venue.off_venue_holm_p(src, "bm25", "off_venue@1", "adversarial")


def test_comparison_rows_list_each_scope_and_method_with_p_values_unless_descriptive(src):
    rows = off_venue.off_venue_comparison_rows(src)
    assert [(r.b, r.scope) for r in rows] == [
        (b, s) for s in ("comparison", "adversarial") for b in ("bm25", "dense", "hybrid")
    ]
    first = rows[0]
    assert (first.b, first.descriptive) == ("bm25", False)
    assert first.share.value == pytest.approx(-0.2)
    assert first.first.value == pytest.approx(-BM25_OFF / 50)
    assert None not in (first.share_holm, first.first_mcnemar, first.first_holm)
    assert all(
        r.descriptive and (r.share_holm, r.first_mcnemar, r.first_holm) == (None, None, None)
        for r in rows[3:]
    )


# Fifty comparison questions, so the paired off-venue comparisons get p-values.
# BM25 puts an unnamed exchange first on ten of them, the other three methods a
# CCXT page on five. On the two adversarial questions every method puts an
# unnamed exchange first.
ROUND1_POOL = ("bm25", "dense", "hybrid", "hybrid_rerank")
SPREAD_COMPARISON = [f"cv-c{i:02d}" for i in range(50)]
SPREAD_ADVERSARIAL = ["cv-a1", "cv-a2"]


def spread_first_venue(method: str, qid: str) -> str:
    """The venue of `method`'s first result on `qid` in `off_venue_src`."""
    if qid in SPREAD_ADVERSARIAL:
        return "okx"
    i = SPREAD_COMPARISON.index(qid)
    if method == "bm25":
        return "okx" if i < 10 else "binance"
    return "ccxt" if i < 5 else "bybit"


@pytest.fixture
def off_venue_src(tmp_path):
    questions: list[dict[str, Any]] = [{"id": q} for q in SPREAD_COMPARISON]
    questions += [
        {"id": q, "venues": ["binance"], "category": "adversarial", "qtype": "fact"}
        for q in SPREAD_ADVERSARIAL
    ]
    write_gold(tmp_path, questions)
    rows = [
        {
            "question_id": q,
            "method": m,
            "retrieved_ids": "x y",
            "retrieved_venues": f"{spread_first_venue(m, q)} binance",
        }
        for m in ROUND1_POOL
        for q in SPREAD_COMPARISON + SPREAD_ADVERSARIAL
    ]
    config = {
        "retrieval": {"k": 10},
        "eval": {"k_values": [1, 10], "bootstrap_resamples": 200, "bootstrap_seed": 1},
    }
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config=config, per_query=rows)
    return Sources(tmp_path)


def test_means_are_over_the_questions_of_a_category(off_venue_src):
    size, means = off_venue.off_venue_means(off_venue_src, "bm25", "comparison")
    assert size == 50
    assert means["off_venue@1"] == pytest.approx(0.2)
    assert means["off_venue@10"] == pytest.approx(0.1)


def test_first_rank_counts_are_per_method_in_turn(off_venue_src):
    assert off_venue.first_rank_count(off_venue_src, "off_venue@1", "bm25", "comparison") == (
        10,
        50,
    )
    assert off_venue.first_rank_counts(
        off_venue_src, "via_ccxt@1", "comparison", ("bm25", "dense")
    ) == [
        0,
        50,
        5,
        50,
    ]
    assert off_venue.first_rank_spread(off_venue_src, "off_venue@1", "comparison", ROUND1_POOL) == (
        0,
        10,
    )
    assert off_venue.first_rank_at_most(off_venue_src, "off_venue@1", "comparison", ("dense",), 0)
    assert not off_venue.first_rank_at_most(
        off_venue_src, "off_venue@1", "comparison", ROUND1_POOL, 1
    )
    assert off_venue.first_rank_adjacent(off_venue_src, "off_venue@1", "adversarial", ROUND1_POOL)
    assert not off_venue.first_rank_adjacent(
        off_venue_src, "off_venue@1", "comparison", ROUND1_POOL
    )


def test_top_k_shares_compare_ccxt_with_unnamed_venues(off_venue_src):
    shares = off_venue.top_k_shares(off_venue_src, "comparison", ("bm25", "dense"))
    assert shares == pytest.approx([0.0, 0.1, 0.05, 0.0])
    assert off_venue.via_ccxt_exceeds_off_venue(off_venue_src, "comparison", ("dense", "hybrid"))
    assert not off_venue.via_ccxt_exceeds_off_venue(off_venue_src, "comparison", ("bm25",))


def test_the_paired_rows_give_the_interval_and_p_holm(off_venue_src):
    interval = off_venue.off_venue_interval(off_venue_src, "bm25", "off_venue@1", "comparison")
    assert interval.value == pytest.approx(-0.2)
    assert interval.low <= interval.value <= interval.high
    assert off_venue.off_venue_questions(off_venue_src, "bm25", "off_venue@10", "comparison") == 50
    assert 0 < off_venue.off_venue_holm_p(off_venue_src, "bm25", "off_venue@1", "comparison") <= 1
    # 200 resamples cannot give p below 0.001.
    assert not off_venue.off_venue_share_below_bound(off_venue_src, "bm25", "comparison")


def test_a_first_chunk_the_baseline_recovers_is_a_named_venue_or_a_ccxt_page(off_venue_src):
    # bm25 is off-venue first on ten questions; hybrid_rerank puts CCXT first on five of them.
    assert off_venue.first_rank_recovered(off_venue_src, "bm25", "comparison") == (5, 5)


def test_a_comparison_row_or_question_the_run_lacks_raises_a_value_error(src, tmp_path):
    with pytest.raises(ValueError, match="no row of bm25 on multihop"):
        off_venue.off_venue_means(src, "bm25", "multihop")
    with pytest.raises(ValueError, match="off-venue metrics has no 'recall@1'"):
        off_venue.first_rank_count(src, "recall@1", "bm25", "comparison")
    with pytest.raises(
        ValueError, match="no off-venue comparison of bm25 on off_venue@1 over fact"
    ):
        off_venue.off_venue_questions(src, "bm25", "off_venue@1", "fact")
    with pytest.raises(ValueError, match="no off-venue comparison over fact"):
        off_venue.scope_questions(src, "fact")
    rows = [
        {"question_id": q, "method": m, "retrieved_ids": "x", "retrieved_venues": first_venue(m, q)}
        for m in METHODS
        for q in COMPARISON
        if (m, q) != ("hybrid_rerank", COMPARISON[0])
    ]
    write_retrieval_run(tmp_path, RETRIEVAL_RUN, config={"eval": {"k_values": [1]}}, per_query=rows)
    lacking = Sources(tmp_path)
    with pytest.raises(ValueError, match=r"per_query.csv has no \('cv-0001', 'hybrid_rerank'\)"):
        off_venue.same_top_k(lacking, "venue_split", "comparison")
    with pytest.raises(ValueError, match=r"per_query.csv has no \('hybrid_rerank', 'cv-0001'\)"):
        off_venue.first_rank_recovered(lacking, "bm25", "comparison")
    write_gold(tmp_path, [{"id": q} for q in COMPARISON[1:]])
    with pytest.raises(ValueError, match="the gold set has no 'cv-0001'"):
        off_venue.scored_runs(Sources(tmp_path))
