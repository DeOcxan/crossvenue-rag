"""Tests for `crossvenue_rag.evaluation.comparisons`: primary row and Holm families."""

from __future__ import annotations

import pytest

from crossvenue_rag.evaluation.comparisons import PairedRow, mark_primary_and_adjust
from crossvenue_rag.evaluation.paired import holm
from tests.conftest import assert_round_trip, committed_of_kind

PRIMARY = ["venue_split", "hybrid_rerank", "venue_coverage@10", "comparison"]


def row(a: str, b: str, metric: str, scope: str, p: float) -> PairedRow:
    return PairedRow(
        a=a,
        b=b,
        metric=metric,
        scope=scope,
        n=50,
        mean_a=0.5,
        mean_b=0.4,
        diff=0.1,
        low=0.0,
        high=0.2,
        better=5,
        worse=2,
        ties=43,
        mcnemar_p=None,
        randomisation_p=p,
    )


def test_primary_row_is_unadjusted_and_secondaries_are_holm_adjusted_per_family():
    rows = [
        row("venue_split", "hybrid_rerank", "venue_coverage@10", "comparison", 0.0003),
        row("hybrid_rerank", "hybrid", "venue_coverage@10", "comparison", 0.33),
        row("hybrid_rerank", "dense", "venue_coverage@10", "comparison", 0.064),
        row("hybrid_rerank", "hybrid", "recall@10", "labelled comparison", 0.18),
        row("hybrid_rerank", "dense", "recall@10", "labelled comparison", 0.0002),
        row("venue_split", "hybrid_rerank", "recall@10", "labelled comparison", 0.11),
    ]
    rows = mark_primary_and_adjust(rows, PRIMARY)

    assert rows[0].role == "primary" and rows[0].holm_p is None
    assert all(r.role == "secondary" for r in rows[1:])
    # The primary row is not in its family: the coverage family is the other two.
    assert [rows[1].holm_p, rows[2].holm_p] == pytest.approx(holm([0.33, 0.064]))
    # The recall family is separate, and includes venue_split's recall row.
    assert [r.holm_p for r in rows[3:]] == pytest.approx(holm([0.18, 0.0002, 0.11]))


def test_without_a_declared_primary_every_row_is_secondary():
    rows = [
        row("a", "b", "recall@10", "labelled", 0.02),
        row("c", "d", "recall@10", "labelled", 0.03),
    ]
    rows = mark_primary_and_adjust(rows, [])
    assert {r.role for r in rows} == {"secondary"}
    assert [r.holm_p for r in rows] == pytest.approx([0.04, 0.04])


def test_off_venue_comparisons_pair_hybrid_rerank_with_each_first_stage_method():
    from crossvenue_rag.config import Config
    from crossvenue_rag.evaluation.comparisons import OFF_VENUE_PAIRS, off_venue_comparisons
    from crossvenue_rag.evaluation.evaluator import Retrieval, score
    from crossvenue_rag.models import Question

    questions = [
        Question(f"q{i}", "x", "gold", ["bybit", "binance"], "comparison", category="comparison")
        for i in range(6)
    ]
    venues = {
        "hybrid_rerank": ["bybit", "binance", "bybit"],
        "bm25": ["okx", "bybit", "binance"],
        "dense": ["ccxt", "okx", "okx"],
        "hybrid": ["bybit", "okx", "binance"],
    }
    runs = [
        score(q, Retrieval(m, [f"c{j}" for j in range(3)], v), [1, 3])
        for q in questions
        for m, v in venues.items()
    ]
    cfg = Config(eval={"k_values": [1, 3]})
    rows = off_venue_comparisons(cfg, runs, questions)
    assert {(r.a, r.b) for r in rows} == set(OFF_VENUE_PAIRS)
    assert {(r.metric, r.scope) for r in rows} == {
        ("off_venue@3", "comparison"),
        ("off_venue@1", "comparison"),
    }
    dense3 = next(r for r in rows if r.b == "dense" and r.metric == "off_venue@3")
    assert (dense3.mean_a, dense3.mean_b) == (0.0, pytest.approx(2 / 3))
    top1 = next(r for r in rows if r.b == "bm25" and r.metric == "off_venue@1")
    # Lower is better for an off-venue share, so a smaller share counts as better.
    # Six questions are below 50 (methodology section 7): intervals, no p-values.
    assert top1.role == "descriptive"
    assert top1.mcnemar_p is None and top1.randomisation_p is None and top1.holm_p is None
    assert not top1.higher_is_better
    assert (top1.better, top1.worse) == (6, 0)


def test_paired_table_states_each_rows_direction():
    from crossvenue_rag.evaluation.comparisons import paired_table

    down = row("hybrid_rerank", "bm25", "off_venue@10", "comparison", 0.01)
    down = down.model_copy(update={"higher_is_better": False})
    table = paired_table([row("a", "b", "recall@10", "labelled", 0.2), down])
    assert "better is" in table.splitlines()[0]
    assert "| higher |" in table.splitlines()[2] and "| lower |" in table.splitlines()[3]


@pytest.mark.parametrize(("size", "tested"), [(49, False), (50, True)])
def test_a_scope_below_50_questions_reports_intervals_only(size: int, *, tested: bool):
    from crossvenue_rag.config import Config
    from crossvenue_rag.evaluation.comparisons import off_venue_comparisons
    from crossvenue_rag.evaluation.evaluator import Retrieval, score
    from crossvenue_rag.models import Question

    questions = [
        Question(f"q{i}", "x", "gold", ["bybit"], "fact", category="adversarial")
        for i in range(size)
    ]
    venues = {"hybrid_rerank": ["bybit", "okx"], "bm25": ["okx", "okx"]}
    runs = [
        score(q, Retrieval(m, ["c0", "c1"], v), [1, 2])
        for q in questions
        for m, v in venues.items()
    ]
    rows = off_venue_comparisons(Config(eval={"k_values": [1, 2]}), runs, questions)
    assert rows and all(r.n == size for r in rows)
    for r in rows:
        assert r.low <= r.diff <= r.high  # the interval is always there
        assert (r.randomisation_p is not None) is tested
        assert (r.holm_p is not None) is tested
        assert r.role == ("secondary" if tested else "descriptive")


@pytest.mark.parametrize("run", committed_of_kind("paired.csv"), ids=lambda p: p.name)
def test_every_committed_retrieval_paired_table_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.evaluation.comparisons import read_paired

    assert_round_trip(run, "paired.csv", read_paired(run / "paired.csv"), tmp_path)


def test_runs_scored_at_other_cutoffs_are_a_value_error_naming_the_metric(cfg):
    from crossvenue_rag.evaluation.comparisons import off_venue_comparisons, paired_comparisons
    from tests.evaluation.factories import evaluated

    _, questions, runs = evaluated(cfg)
    cfg.eval.k_values = [1, 10]
    cfg.eval.paired_comparisons = [("hybrid", "bm25")]
    with pytest.raises(ValueError, match="has no recall@10 in metrics"):
        paired_comparisons(cfg, runs, questions)
    for q in questions:
        q.category = "comparison"
    with pytest.raises(ValueError, match="has no off_venue@10 in off_venue_metrics"):
        off_venue_comparisons(cfg, runs, questions)
