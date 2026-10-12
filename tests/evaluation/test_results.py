"""Tests for `crossvenue_rag.evaluation.results`."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.evaluation.evaluator import Evaluator, summarise
from crossvenue_rag.evaluation.results import breakdown_file, run_tables, write_eval_run
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Question
from tests.conftest import (
    InMemoryRetriever,
    assert_round_trip,
    build_corpus,
    committed_of_kind,
    output_defects,
)
from tests.evaluation.factories import evaluated


def test_evaluation_writes_metrics_and_a_config_copy(cfg: Config, tmp_path: Path):
    _, questions, runs = evaluated(cfg)
    summaries = summarise(runs, cfg.eval.k_values)
    run_dir = write_eval_run(cfg, run_tables(cfg, runs, summaries, questions, {}), questions, "v1")
    assert (run_dir / "report.md").exists()
    assert (run_dir / "config.yaml").read_text(encoding="utf-8").strip()
    with (run_dir / "per_query.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(runs)
    assert {r["method"] for r in rows} == set(cfg.eval.methods)
    with (run_dir / "by_venue.csv").open(encoding="utf-8") as fh:
        assert list(csv.DictReader(fh))
    assert (run_dir / "by_qtype.csv").exists()


def test_judged_metrics_reach_the_report(cfg: Config, tmp_path: Path):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    gold = [c.chunk_id for c in chunks if c.venue == "uniswap"][:1]
    questions = [Question("q1", "fee tiers", "gold", ["uniswap"], "fact", gold_chunk_ids=gold)]
    cfg.eval.k_values = [3]
    cfg.eval.methods = ["bm25"]
    top = retriever.retrieve("fee tiers", method="bm25", k=3)
    runs = Evaluator(cfg, retriever).run(
        questions, filter_by_question_venues=False, judged={"q1": {top[0].chunk_id}}
    )
    tables = run_tables(cfg, runs, summarise(runs, [3]), questions, {})
    report = (write_eval_run(cfg, tables, questions, "v1") / "report.md").read_text()
    assert "unjudged@3" in report and "ndcg_prime@3" in report


def test_venue_coverage_and_paired_comparisons_reach_the_report(cfg: Config, tmp_path: Path):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    questions = [
        Question(
            "q1",
            "fee tiers uniswap vs pancakeswap",
            "gold",
            ["uniswap", "pancakeswap"],
            "comparison",
            category="comparison",
            gold_chunk_ids=[c.chunk_id for c in chunks if c.venue == "uniswap"][:1],
        ),
        Question("q2", "rate limits binance", "gold", ["binance"], "fact"),
    ]
    cfg.eval.k_values = [1, 3]
    cfg.eval.paired_comparisons = [("venue_split_oracle", "hybrid_rerank")]
    cfg.eval.bootstrap_resamples = 200
    runs = Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None)
    multi = [r for r in runs if r.question_id == "q1"]
    assert all(set(r.venue_metrics) == {"venue_coverage@1", "venue_coverage@3"} for r in multi)
    assert all(not r.venue_metrics for r in runs if r.question_id == "q2")
    oracle = next(r for r in multi if r.method == "venue_split_oracle")
    assert oracle.venue_metrics["venue_coverage@3"] == 1.0

    summaries = summarise(runs, cfg.eval.k_values)
    assert all(s.multi_venue_questions == 1 for s in summaries)
    run_dir = write_eval_run(cfg, run_tables(cfg, runs, summaries, questions, {}), questions, "v1")
    report = (run_dir / "report.md").read_text()
    assert "venue_coverage@3" in report and "## Paired comparisons" in report
    assert "venue_split_oracle (oracle) \u2020" in report
    assert output_defects(report) == []
    assert (run_dir / "paired.csv").exists()
    header = (run_dir / "per_query.csv").read_text().splitlines()[0]
    assert "retrieved_venues" in header and "venue_coverage@3" in header


def test_the_report_counts_the_venue_imbalance_and_hides_coverage_at_one(
    cfg: Config, tmp_path: Path
):
    from crossvenue_rag.corpus.artifacts import save_chunks, venue_chunk_counts

    chunks = build_corpus(cfg)
    save_chunks(cfg, chunks)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    questions = [
        Question("q1", "fee tiers on uniswap and pancakeswap", "gold", ["uniswap", "pancakeswap"],
                 "comparison", gold_chunk_ids=[chunks[0].chunk_id]),
    ]  # fmt: skip
    cfg.eval.k_values = [1, 3]
    cfg.eval.breakdowns = ["venue"]
    runs = Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None)
    run_dir = write_eval_run(
        cfg,
        run_tables(
            cfg, runs, summarise(runs, cfg.eval.k_values), questions, venue_chunk_counts(cfg)
        ),
        questions,
        "v1",
    )
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "| venue_coverage@3 |" in report and "| venue_coverage@1 |" not in report
    assert "The corpus is unbalanced (" in report
    with (run_dir / "summary.csv").open(encoding="utf-8") as fh:
        assert "venue_coverage@1" in next(csv.reader(fh))


def test_summary_csv_carries_each_steps_p50_and_p95_unrounded(cfg: Config, tmp_path: Path):
    from crossvenue_rag.evaluation.evaluator import STEP_COLUMN_STEPS, step_latency_columns

    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    questions = [Question("q1", "fee tiers on pancakeswap", "gold", ["pancakeswap"], "fact")]
    cfg.eval.k_values = [1, 3]
    runs = Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None)
    run_dir = write_eval_run(
        cfg,
        run_tables(cfg, runs, summarise(runs, cfg.eval.k_values), questions, {}),
        questions,
        "v1",
    )
    with (run_dir / "summary.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header, rows = list(reader.fieldnames or []), list(reader)
    steps = [f"{step}_{p}_ms" for step in STEP_COLUMN_STEPS for p in ("p50", "p95")]
    # The step columns follow the total latency columns, in step order.
    start = header.index("p95_ms") + 1
    assert header[start : start + len(steps)] == steps
    expected = step_latency_columns(runs)
    for row in rows:
        assert {c: float(row[c]) for c in steps} == expected[row["method"]]


@pytest.mark.parametrize("run", committed_of_kind("per_query.csv"), ids=lambda p: p.name)
def test_every_committed_per_query_table_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.evaluation.results import read_per_query

    assert_round_trip(run, "per_query.csv", read_per_query(run), tmp_path)


def test_the_pool_runs_per_query_table_writes_back_byte_for_byte(tmp_path):
    from crossvenue_rag.evaluation.results import read_per_query

    [run] = committed_of_kind("per_query.csv", "_pool")
    assert_round_trip(run, "per_query.csv", read_per_query(run), tmp_path)


@pytest.mark.parametrize("run", committed_of_kind("summary.csv"), ids=lambda p: p.name)
def test_every_committed_retrieval_summary_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.evaluation.results import read_summary

    assert_round_trip(run, "summary.csv", read_summary(run), tmp_path)


@pytest.mark.parametrize("run", committed_of_kind("by_category.csv"), ids=lambda p: p.name)
def test_every_committed_category_breakdown_writes_back_byte_for_byte(run, tmp_path):
    from crossvenue_rag.evaluation.results import read_breakdown

    assert_round_trip(run, "by_category.csv", read_breakdown(run, "category"), tmp_path)


def test_a_saved_row_gives_its_venues_or_resolves_them_through_the_index():
    from crossvenue_rag.evaluation.results import PerQueryRow

    saved = PerQueryRow.model_validate(
        {"question_id": "q", "method": "bm25", "retrieved_ids": "a b", "retrieved_venues": "x y"}
    )
    older = PerQueryRow.model_validate(
        {"question_id": "q", "method": "bm25", "retrieved_ids": "a b"}
    )
    assert saved.retrieval({}).venues == ["x", "y"]
    assert older.retrieval({"a": "binance", "b": "bybit"}).venues == ["binance", "bybit"]


def test_a_saved_chunk_id_without_a_venue_is_a_value_error_naming_it():
    from crossvenue_rag.evaluation.results import PerQueryRow

    older = PerQueryRow.model_validate(
        {"question_id": "q", "method": "bm25", "retrieved_ids": "a b"}
    )
    with pytest.raises(ValueError, match="bm25 on q: no venue for b"):
        older.retrieval({"a": "binance"})


def test_a_summary_of_a_method_without_runs_is_refused_by_the_writer(cfg: Config):
    _, questions, runs = evaluated(cfg)
    summaries = summarise(runs, cfg.eval.k_values)
    kept = [r for r in runs if r.method != summaries[-1].method]
    tables = run_tables(cfg, kept, summaries, questions, {})
    with pytest.raises(ValueError, match=f"a summary names {summaries[-1].method}"):
        write_eval_run(cfg, tables, questions, "v1")


def test_a_breakdown_is_written_to_the_file_named_after_its_dimension():
    assert breakdown_file("venue") == "by_venue.csv"
