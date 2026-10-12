"""Tests for `crossvenue_rag.evaluation.report`: the report.md of a retrieval run."""

from __future__ import annotations

from dataclasses import replace

import pytest

from crossvenue_rag import formatting
from crossvenue_rag.evaluation.comparisons import PairedRow
from crossvenue_rag.evaluation.evaluator import Evaluator, summarise, summarise_by
from crossvenue_rag.evaluation.report import RunTables, report_markdown
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Question
from tests.conftest import InMemoryRetriever, build_corpus, output_defects


@pytest.fixture
def run(cfg):
    """A config, two invented questions (one labeled) and their evaluated runs."""
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    gold = next(c.chunk_id for c in chunks if c.venue == "uniswap")
    questions = [
        Question("t1", "fee tiers on uniswap", "gold", ["uniswap"], "fact", gold_chunk_ids=[gold]),
        Question("t2", "order weight on binance", "gold", ["binance"], "fact"),
    ]
    cfg.eval.k_values = [1, 3]
    return (
        cfg,
        questions,
        Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None),
    )


def tables_of(cfg, questions, runs, *, paired=(), off_venue=()):
    summaries = summarise(runs, cfg.eval.k_values)
    breakdowns = {"qtype": summarise_by(runs, questions, cfg.eval.k_values, "qtype")}
    return RunTables(runs, summaries, breakdowns, list(paired), list(off_venue), {})


def paired_row(metric, **fields):
    row = {
        "a": "hybrid_rerank",
        "b": "bm25",
        "metric": metric,
        "scope": "all",
        "n": 1200,
        "mean_a": 0.5,
        "mean_b": 0.5612,
        "diff": -0.0612,
        "low": -0.1,
        "high": 0.02,
        "better": 3,
        "worse": 7,
        "ties": 1190,
        "mcnemar_p": None,
        "randomisation_p": 0.0004,
    }
    return PairedRow(**{**row, **fields})


def test_the_report_states_the_run_settings_and_counts_the_labeled_questions(run):
    cfg, questions, runs = run
    report = report_markdown(cfg, "run-x", "unknown", questions, tables_of(cfg, questions, runs))
    assert report.startswith("# Retrieval ablation\n")
    assert "- run: `run-x`" in report
    assert "- `index_version`: `unknown`" in report
    assert "- questions: 2 (1 with gold labels)" in report
    assert f"- RRF k: {cfg.retrieval.rrf_k}" in report
    for heading in ("## Metrics", "### By qtype", "## Latency (p50, ms per step)"):
        assert heading in report


def test_the_chunking_and_rrf_settings_are_written_through_the_formatting_module(run, monkeypatch):
    cfg, questions, runs = run
    cfg.chunking.max_tokens, cfg.chunking.overlap_tokens = 1200, 60
    tables = tables_of(cfg, questions, runs)
    report = report_markdown(cfg, "run-x", "unknown", questions, tables)
    assert "- chunking: max 1200 tokens, overlap 60\n" in report
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    report = report_markdown(cfg, "run-x", "unknown", questions, tables)
    assert "- chunking: max <1200> tokens, overlap <60>\n" in report
    assert f"- RRF k: <{cfg.retrieval.rrf_k}>\n" in report


def test_a_run_without_labels_or_multi_venue_questions_has_no_metric_table(run):
    cfg, questions, runs = run
    unlabeled = [replace(q, gold_chunk_ids=[]) for q in questions]
    report = report_markdown(cfg, "run-x", "v1", unlabeled, tables_of(cfg, unlabeled, runs))
    assert "holds no gold labels" in report
    assert "### By qtype" not in report
    assert "## Latency (p50, ms per step)" in report


def test_the_paired_and_off_venue_sections_appear_only_with_rows(run):
    cfg, questions, runs = run
    bare = report_markdown(cfg, "run-x", "v1", questions, tables_of(cfg, questions, runs))
    assert "## Paired comparisons" not in bare
    assert "## Off-venue retrieval" not in bare

    tables = tables_of(
        cfg,
        questions,
        runs,
        paired=[paired_row("recall@3")],
        off_venue=[paired_row("off_venue@3", higher_is_better=False)],
    )
    report = report_markdown(cfg, "run-x", "v1", questions, tables)
    assert "## Paired comparisons" in report
    assert "## Off-venue retrieval (exploratory, methodology section 9b)" in report
    assert report.index("## Paired comparisons") < report.index("## Off-venue retrieval")
    assert "| off_venue@3 |" in report.split("## Off-venue retrieval")[1]


def test_numbers_are_written_through_formatting(run):
    cfg, questions, runs = run
    cfg.eval.bootstrap_resamples = 1000
    summaries = [replace(s, latency_p50_ms=2531.4) for s in summarise(runs, cfg.eval.k_values)]
    tables = replace(
        tables_of(cfg, questions, runs, paired=[paired_row("recall@3")]), summaries=summaries
    )
    report = report_markdown(cfg, "run-x", "v1", questions, tables)
    assert "1,000 resamples" in report and "1,000 permutations" in report
    row = next(line for line in report.splitlines() if " vs bm25 |" in line)
    assert f"| {formatting.count(1200)} |" in row
    assert f"| {formatting.signed(-0.0612)} |" in row and "\u22120.061" in row
    assert f"| {formatting.interval(-0.1, 0.02)} |" in row
    assert f"| {formatting.p_value(0.0004)} |" in row
    assert "| 2,531 |" in report


def test_the_seed_is_written_through_the_formatting_module_as_an_identifier(run, monkeypatch):
    cfg, questions, runs = run
    cfg.eval.bootstrap_seed = 20260924
    tables = tables_of(cfg, questions, runs, paired=[paired_row("recall@3")])
    assert report_markdown(cfg, "run-x", "v1", questions, tables).count("seed 20260924)") == 2
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<seed {value}>")
    report = report_markdown(cfg, "run-x", "v1", questions, tables)
    assert report.count("seed <seed 20260924>)") == 2


def test_a_full_report_has_no_raw_lists_dicts_or_other_output_defects(run):
    cfg, questions, runs = run
    tables = tables_of(
        cfg,
        questions,
        runs,
        paired=[paired_row("recall@3")],
        off_venue=[paired_row("off_venue@1", higher_is_better=False)],
    )
    report = report_markdown(cfg, "run-x", "v1", questions, tables)
    assert output_defects(report) == []
    assert "{'" not in report and "['" not in report


def test_a_filtered_run_says_so_and_descriptive_rows_are_named(run):
    cfg, questions, runs = run
    plain = report_markdown(cfg, "run-x", "v1", questions, tables_of(cfg, questions, runs))
    assert "venue filter" not in plain
    descriptive = paired_row("off_venue@3", role="descriptive", randomisation_p=None, holm_p=None)
    tables = replace(tables_of(cfg, questions, runs, off_venue=[descriptive]), venue_filter=True)
    report = report_markdown(cfg, "run-x", "v1", questions, tables)
    assert "- venue filter: each query restricted to the question's labeled venues" in report
    assert "is descriptive instead: an interval and no p-value." in report
