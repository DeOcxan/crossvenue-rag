"""Tests for `reporting.rederive`: a committed run's tables re-derived and compared."""

from __future__ import annotations

import csv
from pathlib import Path

from crossvenue_rag.config import Config
from crossvenue_rag.corpus.artifacts import save_chunks
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.evaluation.evaluator import Evaluator, summarise
from crossvenue_rag.evaluation.results import run_tables, write_eval_run
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Question
from crossvenue_rag.reporting.rederive import rederive_retrieval_run
from tests.conftest import InMemoryRetriever, build_corpus


def retrieval_run(cfg: Config, tmp_path: Path) -> Path:
    chunks = build_corpus(cfg)
    save_chunks(cfg, chunks)
    gold = next(c.chunk_id for c in chunks if "0.25%" in c.text)
    questions = [
        Question("q1", "fee tiers on uniswap and pancakeswap", "gold",
                 ["uniswap", "pancakeswap"], "comparison", category="comparison",
                 gold_chunk_ids=[gold]),
        Question("q2", "binance request weight", "gold", ["binance"], "fact", category="fact"),
    ]  # fmt: skip
    path = tmp_path / "questions.jsonl"
    save_questions(path, questions)
    cfg.eval.questions = path
    cfg.eval.k_values = [1, 3]
    cfg.eval.breakdowns = ["category"]
    cfg.eval.paired_comparisons = [("hybrid_rerank", "hybrid")]
    cfg.eval.primary_comparison = []
    cfg.eval.bootstrap_resamples = 200
    runs = Evaluator(cfg, InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))).run(
        questions, filter_by_question_venues=False, judged=None
    )
    return write_eval_run(
        cfg,
        run_tables(cfg, runs, summarise(runs, cfg.eval.k_values), questions, {}),
        questions,
        "v1",
    )


def test_a_run_re_derives_from_its_per_query_rows(cfg: Config, tmp_path: Path):
    run_dir = retrieval_run(cfg, tmp_path)
    cells, problems = rederive_retrieval_run(run_dir, tmp_path)
    assert cells > 0 and problems == []


def test_an_edited_aggregate_is_caught(cfg: Config, tmp_path: Path):
    run_dir = retrieval_run(cfg, tmp_path)
    rows = list(csv.DictReader((run_dir / "summary.csv").open(encoding="utf-8")))
    rows[0]["recall@3"] = "0.999"
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, problems = rederive_retrieval_run(run_dir, tmp_path)
    assert any("recall@3: '0.999' saved" in p for p in problems)


def test_a_changed_question_file_skips_the_re_derivation(cfg: Config, tmp_path: Path):
    run_dir = retrieval_run(cfg, tmp_path)
    with (tmp_path / "questions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert rederive_retrieval_run(run_dir, tmp_path) == (0, [])


def test_a_saved_row_that_nothing_re_derives_is_caught(cfg: Config, tmp_path: Path):
    run_dir = retrieval_run(cfg, tmp_path)
    rows = list(csv.DictReader((run_dir / "summary.csv").open(encoding="utf-8")))
    rows.append({**rows[0], "method": "stale_method"})
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, problems = rederive_retrieval_run(run_dir, tmp_path)
    assert any("('stale_method',) is saved but not re-derived" in p for p in problems)
