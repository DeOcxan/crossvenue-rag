"""End-to-end smoke test on a four-document fixture corpus.

Runs manifest -> normalize -> chunk -> BM25 -> retrieve (every non-oracle method) ->
evaluate every configured method -> results artifacts, with the embedder, the vector store and the
reranker replaced by in-memory stand-ins. It will not catch a bad embedding
model; it will catch the pipeline being wired together wrongly, which is the
failure that actually happens.
"""

from __future__ import annotations

import csv

from crossvenue_rag.config import Config
from crossvenue_rag.corpus.artifacts import load_chunks
from crossvenue_rag.evaluation.evaluator import Evaluator, summarise
from crossvenue_rag.evaluation.results import run_tables, write_eval_run
from crossvenue_rag.index.bm25_index import INDEX_DIRNAME, BM25Index
from crossvenue_rag.models import Question, Timings
from tests.conftest import AD_HOC_METHODS, InMemoryRetriever, build_corpus


def test_pipeline_runs_end_to_end(cfg: Config):
    chunks = build_corpus(cfg)
    assert chunks
    assert {c.venue for c in chunks} == {"binance", "bybit", "uniswap", "pancakeswap"}

    # Ingestion wrote both artifacts and they reload.
    assert (cfg.paths.processed / "corpus_stats.json").exists()
    assert [c.chunk_id for c in load_chunks(cfg)] == [c.chunk_id for c in chunks]

    bm25 = BM25Index.build(chunks)
    bm25.save(cfg.paths.index / INDEX_DIRNAME)

    retriever = InMemoryRetriever(cfg, chunks, BM25Index.load(cfg.paths.index / INDEX_DIRNAME))
    for method in AD_HOC_METHODS:
        timings = Timings()
        results = retriever.retrieve("what are the fee tiers", method=method, k=3, timings=timings)
        assert results, f"{method} returned nothing"
        assert len(results) <= 3
        assert timings.total_ms > 0
        assert len({r.chunk_id for r in results}) == len(results)

    # Evaluation scores a labeled question and writes the run directory.
    gold = next(c.chunk_id for c in chunks if "0.25%" in c.text)
    questions = [
        Question("q1", "What fee tiers does PancakeSwap v3 support?", "gold", ["pancakeswap"],
                 "fact", gold_chunk_ids=[gold], category="fact"),
    ]  # fmt: skip
    runs = Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None)
    assert {r.method for r in runs} == set(cfg.eval.methods)
    run_dir = write_eval_run(
        cfg,
        run_tables(cfg, runs, summarise(runs, cfg.eval.k_values), questions, {}),
        questions,
        "v1",
    )
    for name in ("per_query.csv", "summary.csv", "report.md", "index.json", "config.yaml"):
        assert (run_dir / name).exists(), name
    with (run_dir / "per_query.csv").open(encoding="utf-8", newline="") as fh:
        assert len(list(csv.DictReader(fh))) == len(runs)
