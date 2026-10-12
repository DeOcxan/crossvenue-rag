"""Builders the evaluation tests share: an evaluated fixture corpus, a run for the power study."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from crossvenue_rag.config import Config
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.evaluation.evaluator import Evaluator, QueryRun
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Question
from tests.conftest import InMemoryRetriever, build_corpus

# The paired.csv columns the power study does not read, each with a neutral value.
NEUTRAL_PAIRED = {
    **dict.fromkeys(("n", "mean_a", "mean_b", "diff", "low", "high"), 0),
    **dict.fromkeys(("better", "worse", "ties"), 0),
    "mcnemar_p": "",
}


def fake_run(tmp_path: Path, *, family: bool = True) -> tuple[Path, list[Question]]:
    """A retrieval run with a two-comparison recall@10 family on six comparison
    questions, one fact question and one method outside the family."""
    questions = [
        Question(f"cv-{i}", "binance vs bybit", "gold", ["binance", "bybit"], "comparison",
                 gold_chunk_ids=["g", "h"] if i == 0 else ["g"], category="comparison")
        for i in range(6)
    ]  # fmt: skip
    questions.append(Question("cv-9", "fact", "gold", ["binance"], "fact", ["g"], category="fact"))
    path = tmp_path / "questions.jsonl"
    save_questions(path, questions)
    run = tmp_path / "run"
    run.mkdir()
    index = {"index_version": "v1", "questions": str(path)}
    (run / "index.json").write_text(json.dumps(index), encoding="utf-8")
    scope = "labelled comparison" if family else "labelled fact"
    with (run / "paired.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "scope", "a", "b", "role", *NEUTRAL_PAIRED])
        writer.writerow(
            ["recall@10", scope, "hybrid", "bm25", "secondary", *NEUTRAL_PAIRED.values()]
        )
        writer.writerow(
            ["recall@10", scope, "hybrid", "dense", "secondary", *NEUTRAL_PAIRED.values()]
        )
        writer.writerow(
            ["recall@10", scope, "hybrid", "venue_split", "primary", *NEUTRAL_PAIRED.values()]
        )
        writer.writerow(
            ["recall@5", scope, "hybrid", "bm25", "secondary", *NEUTRAL_PAIRED.values()]
        )
    retrieved = {"hybrid": "g", "bm25": "x", "dense": "x g", "venue_split": "g"}
    with (run / "per_query.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["question_id", "method", "retrieved_ids"])
        for q in questions:
            for method, ids in retrieved.items():
                writer.writerow([q.id, method, ids if q.id != "cv-1" else "g"])
    return run, questions


def evaluated(cfg: Config) -> tuple[InMemoryRetriever, list[Question], list[QueryRun]]:
    """Every method of `cfg` run at k 1 and 3 over the fixture corpus.

    Two questions: q1 on pancakeswap with gold, and q2 on binance without.
    """
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    gold = next(c.chunk_id for c in chunks if "0.25%" in c.text)
    questions = [
        Question(
            id="q1",
            question="What fee tiers does PancakeSwap v3 support?",
            tier="gold",
            venues=["pancakeswap"],
            qtype="fact",
            gold_chunk_ids=[gold],
        ),
        Question(
            id="q2",
            question="How does Binance report used request weight?",
            tier="silver",
            venues=["binance"],
            qtype="fact",
        ),
    ]
    cfg.eval.k_values = [1, 3]
    return (
        retriever,
        questions,
        Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=None),
    )
