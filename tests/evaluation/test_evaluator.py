"""Tests for `crossvenue_rag.evaluation.evaluator`."""

from __future__ import annotations

from dataclasses import replace

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.evaluation.evaluator import Evaluator, summarise, summarise_by
from crossvenue_rag.index.bm25_index import BM25Index
from crossvenue_rag.models import Question
from tests.conftest import InMemoryRetriever, build_corpus
from tests.evaluation.factories import evaluated


def test_unlabelled_questions_are_excluded_not_scored_zero(cfg: Config):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    gold = next(c.chunk_id for c in chunks if "0.25%" in c.text)

    labelled_only = [
        Question("q1", "fee tiers on PancakeSwap v3", "gold", ["pancakeswap"], "fact", [gold])
    ]
    plus_unlabelled = [*labelled_only, Question("q2", "unrelated", "bronze", [], "fact")]

    cfg.eval.k_values = [3]
    a = summarise(
        Evaluator(cfg, retriever).run(labelled_only, filter_by_question_venues=False, judged=None),
        [3],
    )
    b = summarise(
        Evaluator(cfg, retriever).run(
            plus_unlabelled, filter_by_question_venues=False, judged=None
        ),
        [3],
    )
    assert {s.method: s.metrics for s in a} == {s.method: s.metrics for s in b}


def test_unjudged_share_is_measured_from_the_judgment_history(cfg: Config):
    chunks = build_corpus(cfg)
    retriever = InMemoryRetriever(cfg, chunks, BM25Index.build(chunks))
    gold = [c.chunk_id for c in chunks if c.venue == "uniswap"][:1]
    questions = [
        Question("q1", "fee tiers rate limits", "gold", ["uniswap"], "fact", gold_chunk_ids=gold)
    ]
    cfg.eval.k_values = [3]
    cfg.eval.methods = ["bm25"]
    top = retriever.retrieve("fee tiers rate limits", method="bm25", k=3)
    judged = {"q1": {top[0].chunk_id}}  # one of three judged

    [run] = Evaluator(cfg, retriever).run(questions, filter_by_question_venues=False, judged=judged)
    assert run.judged_metrics["unjudged@3"] == pytest.approx(2 / 3)
    assert set(run.judged_metrics) == {"unjudged@3", "ndcg_prime@3"}
    [without] = Evaluator(cfg, retriever).run(
        questions, filter_by_question_venues=False, judged=None
    )
    assert without.judged_metrics == {}

    summaries = summarise([run], [3])
    assert summaries[0].judged_metrics["unjudged@3"] == pytest.approx(0.6667, abs=1e-4)


def test_every_method_runs_and_only_labeled_questions_are_scored(cfg: Config):
    _, questions, runs = evaluated(cfg)
    assert len(runs) == len(questions) * len(cfg.eval.methods)
    summaries = summarise(runs, cfg.eval.k_values)
    assert {s.method for s in summaries} == set(cfg.eval.methods)
    assert all(s.questions == 2 and s.labelled_questions == 1 for s in summaries)
    assert all("recall@3" in s.metrics for s in summaries)


def test_a_question_counts_in_the_slice_of_every_venue_it_names(cfg: Config):
    retriever, questions, runs = evaluated(cfg)
    by_venue = summarise_by(runs, questions, cfg.eval.k_values, "venue")
    assert {s.slice for s in by_venue} == {"pancakeswap", "binance"}
    assert all(s.questions == 1 for s in by_venue)
    # So venue slices do not sum to the total.
    two_venues = [replace(questions[0], venues=["uniswap", "pancakeswap"])]
    spread = summarise_by(
        Evaluator(cfg, retriever).run(two_venues, filter_by_question_venues=False, judged=None),
        two_venues,
        cfg.eval.k_values,
        "venue",
    )
    assert {s.slice for s in spread} == {"uniswap", "pancakeswap"}
    assert sum(s.questions for s in spread) == 2 * len(cfg.eval.methods)
    by_qtype = summarise_by(runs, questions, cfg.eval.k_values, "qtype")
    assert {s.slice for s in by_qtype} == {"fact"}


def test_runs_scored_at_other_cutoffs_are_a_value_error_naming_the_metric(cfg: Config):
    _, questions, runs = evaluated(cfg)
    with pytest.raises(ValueError, match="has no recall@10 in metrics"):
        summarise(runs, [10])
    with pytest.raises(ValueError, match="has no off_venue@10 in off_venue_metrics"):
        summarise_by(runs, questions, [10], "venue")
