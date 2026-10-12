"""Tests for `generation.estimate`: the cost of an answer run, before running it."""

from __future__ import annotations

from crossvenue_rag.generation.answering import AnswerInputs, run_answers
from crossvenue_rag.generation.llm import LLM
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    respond,
)


class RecordingRetriever(TwoChunkRetriever):
    def __init__(self):
        self.calls = []

    def retrieve(self, query, method, k=None, venues_hint=None):
        self.calls.append((query, method, k, venues_hint))
        return super().retrieve(query, method, k=k, venues_hint=venues_hint)


def test_the_estimate_retrieves_exactly_as_the_run_does_oracle_hint_included(tmp_path):
    from crossvenue_rag.config_generation import MatrixEntry
    from crossvenue_rag.generation.estimate import estimate_answers
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    cfg.generation.matrix = [
        MatrixEntry(name="gemini@oracle", generator="gemini", retriever="venue_split_oracle")
    ]
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    ran, estimated = RecordingRetriever(), RecordingRetriever()
    run_answers(
        cfg, ran, AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None), progress=lambda _msg: None
    )
    rows = estimate_answers(cfg, estimated, QUESTIONS, llm, judge_name=None)
    assert estimated.calls == ran.calls
    assert all(hint == q.venues for q, (*_, hint) in zip(QUESTIONS, ran.calls, strict=True))
    # Every answer the run made is found in the cache by the estimate.
    assert rows[0].cached == rows[0].calls == len(QUESTIONS)


def test_the_total_sums_the_known_estimates_and_says_whether_one_is_unknown():
    from crossvenue_rag.generation.estimate import EstimateRow, estimate_total

    known = EstimateRow("a", "m", 2, 0, 0.5, 1.0)
    unknown = EstimateRow("judge", "j", 2, None, None, None)
    assert estimate_total([known, known]) == (2.0, False)
    assert estimate_total([known, unknown]) == (1.0, True)
    assert estimate_total([]) == (0, False)
