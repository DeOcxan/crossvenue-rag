"""Tests for `generation.answering`: the configuration matrix, generated and judged offline."""

from __future__ import annotations

import csv
import json

from crossvenue_rag.generation.answer_run import write_answer_run
from crossvenue_rag.generation.answering import AnswerInputs, run_answers
from crossvenue_rag.generation.jobs import with_budget
from crossvenue_rag.generation.llm import LLM
from crossvenue_rag.generation.schemas import InvalidResponse
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    respond,
)


def test_a_failed_call_is_recorded_and_the_run_goes_on(tmp_path):
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    cfg.generation.max_retries = 0
    transport = FakeTransport(*[(400, {"error": {"message": "bad request"}})] * 8)
    llm = LLM(cfg.generation, transport, api_key=SECRET)
    records = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    ).records
    assert all(r.error and "HTTP 400" in r.error for r in records)
    run_dir = write_answer_run(cfg, records, tmp_path / "results", note="", index_version="v1")
    summary = list(csv.DictReader((run_dir / "summary.csv").read_text().splitlines()))
    assert summary[0]["errors"] == "2"
    assert json.loads((run_dir / "answers.jsonl").read_text().splitlines()[0])["generation"] is None


def test_a_failed_generation_is_recorded_and_a_programming_error_propagates(tmp_path, monkeypatch):
    import pytest

    import crossvenue_rag.generation.answering as answering
    from crossvenue_rag.errors import UserError
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)

    def schema_error(*_args, **_kwargs):
        raise InvalidResponse("generator response does not match the schema: x")

    monkeypatch.setattr(answering, "generate", schema_error)
    records = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    ).records
    assert all(
        r.error and r.error.startswith("InvalidResponse: generator response") for r in records
    )

    # Any other UserError, such as a prompt that is not the pinned one, stops the run.
    def wrong_prompt(*_args, **_kwargs):
        raise UserError("prompts/generator.md is not the pinned prompt")

    monkeypatch.setattr(answering, "generate", wrong_prompt)
    with pytest.raises(UserError, match="pinned prompt"):
        run_answers(
            cfg,
            TwoChunkRetriever(),
            AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
            progress=lambda _msg: None,
        )

    def defect(*_args, **_kwargs):
        raise KeyError("chunk_ids")

    monkeypatch.setattr(answering, "generate", defect)
    with pytest.raises(KeyError, match="chunk_ids"):
        run_answers(
            cfg,
            TwoChunkRetriever(),
            AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
            progress=lambda _msg: None,
        )


def test_a_budget_stop_ends_the_run_with_the_records_made_before_it(tmp_path):
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    transport = FakeTransport(respond=respond)
    llm = LLM(with_budget(cfg, 1e-9).generation, transport, api_key=SECRET)
    batch = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    )
    assert batch.stopped is not None
    assert len(batch.records) == len(transport.calls) < 4
    assert all(r.generation is not None for r in batch.records)


def test_a_failed_judgment_in_an_answer_run_keeps_the_generation_and_is_counted_apart(
    tmp_path, monkeypatch
):
    import crossvenue_rag.generation.answering as answering
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)

    def invalid_judgment(*_args, **_kwargs):
        raise InvalidResponse("judge response does not match the schema: x")

    monkeypatch.setattr(answering, "judge_answer", invalid_judgment)
    inputs = AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet")
    records = run_answers(cfg, TwoChunkRetriever(), inputs, progress=lambda _msg: None).records
    assert all(r.generation is not None and r.error is None for r in records)
    assert all(r.judge is None and r.judge_error and "judge response" in r.judge_error
               for r in records)  # fmt: skip
    run_dir = write_answer_run(cfg, records, tmp_path / "results", note="", index_version="v1")
    summary = list(csv.DictReader((run_dir / "summary.csv").read_text().splitlines()))
    assert summary[0]["errors"] == "0" and summary[0]["judge_errors"] == "2"


def test_an_answer_id_splits_back_into_its_configuration_and_question():
    from crossvenue_rag.generation.answering import answer_id, split_answer_id

    assert split_answer_id(answer_id("gemini@venue_split", "cv-0001")) == (
        "gemini@venue_split",
        "cv-0001",
    )
