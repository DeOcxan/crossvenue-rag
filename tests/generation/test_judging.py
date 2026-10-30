"""Tests for `generation.judging`: saved answers judged again, offline."""

from __future__ import annotations

from crossvenue_rag.generation.answer_run import write_answer_run
from crossvenue_rag.generation.answering import AnswerInputs, run_answers
from crossvenue_rag.generation.judging import judge_records, select_answers
from crossvenue_rag.generation.llm import LLM
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    respond,
)


def test_saved_answers_are_judged_unchanged_and_a_repeat_costs_nothing(tmp_path):
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    source = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    ).records
    transport = FakeTransport(respond=respond)
    llm = LLM(cfg.generation, transport, api_key=SECRET)
    judged = judge_records(
        cfg, source, AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"), progress=lambda _msg: None
    ).records
    assert [r.generation for r in judged] == [r.generation for r in source]
    assert all(r.judged().judge == "sonnet" and r.judge_error is None for r in judged)
    assert len(transport.calls) == len(judged)
    # Judging the same answers again is served from the cache: no further call.
    judge_records(
        cfg, source, AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"), progress=lambda _msg: None
    )
    assert len(transport.calls) == len(judged)
    run_dir = write_answer_run(
        cfg, judged, tmp_path / "results", "n", suffix="_judged", index_version="v1"
    )
    assert run_dir.name.endswith("_judged") and (run_dir / "paired.csv").exists()


def test_a_sample_naming_an_answer_the_run_lacks_is_refused_with_its_ids():
    import pytest

    from crossvenue_rag.errors import UserError
    from tests.generation.factories import record

    run = [record("g", "cv-0001"), record("g", "cv-0002")]
    chosen = select_answers(run, ["g:cv-0002", "g:cv-0001"])
    assert [r.answer_id for r in chosen] == ["g:cv-0002", "g:cv-0001"]
    with pytest.raises(UserError, match="1 answer id not in the run: g:cv-9999"):
        select_answers(run, ["g:cv-0001", "g:cv-9999"])


def test_a_sample_naming_an_answer_whose_generation_failed_is_refused_with_its_ids():
    import pytest

    from crossvenue_rag.errors import UserError
    from tests.generation.factories import record

    run = [record("g", "cv-0001"), record("g", "cv-0002", error="ModelCallError: 500")]
    with pytest.raises(
        UserError, match="1 answer id whose generation failed in the run: g:cv-0002"
    ):
        select_answers(run, ["g:cv-0001", "g:cv-0002"])


def test_judging_answers_to_questions_the_file_lacks_is_refused_before_any_call(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError
    from tests.generation.factories import (
        record,
    )
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    transport = FakeTransport(respond=respond)
    llm = LLM(cfg.generation, transport, api_key=SECRET)
    with pytest.raises(UserError, match="1 question of the run not in the question file: cv-7777"):
        judge_records(
            cfg,
            [record("g", "cv-7777")],
            AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"),
            progress=lambda _msg: None,
        )
    assert transport.calls == []


def test_a_rejudged_record_takes_its_category_and_expected_behavior_from_the_question_file(
    tmp_path,
):
    from dataclasses import replace

    from tests.generation.offline import FakeTransport

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    source = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    ).records
    moved = QUESTIONS[0].id
    questions = [
        replace(q, category="noisy", expected_behavior="ask_clarification") if q.id == moved else q
        for q in QUESTIONS
    ]
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    judged = judge_records(
        cfg, source, AnswerInputs(questions, CHUNKS, llm, "sonnet"), progress=lambda _msg: None
    ).records
    assert [r.generation for r in judged] == [r.generation for r in source]
    for before, after in zip(source, judged, strict=True):
        if before.question_id == moved:
            assert (after.category, after.expected_behavior) == ("noisy", "ask_clarification")
        else:
            assert after.category == before.category
            assert after.expected_behavior == before.expected_behavior
