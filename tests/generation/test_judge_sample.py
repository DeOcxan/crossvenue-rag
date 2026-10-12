"""Tests for `generation.judge_sample`: a candidate grading the sample."""

from __future__ import annotations

from crossvenue_rag.generation.answering import AnswerInputs, run_answers
from crossvenue_rag.generation.llm import LLM
from tests.generation.factories import (
    record,
)
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    openrouter_payload,
    respond,
)


def test_a_judge_grades_the_listed_answers_and_the_file_round_trips(tmp_path):
    from crossvenue_rag.generation.answer import generate
    from crossvenue_rag.generation.judge_sample import (
        judge_answers,
        judgements_path,
        load_judgements,
    )
    from crossvenue_rag.generation.records import JudgementRow, write_judgements
    from tests.generation.offline import (
        CHUNKS,
        QUESTIONS,
        SECRET,
        FakeTransport,
        TwoChunkRetriever,
        config,
        respond,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    answers = []
    for question in QUESTIONS:
        results = TwoChunkRetriever().retrieve(question.question, "venue_split", k=2)
        generation = generate(
            llm, cfg.generation, cfg.generation.generators["gemini"], question, results
        )
        answers.append(record("g", question.id, generation=generation.to_json()))
    rows = judge_answers(
        cfg, answers, AnswerInputs(QUESTIONS, CHUNKS, llm, "opus"), ["g:cv-0002"], replicate=0
    ).rows
    assert [r.answer_id for r in rows] == ["g:cv-0002"]
    path = judgements_path(tmp_path, "opus", replicate=1)
    assert path.name == "judgements_opus_r1.jsonl"
    write_judgements(path, [*rows, JudgementRow(answer_id="g:cv-0001", error="invalid")])
    loaded = load_judgements(path)
    assert list(loaded) == ["g:cv-0002"] and loaded["g:cv-0002"].behaviour_correct is True


def test_calibration_judging_records_a_parse_error_and_goes_on(tmp_path):
    from crossvenue_rag.generation.judge_sample import (
        judge_answers,
        judgements_path,
        load_judgements,
    )
    from crossvenue_rag.generation.records import write_judgements
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    records = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, judge_name=None),
        progress=lambda _msg: None,
    ).records
    ids = [r.answer_id for r in records if r.config == "gemini@venue_split"]

    def broken_for_the_fact_question(url, body):
        if body["model"].startswith("anthropic/") and "binance recvWindow max?" in str(body):
            return openrouter_payload("not json", model=body["model"], provider="Anthropic")
        return respond(url, body)

    llm = LLM(cfg.generation, FakeTransport(respond=broken_for_the_fact_question), api_key=SECRET)
    rows = judge_answers(
        cfg, records, AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"), ids, replicate=0
    ).rows
    assert [(r.error is not None, r.judge is not None) for r in rows] == [
        (False, True),
        (True, False),
    ]
    path = judgements_path(tmp_path, "sonnet")
    write_judgements(path, rows)
    assert list(load_judgements(path)) == [ids[0]]


def test_judging_a_sample_id_the_run_lacks_is_refused(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError
    from crossvenue_rag.generation.judge_sample import judge_answers
    from tests.generation.factories import (
        record,
    )
    from tests.generation.offline import (
        CHUNKS,
        QUESTIONS,
        SECRET,
        FakeTransport,
        config,
        respond,
    )

    cfg = config(tmp_path)
    llm = LLM(cfg.generation, FakeTransport(respond=respond), api_key=SECRET)
    inputs = AnswerInputs(QUESTIONS, CHUNKS, llm, "opus")
    with pytest.raises(UserError, match="not in the run: g:cv-9999"):
        judge_answers(cfg, [record("g", "cv-0001")], inputs, ["g:cv-9999"], replicate=0)


def test_a_judge_name_that_is_not_a_plain_name_is_refused_before_anything_is_written(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError
    from crossvenue_rag.generation.judge_sample import judgements_path

    run = tmp_path / "a" / "b" / "run"
    run.mkdir(parents=True)
    for name in ("../x", "x/y", "../../../x", ".."):
        with pytest.raises(UserError, match=r"judge name") as caught:
            judgements_path(run, name)
        assert caught.value.hint == "rename the judge under generation.judges"
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
