"""Tests for `generation.records`: every saved record reads and writes back byte for byte."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from crossvenue_rag import jsonl
from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.records import (
    append_rating,
    read_answers,
    read_judgements,
    read_ratings,
    write_answers,
    write_judgements,
)
from crossvenue_rag.run_files import RUNS_REGISTRY, read_registry
from tests.conftest import REPO
from tests.generation.factories import (
    answer_json,
    call_json,
    generation_json,
    judge_json,
    judgement_json,
    record_json,
)

RATINGS = REPO / "evals/annotations/judge_ratings.jsonl"

# Non-ASCII text is saved unescaped, and a float keeps its shortest representation.
TEXT = "Bybit \u2014 recv_window \u2264 60 000 ms"
COST = 0.1 + 0.2


def _committed_record_files() -> list[str]:
    """answers.jsonl and judgements_*.jsonl of every run results/runs.json registers."""
    runs = read_registry(REPO / RUNS_REGISTRY).runs
    return [
        f"results/{run}/{name}"
        for run, entry in sorted(runs.items())
        for name in sorted(entry.files)
        if name == "answers.jsonl" or (name.startswith("judgements_") and name.endswith(".jsonl"))
    ]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_different_line(a: bytes, b: bytes) -> int | None:
    for number, (x, y) in enumerate(zip(a.splitlines(), b.splitlines(), strict=False), start=1):
        if x != y:
            return number
    return None


def _assert_same_bytes(original: Path, written: Path) -> None:
    # Digests, not texts: an answer record holds question text, which stays out of the output.
    a, b = original.read_bytes(), written.read_bytes()
    assert _digest(b) == _digest(a), f"first different line: {_first_different_line(a, b)}"


def _saved(tmp_path: Path, *rows: dict) -> Path:
    """`rows` saved as JSON Lines the way the pipeline saves them."""
    path = tmp_path / "saved.jsonl"
    jsonl.write(path, rows)
    return path


def _answers_round_trip(tmp_path: Path, *records: dict) -> None:
    path = _saved(tmp_path, *records)
    copy = tmp_path / "copy.jsonl"
    write_answers(copy, read_answers(path))
    _assert_same_bytes(path, copy)


@pytest.mark.parametrize("relative", _committed_record_files())
def test_a_committed_record_file_is_written_back_with_the_same_bytes(relative, tmp_path):
    path = REPO / relative
    copy = tmp_path / path.name
    if path.name == "answers.jsonl":
        write_answers(copy, read_answers(path))
    else:
        write_judgements(copy, read_judgements(path))
    _assert_same_bytes(path, copy)


def test_the_human_ratings_are_written_back_with_the_same_bytes(tmp_path):
    assert RATINGS.exists(), "evals/annotations/judge_ratings.jsonl is committed"
    copy = tmp_path / "ratings.jsonl"
    for row in read_ratings(RATINGS):
        append_rating(copy, row)
    _assert_same_bytes(RATINGS, copy)


def test_a_record_of_the_first_answer_runs_keeps_its_parameters_and_cost_source(tmp_path):
    """generation.params; call records with cost_source and without reasoning_tokens,
    invalid_responses and params; a judge record with params and without a judge name.
    """
    call = {
        "requested_model": "m",
        "actual_model": "m-1",
        "provider": "p",
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": COST,
        "cost_source": "openrouter",
        "latency_ms": 12.5,
        "retries": 0,
        "cache_key": "k",
        "cached": False,
    }
    claim = {"text": TEXT, "venue": "bybit", "chunks": [1]}
    generation = {
        "answer": {
            "answer": TEXT,
            "claims": [claim],
            "refused": False,
            "asked_clarification": False,
        },
        "citations": generation_json()["citations"],
        "chunk_ids": ["b1"],
        "prompt_sha256": "x",
        "params": {"reasoning_effort": "low", "max_tokens": 900, "temperature": 0.0},
        "call": call,
    }
    judge = {
        "judgement": judgement_json(),
        "metrics": {
            "faithfulness": 1.0,
            "venue_attribution_error": 0.0,
            "coverage": None,
            "behaviour_correct": 1.0,
        },
        "prompt_sha256": "j",
        "params": {"effort": "high", "max_tokens": 4000},
        "replicate": 0,
        "call": call,
    }
    _answers_round_trip(tmp_path, record_json(generation=generation, judge=judge))


def test_a_schema_v1_answer_keeps_no_partial_and_no_schema_version(tmp_path):
    generation = generation_json(call=call_json(cost_usd=COST))
    del generation["answer_schema_version"]
    for field in ("partial", "missing"):
        del generation["answer"][field]
    [record] = read_answers(_saved(tmp_path, record_json(generation=generation)))
    assert record.generated().answer.stated_partial() is None
    _answers_round_trip(tmp_path, record_json(generation=generation))


def test_a_schema_v2_answer_keeps_what_it_says_is_missing(tmp_path):
    generation = generation_json(answer={**answer_json(partial=True), "answer": TEXT})
    [record] = read_answers(_saved(tmp_path, record_json(generation=generation)))
    assert record.generated().answer.stated_partial() is True
    _answers_round_trip(tmp_path, record_json(generation=generation))


def test_a_judged_record_keeps_its_judge_error_null_or_set(tmp_path):
    judged = {**record_json(judge=judge_json()), "judge_error": None}
    failed = {**record_json("qwen@venue_split"), "judge_error": "ModelCallError: 3 invalid"}
    _answers_round_trip(tmp_path, judged, failed)


def test_a_failed_generation_keeps_its_error_and_null_parts(tmp_path):
    _answers_round_trip(tmp_path, record_json(error="UserError: HTTP 400"))


def test_a_judgement_file_keeps_both_its_gradings_and_its_errors(tmp_path):
    path = _saved(
        tmp_path,
        {"answer_id": "gemini@venue_split:cv-0001", "judge": judge_json()},
        {"answer_id": "gemini@venue_split:cv-0002", "error": f"invalid {TEXT}"},
    )
    copy = tmp_path / "copy.jsonl"
    write_judgements(copy, read_judgements(path))
    _assert_same_bytes(path, copy)


def test_a_rating_keeps_its_index_version_only_when_it_has_one(tmp_path):
    rating = {"answer_id": "a", "rater": "r", "at": "2026-09-25T00:00:00Z"}
    path = _saved(
        tmp_path,
        {**rating, "rating": judgement_json()},
        {**rating, "rating": judgement_json(claims=2), "index_version": "2825a7160c4d09bf"},
    )
    copy = tmp_path / "copy.jsonl"
    for row in read_ratings(path):
        append_rating(copy, row)
    _assert_same_bytes(path, copy)
    assert read_ratings(tmp_path / "absent.jsonl") == []


def test_a_field_the_reader_does_not_know_is_refused_not_dropped(tmp_path):
    with pytest.raises(UserError, match=r"\.jsonl:1: .*unknown_field"):
        read_answers(_saved(tmp_path, {**record_json(), "unknown_field": 1}))
    generation = {**generation_json(), "unknown_field": 1}
    with pytest.raises(UserError, match=r"\.jsonl:1: .*unknown_field"):
        read_answers(_saved(tmp_path, record_json(generation=generation)))


def test_the_judges_of_a_run_are_the_named_ones_sorted():
    from crossvenue_rag.generation.records import AnswerRecord, named_judges
    from tests.generation.factories import judge_json, record

    records = [
        AnswerRecord.model_validate(record("g", "cv-0001", judge=judge_json(judge="opus"))),
        AnswerRecord.model_validate(record("g", "cv-0002", judge=judge_json(judge="opus"))),
        AnswerRecord.model_validate(record("g", "cv-0003")),
    ]
    assert named_judges(records) == ["opus"]


def test_a_claims_chunk_number_counts_from_1_over_the_chunks_the_generator_saw():
    from tests.generation.factories import generation_json, record

    generated = record(generation=generation_json(chunk_ids=("b1", "y1"))).generated()
    assert generated.numbered_chunk_ids() == {1: "b1", 2: "y1"}


def test_a_claim_citing_a_chunk_outside_its_context_is_refused_naming_its_line_and_the_fix(
    tmp_path,
):
    generation = generation_json(chunk_ids=("c1",))
    generation["answer"]["claims"][0]["chunks"] = [2]
    path = _saved(tmp_path, record_json(), record_json(generation=generation))
    with pytest.raises(
        UserError, match=r"saved\.jsonl:2: .* cites chunk 2 of a context of 1 chunk$"
    ) as refused:
        read_answers(path)
    assert "crossvenue answer-eval" in refused.value.hint
