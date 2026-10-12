"""Tests for `generation.answer_run`: an answer run's files, from a matrix run."""

from __future__ import annotations

import csv

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation.answer_run import write_answer_run
from crossvenue_rag.generation.answering import AnswerInputs, run_answers
from crossvenue_rag.generation.llm import LLM
from tests.conftest import output_defects
from tests.generation.factories import (
    judge_json,
    record,
)
from tests.generation.offline import (
    CHUNKS,
    QUESTIONS,
    SECRET,
    TwoChunkRetriever,
    config,
    respond,
)


def test_matrix_run_writes_every_artifact_with_call_records(tmp_path):
    from tests.generation.offline import (
        FakeTransport,
    )

    cfg = config(tmp_path)
    transport = FakeTransport(respond=respond)
    llm = LLM(cfg.generation, transport, api_key=SECRET)
    records = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"),
        progress=lambda _msg: None,
    ).records
    assert len(records) == 4 and not any(r.error for r in records)

    first = records[0]
    generation, graded = first.generated(), first.judged()
    assert generation.prompt_sha256 and graded.prompt_sha256
    sent = generation.call.params
    assert sent is not None and sent["reasoning"] == {"effort": "minimal", "exclude": True}
    assert "temperature" not in sent  # gemini's endpoint takes none
    assert graded.judge == "sonnet"
    assert graded.call.provider == "Anthropic"
    citation = generation.citations[0][0]
    assert citation.chunk_id == "b1" and citation.is_primary
    assert graded.metrics.faithfulness == 1.0

    run_dir = write_answer_run(cfg, records, tmp_path / "results", note="", index_version="v1")
    for name in (
        "answers.jsonl",
        "per_answer.csv",
        "summary.csv",
        "paired.csv",
        "report.md",
        "config.yaml",
    ):
        assert (run_dir / name).exists()
    rows = list(csv.DictReader((run_dir / "per_answer.csv").read_text().splitlines()))
    assert rows[0]["requested_model"] == "google/gemini-3.8-flash"
    assert rows[0]["actual_model"] and rows[0]["provider"] and rows[0]["cache_key"]
    paired = list(csv.DictReader((run_dir / "paired.csv").read_text().splitlines()))
    primary = [r for r in paired if r["role"] == "primary"]
    assert {(r["metric"], r["scope"]) for r in primary} == {
        ("venue_attribution_error", "comparison"),
        ("coverage", "comparison"),
    }
    assert primary[0]["a"] == "gemini@venue_split"
    assert SECRET not in (run_dir / "answers.jsonl").read_text()
    assert output_defects((run_dir / "report.md").read_text(encoding="utf-8")) == []

    # A second run is served entirely from the cache.
    calls = len(transport.calls)
    again = run_answers(
        cfg,
        TwoChunkRetriever(),
        AnswerInputs(QUESTIONS, CHUNKS, llm, "sonnet"),
        progress=lambda _msg: None,
    ).records
    assert len(transport.calls) == calls
    assert all(r.generated().call.cached for r in again)


def test_a_run_graded_by_two_judges_leaves_no_run_directory(tmp_path):
    configs = [m.name for m in config(tmp_path).generation.matrix]
    records = [
        record(c, q, judge=judge_json(judge=judge))
        for c, judge in zip(configs[:2], ("opus", "sonnet"), strict=True)
        for q in ("cv-0001", "cv-0002")
    ]
    with pytest.raises(UserError, match="opus, sonnet"):
        write_answer_run(
            config(tmp_path), records, tmp_path / "results", note="", index_version="v1"
        )
    assert not (tmp_path / "results").exists()
