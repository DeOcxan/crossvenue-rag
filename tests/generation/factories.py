"""Saved records of an answer run for the generation tests: answers, judgments and human
ratings, as their JSONL files hold them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from crossvenue_rag import jsonl
from crossvenue_rag.generation.records import AnswerRecord
from crossvenue_rag.generation.schemas import Judgement


def call_json(**fields: Any) -> dict[str, Any]:
    """A call record as answers.jsonl saves it; `fields` replace the defaults."""
    base: dict[str, Any] = {
        "requested_model": "google/gemini-3.8-flash",
        "actual_model": "google/gemini-3.8-flash-20260901",
        "provider": "Google",
        "input_tokens": 1000,
        "output_tokens": 170,
        "reasoning_tokens": 30,
        "cost_usd": 0.01,
        "latency_ms": 1000.0,
        "retries": 0,
        "invalid_responses": 0,
        "cache_key": "k",
        "params": {"model": "google/gemini-3.8-flash"},
        "cached": False,
    }
    return {**base, **fields}


def answer_json(
    *,
    claims: int = 1,
    refused: bool = False,
    asked: bool = False,
    partial: bool = False,
    venue: str = "binance",
    text: str = "t",
) -> dict[str, Any]:
    """A schema v2 generated answer with `claims` identical claims, each citing chunk 1."""
    claim = {"text": text, "venue": venue, "chunks": [1]}
    return {
        "answer": "a",
        "claims": [claim] * claims,
        "refused": refused,
        "asked_clarification": asked,
        "partial": partial,
        "missing": [{"venue": "bybit", "missing": "m"}] if partial else [],
    }


def judgement_json(
    *,
    claims: int = 1,
    supported: bool = True,
    venue_correct: bool = True,
    behaviour_correct: bool = True,
    coverage: Sequence[tuple[str, int, int]] = (),
) -> dict[str, Any]:
    """A judgment with `claims` identical verdicts and one coverage entry per
    (venue, gold facts, covered facts) of `coverage`.
    """
    return {
        "claims": [
            {"index": i, "supported": supported, "venue_correct": venue_correct, "rationale": ""}
            for i in range(1, claims + 1)
        ],
        "coverage": [
            {"venue": v, "gold_facts": g, "covered_facts": c, "rationale": ""}
            for v, g, c in coverage
        ],
        "behaviour_correct": behaviour_correct,
        "behaviour_rationale": "",
    }


def metrics_json(**fields: Any) -> dict[str, Any]:
    """The metrics of a judged answer; `fields` replace the defaults."""
    base = {
        "faithfulness": 1.0,
        "venue_attribution_error": 0.0,
        "coverage": 1.0,
        "behaviour_correct": 1.0,
    }
    return {**base, **fields}


def judge_json(
    *,
    judge: str = "opus",
    judgement: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    call: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A judge record as answers.jsonl saves it."""
    return {
        "judge": judge,
        "judgement": judgement or judgement_json(),
        "metrics": metrics or metrics_json(),
        "prompt_sha256": "j",
        "replicate": 0,
        "call": call or call_json(requested_model="anthropic/claude-opus-5.5"),
    }


def generation_json(
    *,
    answer: dict[str, Any] | None = None,
    chunk_ids: Sequence[str] = ("b1",),
    call: dict[str, Any] | None = None,
    prompt: str = "x",
    schema_version: int | None = 2,
) -> dict[str, Any]:
    """The generation part of an answer record; one citation per chunk a claim cites.

    `schema_version` None leaves `answer_schema_version` out, as a v1 record has it.
    """
    answer = answer or answer_json()
    citations = [
        [
            {
                "chunk_number": n,
                "chunk_id": chunk_ids[n - 1],
                "venue": claim["venue"],
                "doc_path": "p.md",
                "source_url": "https://u",
                "environment": None,
                "api_version": None,
                "product": None,
                "is_primary": True,
                "matched": [],
            }
            for n in claim["chunks"]
        ]
        for claim in answer["claims"]
    ]
    versioned = {} if schema_version is None else {"answer_schema_version": schema_version}
    return {
        "answer": answer,
        "citations": citations,
        "chunk_ids": list(chunk_ids),
        "prompt_sha256": prompt,
        **versioned,
        "call": call or call_json(),
    }


def record_json(
    config: str = "gemini@venue_split",
    question_id: str = "cv-0001",
    *,
    generation: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
    error: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """An answer record as answers.jsonl saves it; a generated one unless `error` is given.

    `fields` replace the record's other fields (`category`, `expected_behavior`).
    """
    if generation is None and error is None:
        generation = generation_json()
    generator, _, retriever = config.partition("@")
    base = {
        "answer_id": f"{config}:{question_id}",
        "config": config,
        "generator": generator,
        "retriever": retriever,
        "question_id": question_id,
        "question": "q",
        "category": "comparison",
        "expected_behavior": "answer",
        "venues": ["binance", "bybit"],
    }
    return {**base, **fields, "generation": generation, "judge": judge, "error": error}


def record(*args: Any, **fields: Any) -> AnswerRecord:
    """`record_json(*args, **fields)` read as an answer record."""
    return AnswerRecord.model_validate(record_json(*args, **fields))


def verdict(i: int, *, supported: bool, venue_ok: bool) -> dict[str, Any]:
    """A claim verdict of a judgment, as the judge writes it."""
    return {"index": i, "supported": supported, "venue_correct": venue_ok, "rationale": ""}


def judgement(
    claims: list[dict[str, Any]], coverage: list[dict[str, Any]], *, behaviour: bool = True
) -> Judgement:
    """A parsed judgment of claim verdicts and coverage per venue."""
    return Judgement.model_validate(
        {
            "claims": claims,
            "coverage": coverage,
            "behaviour_correct": behaviour,
            "behaviour_rationale": "",
        }
    )


def prompt_answer(
    *,
    refused: bool = False,
    partial: bool = False,
    claims: list[dict[str, Any]] | None = None,
    missing: Sequence[str] = (),
) -> dict[str, Any]:
    """A generated answer's JSON: one binance claim unless refused, `missing` venues listed."""
    if claims is None:
        claims = [] if refused else [{"text": "t", "venue": "binance", "chunks": [1]}]
    return {
        "answer": "a",
        "claims": claims,
        "refused": refused,
        "asked_clarification": False,
        "partial": partial,
        "missing": [{"venue": v, "missing": "x"} for v in missing],
    }


def prompt_record(
    config: str,
    qid: str,
    expected: str,
    *,
    vae: float | None = None,
    schema_version: int | None = 2,
    **answer_fields: Any,
) -> AnswerRecord:
    """An answer record for a prompt comparison: judged when it has claims or a `vae`.

    `schema_version` None makes it a record of prompt v1 (`generation_json`).
    """
    answer = prompt_answer(**answer_fields)
    metrics = metrics_json(venue_attribution_error=vae)
    judge = judge_json(metrics=metrics) if vae is not None or answer["claims"] else None
    generation = generation_json(
        answer=answer, chunk_ids=("c-bin", "c-byb"), schema_version=schema_version
    )
    return record(config, qid, expected_behavior=expected, generation=generation, judge=judge)


def matrix_records(
    cache_key: Callable[[str, int], str] = lambda config, i: f"{config}:{i}",
) -> list[AnswerRecord]:
    """Four configurations of 30 answers; `cache_key(name, i)` names each answer's call."""
    out = []
    for name in (
        "gemini@venue_split",
        "qwen@venue_split",
        "mercury@venue_split",
        "gemini@hybrid_rerank",
    ):
        for i in range(30):
            category = "comparison" if i < 15 else ("fact" if i < 25 else "unanswerable")
            call = call_json(cache_key=cache_key(name, i))
            generation = generation_json(call=call)
            out.append(record(name, f"cv-{i:04d}", category=category, generation=generation))
    return out


def graded_judgement(i: int, *, flip: bool = False) -> dict[str, Any]:
    """A grading in the judge's schema; `flip` disagrees on every other answer."""
    wrong = flip and i % 2 == 0
    return {
        "claims": [
            {"index": 1, "supported": (i % 3 != 0) != wrong, "venue_correct": i % 4 != 0,
             "rationale": ""},
            {"index": 2, "supported": i % 2 == 0, "venue_correct": (i % 5 != 0) != wrong,
             "rationale": ""},
        ],
        "coverage": [{"venue": "binance", "gold_facts": 4, "covered_facts": i % 5,
                      "rationale": ""}],
        "behaviour_correct": (i % 3 != 1) != wrong,
        "behaviour_rationale": "",
    }  # fmt: skip


# The answers the judgment and rating fixtures grade.
ANSWER_IDS = [f"gemini@venue_split:cv-{i:04d}" for i in range(12)]


def write_judgements(run: Path, judge: str, *, flip: bool = False, cost: float = 0.01) -> None:
    """`judge`'s judgments of `ANSWER_IDS` in `run`, and one failed judgment.

    Writes the run's answers.jsonl of `ANSWER_IDS` too when it has none, so the
    directory is an answer run.
    """
    if not (run / "answers.jsonl").exists():
        config, _, _ = ANSWER_IDS[0].rpartition(":")
        answers = [record_json(config, a.rpartition(":")[2]) for a in ANSWER_IDS]
        jsonl.write(run / "answers.jsonl", answers)
    rows = [
        {
            "answer_id": a,
            "judge": judge_json(
                judgement=graded_judgement(i, flip=flip), call=call_json(cost_usd=cost)
            ),
        }
        for i, a in enumerate(ANSWER_IDS)
    ]
    jsonl.write(run / f"judgements_{judge}.jsonl", [*rows, {"answer_id": "x", "error": "bad"}])


def write_ratings(path: Path) -> Path:
    """A human rating of each of `ANSWER_IDS` at `path`, agreeing with `graded_judgement`."""
    rows = [
        {
            "answer_id": a,
            "rater": "tester",
            "at": "2026-09-25T00:00:00Z",
            "rating": graded_judgement(i),
        }
        for i, a in enumerate(ANSWER_IDS)
    ]
    jsonl.write(path, rows)
    return path


def rated_in(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Points the ratings file the calibration reads (`judge_sample.RATINGS_PATH`) at `path`."""
    monkeypatch.setattr("crossvenue_rag.generation.calibration.RATINGS_PATH", path)
