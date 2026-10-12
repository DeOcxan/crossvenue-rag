"""Tests for `labeling.rate`: the rating screen never shows a judge verdict."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial

from rich.console import Console
from rich.text import Text

from crossvenue_rag.cli.app import say
from crossvenue_rag.generation.schemas import Claim
from crossvenue_rag.labeling.rate import ask_rating, rate_session, rating_view, render_rating
from tests.conftest import output_defects
from tests.generation.factories import (
    answer_json,
    generation_json,
    judge_json,
    metrics_json,
    record,
)
from tests.generation.offline import (
    chunk,
)


def judgement(claims, coverage, *, behaviour=True):
    return {
        "claims": claims,
        "coverage": coverage,
        "behaviour_correct": behaviour,
        "behaviour_rationale": "",
    }


def rated_record():
    answer = {
        **answer_json(text="bans scale 2m to 3d"),
        "answer": "From 2 minutes to 3 days [1].",
    }
    # The judge's grade: none of this may reach the rater's screen.
    judge = judge_json(
        judgement=judgement(
            [
                {
                    "index": 1,
                    "supported": True,
                    "venue_correct": True,
                    "rationale": "JUDGE-RATIONALE-SECRET",
                }
            ],
            [
                {
                    "venue": "binance",
                    "gold_facts": 7,
                    "covered_facts": 7,
                    "rationale": "JUDGE-COVERAGE-SECRET",
                }
            ],
        )
        | {"behaviour_rationale": "JUDGE-BEHAVIOUR-SECRET"},
        metrics=metrics_json(faithfulness=0.987654),
    )
    generation = generation_json(answer=answer, chunk_ids=("c1", "c2"))
    return record("gemini@venue_split", "cv-0001", generation=generation, judge=judge)


def test_rate_screen_never_shows_the_judges_grade(tmp_path):
    chunks = {
        "c1": chunk("c1", "binance", "IP bans scale from 2 minutes to 3 days."),
        "c2": chunk("c2", "binance", "unrelated"),
    }
    view = rating_view(rated_record(), ["c1"], chunks)
    assert not hasattr(view, "judge") and "judge" not in view.__slots__
    console = Console(record=True, width=160)
    render_rating(partial(say, console), view, position="")
    screen = console.export_text()
    assert "bans scale 2m to 3d" in screen and "IP bans scale" in screen
    for secret in (
        "JUDGE-RATIONALE-SECRET",
        "JUDGE-COVERAGE-SECRET",
        "JUDGE-BEHAVIOUR-SECRET",
        "0.987654",
        "judge",
    ):
        assert secret not in screen


def test_rating_is_recorded_in_the_judges_schema(tmp_path):
    chunks = {"c1": chunk("c1", "binance", "IP bans scale."), "c2": chunk("c2", "binance", "x")}
    view = rating_view(rated_record(), ["c1"], chunks)
    answers = iter(["y", "n", "2", "1", "y"])
    rating = ask_rating(view, prompt=lambda *a, **k: next(answers))
    saved = rating.model_dump()
    assert saved["claims"] == [
        {"index": 1, "supported": True, "venue_correct": False, "rationale": ""}
    ]
    assert saved["coverage"] == [
        {"venue": "binance", "gold_facts": 2, "covered_facts": 1, "rationale": ""}
    ]
    assert saved["behaviour_correct"] is True
    answers = iter(["y", "y", "2", "2", "y"])
    path = tmp_path / "ratings.jsonl"
    console = Console(record=True, width=160)
    assert (
        rate_session(
            [view],
            path,
            "me",
            partial(say, console),
            prompt=lambda *a, **k: next(answers),
            index_version="v1",
        )
        == 1
    )
    assert (
        rate_session(
            [view],
            path,
            "me",
            partial(say, console),
            prompt=lambda *a, **k: "y",
            index_version="v1",
        )
        == 0
    )  # done
    [row] = path.read_text(encoding="utf-8").splitlines()
    assert list(json.loads(row)) == ["answer_id", "rater", "at", "rating", "index_version"]


def test_a_rating_records_the_index_its_chunks_were_resolved_against(tmp_path):
    chunks = {"c1": chunk("c1", "binance", "IP bans scale."), "c2": chunk("c2", "binance", "x")}
    view = rating_view(rated_record(), ["c1"], chunks)
    answers = iter(["y", "y", "2", "2", "y"])
    path = tmp_path / "ratings.jsonl"
    console = Console(record=True, width=160)
    rate_session(
        [view],
        path,
        "me",
        partial(say, console),
        prompt=lambda *a, **k: next(answers),
        index_version="v1",
    )
    [row] = path.read_text(encoding="utf-8").splitlines()
    saved = json.loads(row)
    assert list(saved) == ["answer_id", "rater", "at", "rating", "index_version"]
    assert saved["index_version"] == "v1"


def test_the_rating_screen_and_prompts_are_plain_us_english_text(tmp_path):
    chunks = {"c1": chunk("c1", "binance", "IP bans scale."), "c2": chunk("c2", "binance", "x")}
    view = rating_view(rated_record(), ["c1"], chunks)
    view = replace(view, claims=[Claim(text="t", venue="binance", chunks=[1, 2])])
    console = Console(record=True, width=160)
    render_rating(partial(say, console), view, position="")
    screen = console.export_text()
    assert "(cites [1] and [2])" in screen and "[1, 2]" not in screen
    asked = []
    answers = iter(["y", "y", "1", "1", "y"])

    def prompt(question, *args, **kwargs):
        asked.append(question)
        return next(answers)

    ask_rating(view, prompt=prompt)
    assert all(isinstance(q, Text) for q in asked)
    assert any(q.plain.startswith("behavior as expected") for q in asked)
    assert output_defects("\n".join([screen, *(q.plain for q in asked)])) == []


def test_a_value_on_the_rating_screen_is_shown_as_written_not_read_as_markup():
    chunks = {"c1": chunk("c1", "binance", "IP bans scale."), "c2": chunk("c2", "binance", "x")}
    view = replace(rating_view(rated_record(), ["c1"], chunks), answer_id="[red]a[/red]")
    console = Console(record=True, width=160)
    render_rating(partial(say, console), view, "[bold]1/1[/bold]")
    screen = console.export_text()
    assert "[bold]1/1[/bold]  [red]a[/red]" in screen


def test_a_cited_chunk_missing_from_the_chunks_is_a_value_error_naming_it():
    import pytest

    with pytest.raises(ValueError, match="cites c1, which the chunks lack"):
        rating_view(rated_record(), ["c1"], {"c2": chunk("c2", "binance", "unrelated")})
