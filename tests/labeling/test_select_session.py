"""Tests for `labeling.select_session`: the selection loop and its screens."""

from __future__ import annotations

import pytest
from rich.console import Console

from crossvenue_rag.config import Config
from crossvenue_rag.dataset.selection import load_decisions, progress
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling import select_session
from crossvenue_rag.labeling.select_session import SelectionPlan
from tests.dataset.factories import record


def candidates():
    return [
        record(id=f"q{i}", question=f"Question {i} about Bybit order placement?") for i in range(4)
    ]


def test_each_answer_is_a_decision_appended_as_it_is_made(cfg: Config, tmp_path):
    answers = iter(["y", "n", "g", "s", "q"])
    log = tmp_path / "selection.jsonl"
    console = Console(record=True, width=120)
    select_session.run(
        cfg,
        candidates(),
        SelectionPlan("questions.jsonl", str(log)),
        console=console,
        prompt=lambda *a, **k: next(answers),
    )
    decisions = load_decisions(log)
    assert {q: (d.decision, d.reason) for q, d in decisions.items()} == {
        "q0": ("accept", ""),
        "q1": ("reject", "not grounded"),
        "q2": ("skip", ""),
    }
    text = console.export_text()
    assert "source previews are unavailable" in text
    assert "saved; run again to continue" in text


def test_nothing_left_to_decide_prints_the_progress_only(cfg: Config, tmp_path):
    log = tmp_path / "selection.jsonl"
    answers = iter(["y"] * 4)
    select_session.run(
        cfg,
        candidates(),
        SelectionPlan("q.jsonl", str(log)),
        console=Console(record=True),
        prompt=lambda *a, **k: next(answers),
    )
    console = Console(record=True, width=120)
    select_session.run(cfg, candidates(), SelectionPlan("q.jsonl", str(log)), console=console)
    assert "Nothing left to decide" in console.export_text()


def test_the_grounding_table_names_a_chunk_missing_from_the_corpus():
    console = Console(record=True, width=120)
    console.print(select_session.grounding_table(record(source_chunk_ids=["gone"]), {}))
    console.print(select_session.grounding_table(record(), {}))
    text = console.export_text()
    assert "not in the current corpus" in text and "no grounding recorded" in text


def test_the_progress_table_counts_accepted_against_target():
    console = Console(record=True, width=140)
    console.print(select_session.selection_progress_table(progress(candidates(), {}), "fact"))
    text = console.export_text()
    assert "answerable comparison" in text and "total" in text


def test_check_categories_names_the_unknown_ones():
    from crossvenue_rag.labeling.select_session import SelectionPlan, check_categories

    check_categories(None)
    check_categories(["fact", "comparison"])
    with pytest.raises(UserError, match=r"unknown category: x$"):
        check_categories(["fact", "x"])
    with pytest.raises(UserError, match=r"unknown categories: x, y$") as refused:
        check_categories(["y", "x"])
    assert "comparison" in (refused.value.hint or "")
    plan = SelectionPlan("q.jsonl", "s.jsonl")
    assert plan.categories is None and not plan.include_skipped


def test_the_prompt_shows_its_bracketed_keys(cfg: Config, tmp_path):
    asked: list[str] = []

    def prompt(text, **_):
        console = Console(record=True, width=120)
        console.print(text)
        asked.append(console.export_text())
        return "q"

    select_session.run(
        cfg,
        candidates(),
        SelectionPlan("q.jsonl", str(tmp_path / "s.jsonl")),
        console=Console(record=True),
        prompt=prompt,
    )
    assert asked == ["[y]keep / [n]reject / [s]skip / [q]uit\n"]
