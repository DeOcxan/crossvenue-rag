"""Tests for `labeling.select_session`: the selection loop and its screens."""

from __future__ import annotations

from rich.console import Console

from crossvenue_rag.config import Config
from crossvenue_rag.dataset.selection import load_decisions, progress
from crossvenue_rag.labeling import select_session
from crossvenue_rag.labeling.select_session import SelectionPlan
from tests.conftest import make_chunk
from tests.dataset.factories import record


def every_category(questions_path, log):
    return SelectionPlan(questions_path, str(log), categories=None, include_skipped=False)


def candidates():
    return [
        record(id=f"q{i}", question=f"Question {i} about Bybit order placement?") for i in range(4)
    ]


def test_each_answer_is_a_decision_appended_as_it_is_made(cfg: Config, tmp_path):
    answers = iter(["y", "n", "g", "s", "q"])
    log = tmp_path / "selection.jsonl"
    console = Console(record=True, width=120)
    logged_at_each_prompt: list[list[str]] = []

    def prompt(*_, **__):
        logged_at_each_prompt.append(sorted(load_decisions(log)))
        return next(answers)

    select_session.run(
        cfg,
        candidates(),
        every_category("questions.jsonl", log),
        console=console,
        prompt=prompt,
    )
    # The third prompt asks q1's reject reason, before q1's decision is made.
    assert logged_at_each_prompt == [[], ["q0"], ["q0"], ["q0", "q1"], ["q0", "q1", "q2"]]
    decisions = load_decisions(log)
    assert {q: (d.decision, d.reason) for q, d in decisions.items()} == {
        "q0": ("accept", ""),
        "q1": ("reject", "not grounded"),
        "q2": ("skip", ""),
    }
    text = console.export_text()
    assert "source previews are unavailable" in text
    assert "saved; the next run continues" in text


def test_nothing_left_to_decide_says_so(cfg: Config, tmp_path):
    log = tmp_path / "selection.jsonl"
    answers = iter(["y"] * 4)
    select_session.run(
        cfg,
        candidates(),
        every_category("q.jsonl", log),
        console=Console(record=True),
        prompt=lambda *a, **k: next(answers),
    )
    console = Console(record=True, width=120)
    select_session.run(cfg, candidates(), every_category("q.jsonl", log), console=console)
    assert "Nothing left to decide" in console.export_text()


def test_the_grounding_table_names_a_chunk_missing_from_the_corpus():
    console = Console(record=True, width=120)
    console.print(select_session.grounding_table(record(source_chunk_ids=["gone"]), {}))
    console.print(select_session.grounding_table(record(), {}))
    text = console.export_text()
    assert "not in the current corpus" in text and "no grounding recorded" in text


def test_the_grounding_table_marks_only_a_text_longer_than_the_preview_as_cut():
    exact = "x" * select_session.CHUNK_PREVIEW_CHARS
    chunks = {"a": make_chunk("a", "bybit", exact), "b": make_chunk("b", "bybit", exact + "y")}
    console = Console(record=True, width=600)
    console.print(select_session.grounding_table(record(source_chunk_ids=["a", "b"]), chunks))
    assert console.export_text().count(" ...") == 1


def test_the_progress_table_has_the_total_and_answerable_comparison_rows():
    console = Console(record=True, width=140)
    console.print(select_session.selection_progress_table(progress(candidates(), {}), "fact"))
    text = console.export_text()
    assert "answerable comparison" in text and "total" in text


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
        every_category("q.jsonl", tmp_path / "s.jsonl"),
        console=Console(record=True),
        prompt=prompt,
    )
    assert asked == ["[y]keep / [n]reject / [s]skip / [q]uit\n"]
