"""Tests for `crossvenue_rag.labeling.relabel`: judging named questions on the label screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.dataset.journal import RELABELED, Judgment
from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.adjudicate import CRITERION_DATE
from crossvenue_rag.labeling.relabel import RelabelTarget, relabel_candidates, relabel_session
from crossvenue_rag.models import Question
from tests.conftest import journal_dict, write_journal
from tests.labeling.conftest import VERSION, Script, chunk, quiet

CHUNKS = {c: chunk(c, "bybit") for c in ("p1", "p2", "x1", "x2")}


def _file(tmp_path: Path) -> str:
    path = tmp_path / "questions.jsonl"
    save_questions(
        path,
        [
            Question(
                "q1", "does bybit cancel all?", "gold", ["bybit"], "fact", category="unanswerable"
            ),
            Question("q2", "other?", "gold", ["bybit"], "fact", category="unanswerable"),
        ],
    )
    entry = {"pool": ["p1", "p2"], "relevant": [], "index_version": VERSION}
    write_journal(path, {"q1": {"1": dict(entry)}, "q2": {"1": dict(entry)}})
    return str(path)


def test_candidates_are_the_recorded_pool_then_the_added_chunks_once():
    entry = Judgment(pool=["p1", "p2"])
    ids = [r.chunk_id for r in relabel_candidates(CHUNKS, "q1", entry, ["x1", "p1", "x1"])]
    assert ids == ["p1", "p2", "x1"]
    with pytest.raises(UserError, match="not in the corpus"):
        relabel_candidates(CHUNKS, "q1", entry, ["zz"])


def test_a_verdict_is_journaled_under_relabeled_and_nothing_else_changes(tmp_path):
    path = _file(tmp_path)
    before_questions, before_rounds = load_questions(path), journal_dict(path)
    console = quiet()
    target = RelabelTarget(path, ["q1", "q2"], {"q1": ["x1"]}, CHUNKS, VERSION, "me")
    relabel_session(target, console=console, prompt=Script("3", ""))
    rounds = journal_dict(path)
    entry = rounds["q1"][RELABELED]
    assert entry["pool"] == ["p1", "p2", "x1"] and entry["relevant"] == ["x1"]
    assert entry["criterion"] == CRITERION_DATE and entry["index_version"] == VERSION
    assert RELABELED not in rounds["q2"]  # skipped
    assert rounds["q1"]["1"] == before_rounds["q1"]["1"]
    assert load_questions(path) == before_questions
    out = console.export_text()
    assert "amended 2026-10-10" in out and "1/2 questions recorded" in out


def test_a_question_without_round_1_is_refused(tmp_path):
    path = _file(tmp_path)
    with pytest.raises(UserError, match="no round-1 judgment"):
        relabel_session(RelabelTarget(path, ["q9"], {}, CHUNKS, VERSION, "me"), console=quiet())


def test_an_interrupt_keeps_the_verdicts_given(tmp_path):
    path = _file(tmp_path)
    answers = iter(["n"])

    def prompt(message: str, **_: object) -> str:
        answer = next(answers, None)
        if answer is None:
            raise KeyboardInterrupt
        return answer

    console = quiet()
    relabel_session(
        RelabelTarget(path, ["q1", "q2"], {}, CHUNKS, VERSION, "me"), console=console, prompt=prompt
    )
    rounds = journal_dict(path)
    assert rounds["q1"][RELABELED]["relevant"] == [] and RELABELED not in rounds["q2"]
    assert "verdicts so far are saved" in console.export_text()
    assert json.loads(json.dumps(rounds))  # the journal stays plain JSON
