"""Tests for `crossvenue_rag.labeling.round3`: the blind relabel of the whole set."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from crossvenue_rag.dataset.journal import FULL_RELABEL_ROUND, load_rounds
from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.adjudicate import CRITERION_DATE
from crossvenue_rag.labeling.pool import (
    RETEST_SHUFFLE_SEED,
    ROUND3_SHUFFLE_SEED,
    retest_order,
)
from crossvenue_rag.labeling.round3 import Round3Target, round3_candidates, round3_session
from crossvenue_rag.models import Question
from tests.conftest import journal_dict, write_journal
from tests.labeling.conftest import VERSION, Script, chunk, quiet

ROUND = str(FULL_RELABEL_ROUND)
POOL = [f"p{i}" for i in range(8)]
CHUNKS = {c: chunk(c, "bybit") for c in [*POOL, "x1"]}


def _questions() -> list[Question]:
    return [
        Question("q1", "does bybit cancel all?", "gold", ["bybit"], "fact", category="fact"),
        Question("q2", "and the other one?", "gold", ["bybit"], "fact", category="fact"),
        Question(
            "q3",
            "is this one unanswerable?",
            "gold",
            ["bybit"],
            "fact",
            category="unanswerable",
            is_answerable=False,
            expected_behavior="refuse",
        ),
    ]


def _target(tmp_path: Path, added: dict[str, list[str]] | None = None) -> Round3Target:
    path = tmp_path / "questions.jsonl"
    save_questions(path, _questions())
    entry = {"pool": POOL, "relevant": [], "index_version": VERSION}
    write_journal(
        path,
        {
            "q1": {"1": dict(entry), "2": dict(entry, shuffle_seed=RETEST_SHUFFLE_SEED)},
            "q2": {"1": dict(entry)},
            "q3": {"1": dict(entry)},
        },
    )
    return Round3Target(str(path), _questions(), added or {}, CHUNKS, VERSION, "me")


def _interrupting(*answers: str):
    """A prompt that replays `answers`, then interrupts."""
    remaining = list(answers)

    def prompt(message: object, **_: object) -> str:
        if not remaining:
            raise KeyboardInterrupt
        return remaining.pop(0)

    return prompt


def test_the_candidates_are_the_recorded_set_and_the_added_chunks_shuffled_in(tmp_path):
    target = _target(tmp_path, {"q1": ["x1", "p0"]})
    shown = [
        r.chunk_id for r in round3_candidates(target, load_rounds(target.questions_path))["q1"]
    ]
    assert sorted(shown) == sorted([*POOL, "x1"])
    assert shown != [*POOL, "x1"]  # the added chunk is not appended


def test_round_3_has_its_own_seed_so_it_does_not_repeat_round_2s_order(tmp_path):
    assert ROUND3_SHUFFLE_SEED != RETEST_SHUFFLE_SEED
    first = load_rounds(_target(tmp_path).questions_path).require_first("q1")
    assert retest_order("q1", first, ROUND3_SHUFFLE_SEED) != retest_order(
        "q1", first, RETEST_SHUFFLE_SEED
    )


def test_adding_nothing_keeps_round_2s_order(tmp_path):
    first = load_rounds(_target(tmp_path).questions_path).require_first("q1")
    expected = list(POOL)
    random.Random(f"{RETEST_SHUFFLE_SEED}:q1").shuffle(expected)
    assert retest_order("q1", first, RETEST_SHUFFLE_SEED) == expected


def test_the_screen_is_blind_and_shows_the_criterion_on_every_question(tmp_path):
    target = _target(tmp_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("n", "n"))
    out = console.export_text()
    assert out.count("amended 2026-10-10") == 3
    assert "found by" not in out and "#s = the question's source chunk" not in out
    assert "notes:" not in out


def test_a_verdict_goes_under_round_3_and_nothing_else_changes(tmp_path):
    target = _target(tmp_path)
    path = target.questions_path
    before_questions = Path(path).read_bytes()
    before = journal_dict(path)
    round3_session(target, console=quiet(), prompt=_interrupting("2"))
    after = journal_dict(path)
    entry = after["q1"].pop(ROUND)
    assert after == before
    assert Path(path).read_bytes() == before_questions
    assert entry["shuffle_seed"] == ROUND3_SHUFFLE_SEED and entry["criterion"] == CRITERION_DATE
    assert entry["index_version"] == VERSION and entry["annotator"] == "me"
    assert sorted(entry["pool"]) == sorted(POOL) and len(entry["relevant"]) == 1
    assert "flagged" not in entry


def test_relevant_is_stored_in_candidate_order_not_in_shuffled_order(tmp_path):
    target = _target(tmp_path)
    round3_session(target, console=quiet(), prompt=_interrupting("1-8"))
    entry = journal_dict(target.questions_path)["q1"][ROUND]
    assert entry["relevant"] == POOL and entry["pool"] != POOL


def test_a_run_resumes_at_the_first_question_without_an_entry(tmp_path):
    target = _target(tmp_path)
    round3_session(target, console=quiet(), prompt=_interrupting("n"))
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting())
    out = console.export_text()
    assert "2 questions left of 3" in out and "and the other one?" in out
    assert "does bybit cancel all?" not in out
    assert "1/3 questions recorded" in out


def test_a_skipped_question_comes_back_on_the_next_run(tmp_path):
    target = _target(tmp_path)
    round3_session(target, console=quiet(), prompt=_interrupting("", "n"))
    rounds = journal_dict(target.questions_path)
    assert ROUND not in rounds["q1"] and ROUND in rounds["q2"]
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting())
    assert "does bybit cancel all?" in console.export_text()


def test_an_interrupt_keeps_the_verdicts_given_and_a_finished_round_says_so(tmp_path):
    target = _target(tmp_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("n"))
    assert "verdicts so far are saved" in console.export_text()
    round3_session(target, console=quiet(), prompt=Script("n", "n"))
    console = quiet()
    round3_session(target, console=console, prompt=Script())
    assert "complete: 3/3" in console.export_text()


def test_u_reopens_the_previous_question_without_its_marks(tmp_path):
    target = _target(tmp_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("1", "u", "2-3"))
    entry = journal_dict(target.questions_path)["q1"][ROUND]
    assert len(entry["relevant"]) == 2
    assert ROUND not in journal_dict(target.questions_path)["q2"]


def test_skipping_a_reopened_question_keeps_its_verdict(tmp_path):
    target = _target(tmp_path)
    round3_session(target, console=quiet(), prompt=_interrupting("1", "u", ""))
    assert len(journal_dict(target.questions_path)["q1"][ROUND]["relevant"]) == 1


def test_u_reaches_back_one_question_only(tmp_path):
    target = _target(tmp_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("n", "n", "u", "u", "n"))
    rounds = journal_dict(target.questions_path)
    assert all(ROUND in rounds[q] for q in ("q1", "q2"))
    assert "u re-opens the previous question of this session." in console.export_text()


def test_u_does_not_cross_a_restart(tmp_path):
    target = _target(tmp_path)
    round3_session(target, console=quiet(), prompt=_interrupting("1"))
    before = journal_dict(target.questions_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("u"))
    assert journal_dict(target.questions_path) == before
    assert "u re-opens the previous question of this session." in console.export_text()


def _to_unanswerable(target: Round3Target) -> None:
    round3_session(target, console=quiet(), prompt=_interrupting("n", "n"))


def test_an_unanswerable_question_refuses_numbers(tmp_path):
    target = _target(tmp_path)
    _to_unanswerable(target)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("1", "2-3", "n"))
    entry = journal_dict(target.questions_path)["q3"][ROUND]
    assert entry["relevant"] == [] and "flagged" not in entry
    assert "Expected n or x" in console.export_text()


def test_x_flags_an_unanswerable_question_with_empty_relevant(tmp_path):
    target = _target(tmp_path)
    _to_unanswerable(target)
    round3_session(target, console=quiet(), prompt=_interrupting("x"))
    entry = journal_dict(target.questions_path)["q3"][ROUND]
    assert entry["relevant"] == [] and entry["flagged"] is True


def test_x_means_nothing_on_an_answerable_question(tmp_path):
    target = _target(tmp_path)
    console = quiet()
    round3_session(target, console=console, prompt=_interrupting("x", "n"))
    assert "flagged" not in journal_dict(target.questions_path)["q1"][ROUND]
    assert "Expected numbers" in console.export_text()


def test_the_screen_shows_the_questions_it_is_given_not_the_question_file(tmp_path):
    target = _target(tmp_path)
    path = Path(target.questions_path)
    assert load_questions(path)[0].category == "fact"
    shown = [
        Question("q1", "does bybit cancel all?", "gold", ["bybit"], "fact", category="noisy"),
        *_questions()[1:],
    ]
    console = quiet()
    round3_session(
        Round3Target(str(path), shown, {}, CHUNKS, VERSION, "me"),
        console=console,
        prompt=_interrupting(),
    )
    assert "category: noisy" in console.export_text()


def test_every_question_is_checked_before_the_first_is_shown(tmp_path):
    target = _target(tmp_path)
    missing = {k: v for k, v in CHUNKS.items() if k != "p3"}
    bad = Round3Target(target.questions_path, target.questions, {}, missing, VERSION, "me")
    with pytest.raises(UserError, match="not in the current index"):
        round3_session(bad, console=quiet(), prompt=Script())
    with pytest.raises(UserError, match="outside the round"):
        round3_session(
            Round3Target(
                target.questions_path, target.questions, {"q9": []}, CHUNKS, VERSION, "me"
            ),
            console=quiet(),
            prompt=Script(),
        )
    other = Question("q9", "never judged?", "gold", ["bybit"], "fact", category="fact")
    with pytest.raises(UserError, match="no round-1 judgment"):
        round3_session(
            Round3Target(target.questions_path, [other], {}, CHUNKS, VERSION, "me"),
            console=quiet(),
            prompt=Script(),
        )
