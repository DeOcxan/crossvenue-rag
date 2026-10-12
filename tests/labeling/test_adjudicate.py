"""Tests for `crossvenue_rag.labeling.adjudicate`: the third pass over rounds 1 and 2."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from crossvenue_rag.dataset.agreement import KappaRequest, kappa_study
from crossvenue_rag.dataset.journal import ADJUDICATED, Journal, Judgment, load_rounds
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.adjudicate import (
    CRITERION_DATE,
    AdjudicationTarget,
    adjudicate_session,
    decided,
    disagreeing_pairs,
)
from crossvenue_rag.labeling.pool import RETEST_SHUFFLE_SEED, retest_order
from crossvenue_rag.models import Question
from tests.conftest import journal_dict, write_journal
from tests.labeling.conftest import VERSION, Script, chunk, quiet

CHUNKS = {c: chunk(c, "binance") for c in ("a", "b", "c", "d", "e")}


def _entry(relevant: list[str], stamp: str, **extra) -> dict:
    return {
        "pool": ["a", "b", "c", "d"],
        "relevant": relevant,
        "index_version": VERSION,
        "annotated_at": stamp,
        **extra,
    }


def _journal(tmp_path: Path) -> str:
    """q1: a in round 1, a b c in round 2; q2: d, then nothing; q3: round 1 only."""
    path = tmp_path / "questions.jsonl"
    save_questions(
        path,
        [
            Question(q, f"{q}?", "gold", ["binance"], "fact", category="fact")
            for q in ("q1", "q2", "q3")
        ],
    )
    first, second = "2026-09-24T12:00:00Z", "2026-10-10T12:00:00Z"
    retest = {"shuffle_seed": RETEST_SHUFFLE_SEED}
    write_journal(
        path,
        {
            "q1": {"1": _entry(["a"], first), "2": _entry(["a", "b", "c"], second, **retest)},
            "q2": {"1": _entry(["d"], first), "2": _entry([], second, **retest)},
            "q3": {"1": _entry(["a"], first)},
        },
    )
    return str(path)


def second(journal: Journal, question_id: str) -> Judgment:
    entry = journal.get(question_id).judgment(2)
    assert entry is not None
    return entry


def _target(path: str) -> AdjudicationTarget:
    return AdjudicationTarget(path, CHUNKS, VERSION, "me")


def test_the_pairs_are_the_disagreements_by_question_in_round_2_screen_order(tmp_path):
    path = _journal(tmp_path)
    rounds = load_rounds(path)
    pairs = disagreeing_pairs(rounds)
    assert [(p.question_id, p.chunk_id, p.first, p.second) for p in pairs] == [
        ("q1", "b", False, True),
        ("q1", "c", False, True),
        ("q2", "d", True, False),
    ]
    for p in pairs:
        assert second(rounds, p.question_id).pool[p.number - 1] == p.chunk_id


def _late_augmentation(shown: list[str]) -> dict:
    return {
        "at": "2026-10-11T12:00:00Z", "annotator": "me", "index_version": VERSION,
        "source": "run", "shown": shown, "relevant": shown, "coverage_changes": [],
    }  # fmt: skip


def test_a_chunk_round_1_judged_after_round_2_is_no_pair(tmp_path):
    path = tmp_path / "questions.jsonl"
    save_questions(path, [Question("q1", "q1?", "gold", ["binance"], "fact", category="fact")])
    # What `augment.record_augmentation` writes: the relevant chunk joins round 1's `relevant`.
    late = _late_augmentation(["e"])
    write_journal(
        path,
        {
            "q1": {
                "1": _entry(["a", "e"], "2026-09-24T12:00:00Z", augmentations=[late]),
                "2": _entry(["a"], "2026-10-10T12:00:00Z", shuffle_seed=RETEST_SHUFFLE_SEED),
            }
        },
    )
    assert disagreeing_pairs(load_rounds(path)) == []
    assert kappa_study(KappaRequest(str(path), short_gap=True)).report.disagreements == 0


def test_a_late_augmentation_leaves_the_round_2_screen_numbers(tmp_path):
    path = tmp_path / "questions.jsonl"
    save_questions(path, [Question("q1", "q1?", "gold", ["binance"], "fact", category="fact")])
    pool = ["a", "b", "c", "d", "f", "g", "h"]
    shown = retest_order("q1", Judgment(pool=pool, relevant=[]), RETEST_SHUFFLE_SEED)
    retest = {"pool": shown, "shuffle_seed": RETEST_SHUFFLE_SEED}
    write_journal(
        path,
        {
            "q1": {
                "1": _entry(["a", "e"], "2026-09-24T12:00:00Z", pool=pool,
                            augmentations=[_late_augmentation(["e"])]),
                "2": _entry(["a", "b", "c"], "2026-10-10T12:00:00Z", **retest),
            }
        },
    )  # fmt: skip
    pairs = disagreeing_pairs(load_rounds(path))
    assert [p.chunk_id for p in pairs] == [c for c in shown if c in ("b", "c")]
    assert [p.number for p in pairs] == [shown.index(p.chunk_id) + 1 for p in pairs]


def test_each_decision_is_journaled_apart_from_rounds_1_and_2(tmp_path):
    path = _journal(tmp_path)
    before = journal_dict(path)
    kappa_before = kappa_study(KappaRequest(path)).report
    adjudicate_session(_target(path), console=quiet(), prompt=Script("y", "n", "n"))
    after = journal_dict(path)
    for q in before:
        assert {k: v for k, v in after[q].items() if k != ADJUDICATED} == before[q]
    assert kappa_study(KappaRequest(path)).report == kappa_before
    entry = after["q1"][ADJUDICATED]
    assert entry["criterion"] == CRITERION_DATE and entry["index_version"] == VERSION
    assert entry["annotator"] == "me" and "annotated_at" in entry
    rounds = load_rounds(path)
    pairs = disagreeing_pairs(rounds)
    assert [decided(rounds, p) for p in pairs] == [True, False, False]
    assert ADJUDICATED not in after["q3"]


def test_back_redecides_the_previous_pair_and_view_commands_ask_again(tmp_path):
    path = _journal(tmp_path)
    console = quiet()
    script = Script("v1", "v9", "f", "x", "y", "b", "n", "y", "y")
    adjudicate_session(_target(path), console=console, prompt=script)
    rounds = load_rounds(path)
    assert [decided(rounds, p) for p in disagreeing_pairs(rounds)] == [False, True, True]
    out = console.export_text()
    assert "Expected v1 .. v2." in out and "Expected y, n, b, v<n> or f." in out
    assert "round 1: not relevant   round 2: relevant" in out
    assert "amended 2026-10-10" in out and "3/3 pairs decided" in out


def test_an_interrupt_saves_and_the_next_run_resumes_at_the_first_open_pair(tmp_path):
    path = _journal(tmp_path)

    def interrupt_after_one(answers=iter(["n"])):
        def prompt(message: str, **_: object) -> str:
            answer = next(answers, None)
            if answer is None:
                raise KeyboardInterrupt
            return answer

        return prompt

    console = quiet()
    adjudicate_session(_target(path), console=console, prompt=interrupt_after_one())
    assert "decisions so far are saved" in console.export_text()
    rounds = load_rounds(path)
    assert [decided(rounds, p) for p in disagreeing_pairs(rounds)] == [False, None, None]
    resumed = quiet()
    adjudicate_session(_target(path), console=resumed, prompt=Script("y", "y"))
    assert "1 decided" in resumed.export_text()
    rounds = load_rounds(path)
    assert [decided(rounds, p) for p in disagreeing_pairs(rounds)] == [False, True, True]


def test_a_disagreeing_chunk_missing_from_the_corpus_is_refused(tmp_path):
    path = _journal(tmp_path)
    chunks = copy.copy(CHUNKS)
    del chunks["b"]
    with pytest.raises(UserError, match="not in the corpus"):
        adjudicate_session(AdjudicationTarget(path, chunks, VERSION, "me"), console=quiet())


def test_no_disagreement_means_nothing_to_decide(tmp_path):
    path = tmp_path / "questions.jsonl"
    save_questions(path, [Question("q1", "q?", "gold", ["binance"], "fact")])
    write_journal(path, {"q1": {"1": _entry(["a"], "x"), "2": _entry(["a"], "y")}})
    console = quiet()
    adjudicate_session(_target(str(path)), console=console, prompt=Script())
    assert "0/0 pairs decided" in console.export_text()
