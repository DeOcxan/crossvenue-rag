"""Tests for `dataset.finalize`: writing the question file from accepted candidates."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.dataset import finalize
from crossvenue_rag.dataset.finalize import Finalized, finalize_selection
from crossvenue_rag.dataset.questions import load_questions
from crossvenue_rag.dataset.records import QuestionRecord
from crossvenue_rag.dataset.selection import TARGETS, Decision, append_decision
from crossvenue_rag.dataset.validate import ValidationReport
from crossvenue_rag.errors import UserError
from tests.dataset.factories import record

# Fields each category's records need to pass QuestionRecord's rules.
CATEGORY_FIELDS: dict[str, dict] = {
    "comparison": {"venues": ["bybit", "okx"], "qtype": "comparison"},
    "unanswerable": {"is_answerable": False, "expected_behavior": "refuse"},
    "adversarial": {"adversarial_type": "false premise"},
    "ambiguous": {"expected_behavior": "ask_clarification"},
}


def pool() -> list[QuestionRecord]:
    """Exactly the target number of candidates per category, plus one extra fact."""
    out = [
        record(id=f"{category}-{i}", category=category, **CATEGORY_FIELDS.get(category, {}))
        for category, target in TARGETS.items()
        for i in range(target)
    ]
    return [*out, record(id="extra")]


def decide(path: Path, candidates: list[QuestionRecord], *, accept_all: bool = True) -> None:
    """Records an accept for every target candidate (all but the last when not
    `accept_all`) and a reject for the extra one."""
    kept = [r for r in candidates if r.id != "extra"]
    if not accept_all:
        kept = kept[:-1]
    for r in kept:
        append_decision(path, Decision.now(r.id, "accept"))
    append_decision(path, Decision.now("extra", "reject", "not grounded"))


@pytest.fixture
def validations(monkeypatch) -> list:
    """Replaces the dataset validation (which would load the encoder) and records
    each call."""
    calls: list = []

    def validate(path, cfg, *, check_near_duplicates: bool = True) -> ValidationReport:
        calls.append((path, check_near_duplicates))
        return ValidationReport(warnings=["checked"])

    monkeypatch.setattr(finalize, "validate_dataset", validate)
    return calls


def test_a_set_meeting_the_targets_is_written_and_validated(tmp_path: Path, validations: list):
    candidates = pool()
    selection, questions = tmp_path / "selection.jsonl", tmp_path / "questions.jsonl"
    decide(selection, candidates)
    done = finalize_selection(Config(), candidates, str(questions), str(selection))
    done.require_composition()
    assert done.composition.ok and done.written == sum(TARGETS.values())
    assert done.reasons == Counter({"not grounded": 1})
    assert done.validation is not None and done.validation.warnings == ["checked"]
    assert validations == [(str(questions), True)]
    written = load_questions(questions)
    assert [q.id for q in written] == [r.id for r in candidates if r.id != "extra"]


def test_a_short_set_is_reported_and_nothing_is_written(tmp_path: Path, validations: list):
    candidates = pool()
    selection, questions = tmp_path / "selection.jsonl", tmp_path / "questions.jsonl"
    decide(selection, candidates, accept_all=False)
    short = finalize_selection(Config(), candidates, str(questions), str(selection))
    assert not short.composition.ok and short.composition.missing == {"noisy": 1}
    assert (short.written, short.reasons, short.validation) == (None, None, None)
    assert not questions.exists() and validations == []
    with pytest.raises(UserError, match="does not meet the target composition"):
        short.require_composition()


def test_no_decisions_are_refused(tmp_path: Path):
    with pytest.raises(UserError, match="no decisions recorded"):
        finalize_selection(Config(), pool(), str(tmp_path / "q.jsonl"), str(tmp_path / "s.jsonl"))


def test_a_finalized_outcome_refuses_a_short_or_unexpected_set_only():
    from crossvenue_rag.dataset.selection import CompositionReport

    met = Finalized(CompositionReport(Counter(TARGETS), {}, TARGETS["comparison"], []))
    met.require_composition()
    short = Finalized(CompositionReport(Counter(TARGETS), {"fact": 1}, TARGETS["comparison"], []))
    unexpected = Finalized(CompositionReport(Counter(TARGETS), {}, TARGETS["comparison"], ["x"]))
    for refused in (short, unexpected):
        with pytest.raises(UserError, match="does not meet the target composition"):
            refused.require_composition()
