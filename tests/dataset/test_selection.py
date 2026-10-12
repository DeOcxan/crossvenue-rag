"""Tests for `crossvenue_rag.dataset.selection`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.dataset.records import QuestionRecord
from crossvenue_rag.dataset.selection import (
    TARGETS,
    Decision,
    accepted_records,
    append_decision,
    check_composition,
    load_decisions,
    progress,
    queue,
    rejection_summary,
    selection_path,
)
from crossvenue_rag.errors import UserError
from tests.conftest import REPO


def candidate(question_id: str, category: str = "comparison", **patch) -> QuestionRecord:
    base = {
        "id": question_id,
        "question": f"question {question_id} about rate limits on two venues",
        "tier": "gold",
        "venues": ["binance", "bybit"],
        "qtype": "comparison" if category == "comparison" else "fact",
        "category": category,
        "difficulty": "medium",
        "is_answerable": True,
        "expected_behavior": "answer",
    }
    if category == "unanswerable":
        base |= {"is_answerable": False, "expected_behavior": "refuse", "qtype": "fact"}
    if category == "ambiguous":
        base |= {"expected_behavior": "ask_clarification", "qtype": "fact"}
    if category == "adversarial":
        base |= {"adversarial_type": "near_duplicate_value", "qtype": "fact"}
    return QuestionRecord(**{**base, **patch})


def pool(**counts: int) -> list[QuestionRecord]:
    """A candidate pool with `counts` records per category."""
    out: list[QuestionRecord] = []
    for category, n in counts.items():
        out += [candidate(f"cv-{category[:3]}{i:03d}", category) for i in range(n)]
    return out


def full_pool() -> list[QuestionRecord]:
    return pool(**TARGETS)


def accept_all(candidates) -> dict[str, Decision]:
    return {r.id: Decision.now(r.id, "accept") for r in candidates}


def test_decisions_survive_an_interrupted_session(tmp_path: Path):
    """The log is appended after every keystroke, so Ctrl-C loses at most the
    question on screen."""
    log = tmp_path / "selection.jsonl"
    append_decision(log, Decision.now("cv-0001", "accept"))
    append_decision(log, Decision.now("cv-0002", "reject", "not grounded"))
    append_decision(log, Decision.now("cv-0003", "skip"))

    restored = load_decisions(log)
    assert set(restored) == {"cv-0001", "cv-0002", "cv-0003"}
    assert restored["cv-0002"].decision == "reject"
    assert restored["cv-0002"].reason == "not grounded"
    assert restored["cv-0001"].ts.endswith("Z")

    # A resumed session queues exactly the undecided ones, skips held back.
    candidates = [candidate(f"cv-000{i}") for i in range(1, 5)]
    assert [r.id for r in queue(candidates, restored)] == ["cv-0004"]
    assert [r.id for r in queue(candidates, restored, include_skipped=True)] == [
        "cv-0003",
        "cv-0004",
    ]


def test_changing_your_mind_is_another_line_not_an_edit(tmp_path: Path):
    """Append-only with last-write-wins keeps the history of the reversal."""
    log = tmp_path / "selection.jsonl"
    append_decision(log, Decision.now("cv-0001", "reject", "unrealistic phrasing"))
    append_decision(log, Decision.now("cv-0001", "accept"))

    assert load_decisions(log)["cv-0001"].decision == "accept"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [line["decision"] for line in lines] == ["reject", "accept"]


def test_missing_log_reads_as_no_decisions(tmp_path: Path):
    assert load_decisions(tmp_path / "absent.jsonl") == {}


def test_selection_log_sits_beside_the_question_file():
    assert selection_path("evals/questions.jsonl") == Path("evals/selection/selection.jsonl")


def test_progress_counts_against_targets_and_tracks_the_pool():
    candidates = pool(comparison=4, fact=2)
    decisions = {
        "cv-com000": Decision.now("cv-com000", "accept"),
        "cv-com001": Decision.now("cv-com001", "accept"),
        "cv-com002": Decision.now("cv-com002", "reject", "not grounded"),
        "cv-com003": Decision.now("cv-com003", "skip"),
    }
    state = progress(candidates, decisions)

    assert state.accepted["comparison"] == 2
    # A skip is still available; a reject is not.
    assert state.remaining["comparison"] == 1
    assert state.remaining["fact"] == 2
    assert state.shortfall("comparison") == TARGETS["comparison"] - 2
    assert state.total_target == 105


def test_progress_flags_a_target_the_pool_can_no_longer_reach():
    """Said during selection, while the pool can still be changed, before finalize."""
    candidates = pool(noisy=5)
    none: dict[str, Decision] = {}
    assert progress(candidates, none).reachable("noisy") is True

    rejected = {
        f"cv-noi{i:03d}": Decision.now(f"cv-noi{i:03d}", "reject", "other") for i in range(2)
    }
    assert progress(candidates, rejected).reachable("noisy") is False


def test_an_unanswerable_comparison_does_not_count_towards_the_subgroup():
    """It carries no gold chunks and is excluded from every recall average, so it
    shrinks the subgroup carrying the headline claim without appearing to."""
    candidates = [
        candidate("cv-0001"),
        candidate("cv-0002", is_answerable=False, expected_behavior="refuse", qtype="comparison"),
    ]
    decisions = {r.id: Decision.now(r.id, "accept") for r in candidates}
    state = progress(candidates, decisions)
    assert state.accepted["comparison"] == 2
    assert state.answerable_comparison == 1


def test_finalize_refuses_a_short_set():
    """Writing a set that misses a target would make every breakdown row a
    different size than the methodology claims."""
    candidates = full_pool()
    decisions = accept_all(candidates)
    dropped = next(r for r in candidates if r.category == "noisy")
    decisions[dropped.id] = Decision.now(dropped.id, "reject", "duplicate information need")

    report = check_composition(candidates, decisions)
    assert not report.ok
    assert report.missing == {"noisy": 1}
    assert any("noisy: 4 accepted, target 5" in p for p in report.problems())


def test_finalize_refuses_a_comparison_subgroup_that_is_short_on_answerables():
    candidates = full_pool()
    # One accepted comparison item is swapped for an unanswerable one.
    victim = next(r for r in candidates if r.category == "comparison")
    candidates = [r for r in candidates if r.id != victim.id] + [
        candidate(victim.id, is_answerable=False, expected_behavior="refuse", qtype="comparison")
    ]
    report = check_composition(candidates, accept_all(candidates))

    assert not report.ok
    assert not report.missing  # the count is right; the *answerable* count is not
    assert report.answerable_comparison == TARGETS["comparison"] - 1
    assert any("excluded from every recall average" in p for p in report.problems())


def test_finalize_accepts_a_complete_set():
    candidates = full_pool()
    report = check_composition(candidates, accept_all(candidates))
    assert report.ok, report.problems()
    assert sum(report.accepted.values()) == 105


def test_accepted_records_keep_candidate_file_order():
    """Stable output, so re-finalizing an unchanged selection produces an
    unchanged questions.jsonl."""
    candidates = pool(comparison=3)
    decisions = {"cv-com002": Decision.now("cv-com002", "accept")}
    decisions["cv-com000"] = Decision.now("cv-com000", "accept")
    assert [r.id for r in accepted_records(candidates, decisions)] == ["cv-com000", "cv-com002"]


def test_rejection_reasons_are_summarised_for_the_changelog():
    candidates = pool(comparison=3)
    decisions = {
        "cv-com000": Decision.now("cv-com000", "reject", "not grounded"),
        "cv-com001": Decision.now("cv-com001", "reject", "not grounded"),
        "cv-com002": Decision.now("cv-com002", "reject", "paraphrased heading"),
        "cv-gone": Decision.now("cv-gone", "reject", "other"),  # not in this pool
    }
    summary = rejection_summary(candidates, decisions)
    assert summary["not grounded"] == 2
    assert summary["paraphrased heading"] == 1
    assert "other" not in summary  # decisions about other pools are not counted


def test_an_accepted_id_missing_from_the_pool_is_an_error():
    """A stale selection log against a newer candidate version, which would
    otherwise write a question file with a hole in it."""
    candidates = full_pool()
    decisions = accept_all(candidates)
    decisions["cv-from-an-older-pool"] = Decision.now("cv-from-an-older-pool", "accept")

    report = check_composition(candidates, decisions)
    assert not report.ok
    assert report.unexpected == ["cv-from-an-older-pool"]


def test_the_shipped_candidate_pool_can_satisfy_every_target():
    """If it cannot, selection is impossible and the targets or the pool must
    change, and a test finds that before labeling starts."""
    from crossvenue_rag.dataset.records import load_records

    path = REPO / "evals/datasets/v0.1.2-candidates.jsonl"
    records = load_records(path)

    available = {category: 0 for category in TARGETS}
    answerable_comparison = 0
    for record in records:
        if record.category in available:
            available[record.category] += 1
        if record.category == "comparison" and record.is_answerable:
            answerable_comparison += 1

    short = {c: TARGETS[c] - n for c, n in available.items() if n < TARGETS[c]}
    assert not short, f"pool cannot fill: {short}"
    assert answerable_comparison >= TARGETS["comparison"]


def test_thin_categories_come_before_the_deep_ones():
    """Sequenced by risk of running out: whether the pool can fill ambiguous is
    answered in the first twenty minutes. A regeneration then does not land on top
    of an hour of comparison already reviewed."""
    from crossvenue_rag.dataset.selection import CATEGORY_ORDER

    thin = ["ambiguous", "noisy", "version_sensitive", "unanswerable"]
    deep = ["adversarial", "multihop", "fact"]
    assert CATEGORY_ORDER[: len(thin)] == thin
    assert CATEGORY_ORDER.index("comparison") < min(CATEGORY_ORDER.index(c) for c in deep)

    # And the queue really walks them in that order.
    candidates = pool(fact=1, ambiguous=1, comparison=1, noisy=1)
    walked = [r.category for r in queue(candidates, {})]
    assert walked.index("ambiguous") < walked.index("noisy") < walked.index("comparison")
    assert walked[-1] == "fact"


def test_unseen_shrinks_as_candidates_are_reviewed_but_available_tracks_skips():
    """The two numbers measure different things: `unseen` is "how much is left
    to look at", `available` is "how much could still be accepted"."""
    candidates = pool(ambiguous=5)
    ids = [r.id for r in candidates]

    fresh = progress(candidates, {})
    assert fresh.unseen["ambiguous"] == 5
    assert fresh.remaining["ambiguous"] == 5

    decisions = {
        ids[0]: Decision.now(ids[0], "accept"),
        ids[1]: Decision.now(ids[1], "reject", "not grounded"),
        ids[2]: Decision.now(ids[2], "skip"),
    }
    state = progress(candidates, decisions)
    assert state.unseen["ambiguous"] == 2  # three have been shown
    # A skip is still acceptable later; a reject is gone for good.
    assert state.remaining["ambiguous"] == 3
    assert state.shortfall("ambiguous") == TARGETS["ambiguous"] - 1
    # 1 accepted + 3 still available = 4, short of the target of 5.
    assert state.reachable("ambiguous") is False


def test_a_malformed_decision_line_is_refused_with_its_line(tmp_path):
    log = tmp_path / "selection.jsonl"
    log.write_text('{"id": "cv-0001", "decision": "accept"}\n', encoding="utf-8")
    with pytest.raises(UserError, match=r"selection.jsonl:1: not a decision record"):
        load_decisions(log)
