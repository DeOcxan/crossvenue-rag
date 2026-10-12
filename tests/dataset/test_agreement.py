"""Tests for `crossvenue_rag.dataset.agreement`."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.dataset.agreement import (
    Judgements,
    KappaRequest,
    KappaStudy,
    compare_rounds,
    kappa_study,
)
from crossvenue_rag.dataset.journal import Journal
from crossvenue_rag.dataset.questions import save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.models import Question
from tests.conftest import augmentation, correction, write_journal


def test_rounds_compare_only_pairs_judged_twice():
    """A chunk absent from a round's pool is unjudged in that round and enters no comparison."""
    first = Judgements(1, "amo", {("q1", "c1"): True, ("q1", "c2"): False, ("q2", "c3"): True})
    second = Judgements(2, "amo", {("q1", "c1"): True, ("q1", "c2"): True})

    report = compare_rounds(first, second)
    assert report.pairs == 2  # ("q2", "c3") was not shown in round 2
    assert report.questions == 1
    assert report.disagreements == 1
    assert report.interpretation

    with pytest.raises(UserError, match="share no judged"):
        compare_rounds(first, Judgements(2, "amo", {("q9", "c9"): True}))


def _rounds() -> Journal:
    """Two questions judged twice; q1 was corrected in review and augmented."""
    return Journal.from_json({
        "q1": {
            "1": {
                "pool": ["a", "b", "c"],
                "relevant": ["b", "x"],
                "corrections": [correction(["a"]), correction(["b"])],
                "augmentations": [augmentation(["x"], ["x"])],
            },
            "2": {"pool": ["a", "b", "c", "x"], "relevant": ["b", "x"]},
        },
        "q2": {
            "1": {"pool": ["d", "e"], "relevant": ["d"]},
            "2": {"pool": ["d", "e"], "relevant": ["e"]},
        },
    })  # fmt: skip


def test_first_pass_is_the_verdict_before_the_earliest_correction_plus_augmentation():
    rounds = _rounds()
    q1, q2 = rounds.first("q1"), rounds.first("q2")
    assert q1 is not None and q2 is not None
    assert q1.first_pass_relevant() == {"a", "x"}
    assert q2.first_pass_relevant() == {"d"}


def test_agreement_variants_are_the_four_registered_readings_in_order():
    from crossvenue_rag.dataset.agreement import agreement_variants

    variants = agreement_variants(_rounds(), 1, 2, exposed=["q2"])
    readings = [(v.reference, v.scope) for v in variants]
    assert readings == [
        ("corrected", "all"),
        ("corrected", "unexposed"),
        ("first pass", "all"),
        ("first pass", "unexposed"),
    ]
    by = {(v.reference, v.scope): v.report for v in variants}
    # q1 corrected equals round 2 exactly; q2 disagrees on both chunks.
    assert by[("corrected", "unexposed")].disagreements == 0
    assert by[("corrected", "all")].disagreements == 2
    # First pass marked "a" instead of "b" on q1.
    assert by[("first pass", "unexposed")].disagreements == 2
    assert by[("corrected", "unexposed")].questions == 1


def test_no_exposed_list_gives_the_two_references_over_all_questions():
    from crossvenue_rag.dataset.agreement import agreement_variants

    assert [(v.reference, v.scope) for v in agreement_variants(_rounds(), 1, 2)] == [
        ("corrected", "all"),
        ("first pass", "all"),
    ]


def test_kappa_by_category_puts_the_most_disagreement_first():
    from crossvenue_rag.dataset.agreement import kappa_by_category

    first = Judgements(1, "a", {("q1", "c1"): True, ("q1", "c2"): False, ("q2", "c3"): True})
    second = Judgements(2, "a", {("q1", "c1"): True, ("q1", "c2"): False, ("q2", "c3"): False})
    rows = kappa_by_category(first, second, {"q1": "fact", "q2": "comparison", "q3": "noisy"})
    assert [(c, r.disagreements, r.pairs) for c, r in rows] == [
        ("comparison", 1, 1),
        ("fact", 0, 2),
    ]


def test_retest_gaps_are_days_between_the_two_judgements():
    from crossvenue_rag.dataset.agreement import retest_gaps

    rounds = Journal.from_json({
        "q1": {
            "1": {"annotated_at": "2026-09-24T12:00:00Z"},
            "2": {"annotated_at": "2026-10-10T00:00:00Z"},
        },
        "q2": {"1": {"annotated_at": "2026-09-24T12:00:00Z"}},
    })  # fmt: skip
    assert retest_gaps(rounds, 1, 2) == {"q1": pytest.approx(15.5)}


def entry(stamp: str, relevant: list[str], **extra) -> dict:
    return {
        "pool": ["c1", "c2"],
        "relevant": relevant,
        "index_version": "v",
        "annotated_at": stamp,
        **extra,
    }


def two_rounds(tmp_path: Path, second_stamp: str = "2026-09-20T12:00:00Z") -> Path:
    """Two questions judged in rounds 1 and 2; q2 disagrees on c2."""
    path = tmp_path / "q.jsonl"
    save_questions(
        path,
        [
            Question("q1", "a?", "gold", ["bybit"], "fact", category="fact"),
            Question("q2", "b?", "gold", ["bybit", "okx"], "comparison"),
        ],
    )
    first = "2026-09-01T12:00:00Z"
    write_journal(
        path,
        {
            "q1": {"1": entry(first, ["c1"]), "2": entry(second_stamp, ["c1"])},
            "q2": {"1": entry(first, ["c1"]), "2": entry(second_stamp, ["c1", "c2"])},
        },
    )
    return path


def test_the_study_holds_every_reading(tmp_path: Path):
    path = two_rounds(tmp_path)
    exposed = tmp_path / "exposed.txt"
    exposed.write_text("q2\n", encoding="utf-8")
    study = kappa_study(KappaRequest(str(path), exposed=exposed))
    assert isinstance(study, KappaStudy)
    assert study.report.pairs == 4 and study.report.disagreements == 1
    assert study.gaps == {"q1": 19.0, "q2": 19.0} and study.short == []
    # q2 has no category and falls back to its qtype; most disagreement first.
    assert [category for category, _ in study.by_category] == ["comparison", "fact"]
    assert [(v.reference, v.scope) for v in study.variants] == [
        ("corrected", "all"),
        ("corrected", "unexposed"),
        ("first pass", "all"),
        ("first pass", "unexposed"),
    ]
    assert study.variants[1].report.disagreements == 0


def test_a_missing_exposed_list_gives_the_readings_over_all_questions(tmp_path: Path):
    path = two_rounds(tmp_path)
    for exposed in (None, tmp_path / "absent.txt"):
        study = kappa_study(KappaRequest(str(path), exposed=exposed))
        assert [v.scope for v in study.variants] == ["all", "all"]


def test_a_short_retest_gap_is_refused_unless_asked_for(tmp_path: Path):
    path = two_rounds(tmp_path, second_stamp="2026-09-05T12:00:00Z")
    with pytest.raises(UserError, match="2 questions re-judged less than 14 days"):
        kappa_study(KappaRequest(str(path)))
    study = kappa_study(KappaRequest(str(path), short_gap=True))
    assert study.short == ["q1", "q2"] and study.gaps["q1"] == 4.0


def test_no_round_history_is_refused(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact")])
    with pytest.raises(UserError, match="no annotation history"):
        kappa_study(KappaRequest(str(path)))


def test_an_exposed_list_naming_every_round_2_question_is_refused_by_cause(tmp_path: Path):
    path = two_rounds(tmp_path)
    exposed = tmp_path / "exposed.txt"
    exposed.write_text("q1\nq2\n", encoding="utf-8")
    with pytest.raises(UserError, match="every question judged in round 2 is listed as exposed"):
        kappa_study(KappaRequest(str(path), exposed=exposed))


def _naive_interval(first: Judgements, second: Judgements, resamples: int, seed: int):
    """The interval drawn one resample at a time through `stats.cohens_kappa_or_none`."""
    import numpy as np

    from crossvenue_rag.stats import cohens_kappa_or_none

    shared = set(first.labels) & set(second.labels)
    questions = sorted({q for q, _ in shared})
    by_question = {
        q: [(first.labels[k], second.labels[k]) for k in sorted(shared) if k[0] == q]
        for q in questions
    }
    rng = np.random.default_rng(seed)
    kappas, undefined = [], 0
    for row in rng.integers(0, len(questions), size=(resamples, len(questions))):
        kappa = cohens_kappa_or_none([p for i in row for p in by_question[questions[i]]])
        if kappa is None:
            undefined += 1
        else:
            kappas.append(kappa)
    return [float(x) for x in np.percentile(kappas, (2.5, 97.5))], undefined


def _three_questions() -> tuple[Judgements, Judgements]:
    first = Judgements(1, "a", {
        ("q1", "c1"): True, ("q1", "c2"): False, ("q1", "c3"): False,
        ("q2", "c4"): True, ("q2", "c5"): True,
        ("q3", "c6"): False, ("q3", "c7"): False, ("q3", "c8"): True,
    })  # fmt: skip
    second = Judgements(2, "a", {
        ("q1", "c1"): True, ("q1", "c2"): True, ("q1", "c3"): False,
        ("q2", "c4"): True, ("q2", "c5"): False,
        ("q3", "c6"): False, ("q3", "c7"): True, ("q3", "c8"): True,
        ("q4", "c9"): True,
    })  # fmt: skip
    return first, second


def test_the_kappa_interval_resamples_questions_and_pools_their_pairs():
    from crossvenue_rag.dataset.agreement import Bootstrap, kappa_interval

    first, second = _three_questions()
    interval = kappa_interval(first, second, Bootstrap(500, 7))
    (low, high), undefined = _naive_interval(first, second, 500, 7)
    assert interval.low == pytest.approx(low) and interval.high == pytest.approx(high)
    assert interval.undefined == undefined and interval.resamples == 500
    assert interval == kappa_interval(first, second, Bootstrap(500, 7))


def test_a_resample_with_one_label_throughout_is_counted_undefined():
    from crossvenue_rag.dataset.agreement import Bootstrap, kappa_interval

    # Every pair irrelevant in both rounds: kappa is undefined in every resample.
    first = Judgements(1, "a", {("q1", "c1"): False, ("q2", "c2"): False})
    second = Judgements(2, "a", {("q1", "c1"): False, ("q2", "c2"): False})
    interval = kappa_interval(first, second, Bootstrap(50, 1))
    assert (interval.low, interval.high, interval.undefined) == (None, None, 50)


def test_the_verdict_table_splits_disagreements_by_the_round_that_said_relevant():
    from crossvenue_rag.dataset.agreement import direction_by_category, verdict_table

    first, second = _three_questions()
    table = verdict_table(first, second)
    assert (table.both_relevant, table.first_only, table.second_only, table.both_irrelevant) == (
        3, 1, 2, 2,
    )  # fmt: skip
    assert table.disagreements == compare_rounds(first, second).disagreements
    rows = direction_by_category(first, second, {"q1": "fact", "q2": "fact", "q3": "noisy"})
    assert [(c, t.first_only, t.second_only) for c, t in rows] == [
        ("fact", 1, 1),
        ("noisy", 0, 1),
    ]


def test_source_chunks_are_counted_by_verdict_or_as_unjudged():
    from crossvenue_rag.dataset.agreement import source_chunk_verdicts

    first, second = _three_questions()
    # c5 was dropped in round 2; c0 was never shown; q4 was not judged in round 1.
    sources = source_chunk_verdicts(first, second, {"q2": ["c5", "c0"], "q3": ["c8"], "q4": ["c9"]})
    v = sources.verdicts
    assert (v.both_relevant, v.first_only, v.second_only, v.both_irrelevant) == (1, 1, 0, 0)
    assert sources.unjudged == 1


def test_the_study_carries_intervals_only_when_a_bootstrap_is_given(tmp_path: Path):
    from crossvenue_rag.dataset.agreement import Bootstrap

    path = two_rounds(tmp_path)
    plain = kappa_study(KappaRequest(str(path)))
    assert all(v.interval is None for v in plain.variants)
    assert (plain.direction.first_only, plain.direction.second_only) == (0, 1)
    assert [c for c, _ in plain.direction_by_category] == ["comparison", "fact"]
    assert plain.sources.unjudged == 0 and plain.sources.verdicts.disagreements == 0
    study = kappa_study(KappaRequest(str(path), bootstrap=Bootstrap(200, 3)))
    assert all(v.interval is not None and v.interval.resamples == 200 for v in study.variants)
