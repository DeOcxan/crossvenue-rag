"""Tests for `reporting.compute.agreement`, on journals written under tmp_path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from crossvenue_rag.config_loading import DEFAULT_CONFIG
from crossvenue_rag.dataset.journal import rounds_path
from crossvenue_rag.reporting.compute import agreement
from crossvenue_rag.reporting.run_ids import (
    D23_ANSWER_CHUNKS,
    GOLD,
    GOLD_ADJUDICATED,
    ROUND2_EXPOSED,
    WORKING_QUESTIONS,
)
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import adjudication, relabel
from tests.reporting.conftest import write_file, write_gold

ROOT = Path(__file__).resolve().parents[3]


def _entry(stamp: str, relevant: list[str]) -> dict:
    return {"pool": ["s1", "c2", "c3", "c4"], "relevant": relevant, "annotated_at": stamp}


def _journals(root: Path, *, gold_round1: list[str] | None = None) -> Sources:
    """q1 is re-judged and marks c2 in round 2 only; q2, a fact question, s1 in round 1 only."""
    questions = [
        {"id": "q1", "source_chunk_ids": ["s1", "s9"]},
        {"id": "q2", "category": "fact", "qtype": "fact", "source_chunk_ids": ["s1"]},
    ]
    first, second = "2026-09-24T12:00:00Z", "2026-10-10T12:00:00Z"
    working = {
        "q1": {"1": _entry(first, ["s1"]), "2": _entry(second, ["s1", "c2"])},
        "q2": {"1": _entry(first, ["s1", "c3"]), "2": _entry(second, ["c3"])},
    }
    for path in (GOLD, WORKING_QUESTIONS):
        write_gold(root, questions, path=path)
    write_file(root, str(rounds_path(WORKING_QUESTIONS)), json.dumps(working))
    gold = {q: {"1": dict(e["1"])} for q, e in working.items()}
    if gold_round1 is not None:
        gold["q1"]["1"]["relevant"] = gold_round1
    write_file(root, str(rounds_path(GOLD)), json.dumps(gold))
    write_file(root, ROUND2_EXPOSED, "q2\n")
    (root / DEFAULT_CONFIG).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / DEFAULT_CONFIG, root / DEFAULT_CONFIG)
    return Sources(root)


def test_the_study_is_the_kappa_command_with_the_configured_bootstrap(tmp_path):
    src = _journals(tmp_path)
    cfg = src.default_config.eval
    assert agreement.bootstrap(src).resamples == cfg.bootstrap_resamples
    v = agreement.registered(src)
    assert (v.reference, v.scope, v.report.pairs, v.report.disagreements) == (
        "corrected", "all", 8, 2,
    )  # fmt: skip
    assert v.interval is not None and v.interval.resamples == cfg.bootstrap_resamples
    assert agreement.reading(src, "corrected", "unexposed").report.questions == 1
    with pytest.raises(ValueError, match="no reading of round 1 as corrected over nowhere"):
        agreement.reading(src, "corrected", "nowhere")
    assert agreement.exposed_count(src) == 1


def test_relevant_counts_and_the_source_chunks_come_from_the_working_journal(tmp_path):
    src = _journals(tmp_path)
    assert agreement.relevant_counts(src) == (3, 3)
    sources = agreement.study(src).sources
    assert (sources.verdicts.both_relevant, sources.verdicts.first_only) == (1, 1)
    assert sources.unjudged == 1  # q1's s9 was never shown
    assert agreement.sensitivity_spread(src) >= 0.0


def test_round_1_is_held_to_the_gold_journal(tmp_path):
    assert agreement.round1_is_gold(_journals(tmp_path / "same"))
    assert not agreement.round1_is_gold(_journals(tmp_path / "other", gold_round1=["c4"]))


def test_intervals_hold_the_registered_kappa_only_when_each_one_contains_it(tmp_path):
    src = _journals(tmp_path)
    study = agreement.study(src)
    kappa = agreement.registered(src).report.kappa
    for v in study.variants:
        assert v.interval is not None
        v.interval.low, v.interval.high = kappa - 0.1, kappa + 0.1
    assert agreement.intervals_hold_registered(src)
    last = study.variants[-1].interval
    assert last is not None
    last.high = kappa - 0.05
    assert not agreement.intervals_hold_registered(src)


def test_the_d23_questions_and_the_round_1_criterion_questions_are_counted_apart(tmp_path):
    from crossvenue_rag.reporting.run_ids import D23_ANSWER_CHUNKS, ROUND2_SAMPLE

    write_gold(tmp_path, [{"id": q} for q in ("q1", "q2", "q3", "q4")])
    write_file(tmp_path, ROUND2_SAMPLE, "# sample\nq1\n")
    write_file(
        tmp_path,
        D23_ANSWER_CHUNKS,
        '{"id": "q1", "chunk_ids": []}\n{"id": "q3", "chunk_ids": ["c"]}\n',
    )
    src = Sources(tmp_path)
    assert agreement.d23_questions(src) == ["q1", "q3"]
    assert agreement.outside_sample(src) == ["q2", "q3", "q4"]
    assert agreement.round1_criterion_questions(src) == ["q2", "q4"]


def test_the_adjudicated_version_is_compared_with_the_gold_set(tmp_path):
    from crossvenue_rag.dataset.journal import ADJUDICATED, RELABELED
    from crossvenue_rag.reporting.run_ids import GOLD_ADJUDICATED

    write_gold(
        tmp_path,
        [
            {"id": "q1", "gold_chunk_ids": ["a"]},
            {"id": "q2", "gold_chunk_ids": ["b"], "category": "fact", "qtype": "fact"},
            {"id": "q3", "category": "unanswerable", "qtype": "fact", "is_answerable": False,
             "expected_behavior": "refuse", "venues": ["bybit"]},
            {"id": "q4", "category": "unanswerable", "qtype": "fact", "is_answerable": False,
             "expected_behavior": "refuse", "venues": ["bybit"]},
        ],
    )  # fmt: skip
    write_gold(
        tmp_path,
        [
            {"id": "q1", "gold_chunk_ids": ["a", "c"]},
            {"id": "q2", "gold_chunk_ids": ["b"], "category": "fact", "qtype": "fact"},
            {"id": "q3", "gold_chunk_ids": ["x"], "category": "fact", "qtype": "fact",
             "venues": ["bybit"]},
            {"id": "q4", "category": "unanswerable", "qtype": "fact", "is_answerable": False,
             "expected_behavior": "refuse", "venues": ["bybit"]},
        ],
        path=GOLD_ADJUDICATED,
    )  # fmt: skip
    decisions = {"c": True, "d": False}
    journal = {
        "q1": {"1": {"relevant": ["a"]}, ADJUDICATED: adjudication(decisions)},
        "q2": {"1": {"relevant": ["b"]}, ADJUDICATED: adjudication({"e": False})},
        "q3": {"1": {"relevant": []}, RELABELED: relabel([], ["x"])},
        "q4": {"1": {"relevant": []}, RELABELED: relabel([], [])},
    }
    write_file(tmp_path, str(rounds_path(GOLD_ADJUDICATED)), json.dumps(journal))
    src = Sources(tmp_path)
    assert agreement.version_counts(src.questions) == agreement.VersionCounts(2, 2, 1, 2)
    assert agreement.version_counts(agreement.adjudicated_gold(src)) == agreement.VersionCounts(
        3, 4, 2, 1
    )
    assert agreement.changed_gold(src) == ["q1", "q3"]
    assert agreement.adjudicated_questions(src) == ["q1", "q2"]
    assert agreement.relabeled_questions(src) == ["q3", "q4"]
    assert agreement.adjudication_changed(src) == ["q1"]
    assert agreement.adjudicated_pairs(src) == 3
    assert agreement.recategorized(src) == ["q3"]


def test_the_adjudication_is_split_by_the_round_each_decision_agrees_with(tmp_path):
    from crossvenue_rag.reporting.run_ids import GOLD_ADJUDICATED

    write_gold(tmp_path, [{"id": "q1"}], path=GOLD_ADJUDICATED)
    first = {"pool": ["a", "b", "c"], "relevant": ["a"]}
    second = {"pool": ["a", "b", "c"], "relevant": ["b", "c"]}
    entry = {
        "1": first,
        "2": second,
        "adjudicated": adjudication({"a": False, "b": True, "c": False}),
    }
    write_file(tmp_path, str(rounds_path(GOLD_ADJUDICATED)), json.dumps({"q1": entry}))
    split = agreement.adjudication_split(Sources(tmp_path))
    assert (split.as_first, split.as_second) == (1, 2)
    entry["adjudicated"] = adjudication({"z": True})  # judged alike (not relevant) in both
    write_file(tmp_path, str(rounds_path(GOLD_ADJUDICATED)), json.dumps({"q1": entry}))
    with pytest.raises(ValueError, match="alike in both rounds"):
        agreement.adjudication_split(Sources(tmp_path))


def test_round_3_counts_the_chunks_round_1_judged_plus_the_added_ones(tmp_path):
    write_gold(
        tmp_path,
        [{"id": "q1", "category": "fact", "qtype": "fact"}, {"id": "q2", "category": "noisy"}],
        path=GOLD_ADJUDICATED,
    )
    pool = {"pool": ["a", "b"], "relevant": [], "annotated_at": "2026-09-24T12:00:00Z"}
    write_file(
        tmp_path,
        str(rounds_path(GOLD_ADJUDICATED)),
        json.dumps({"q1": {"1": pool}, "q2": {"1": pool}}),
    )
    write_file(
        tmp_path,
        D23_ANSWER_CHUNKS,
        '{"id": "q1", "chunk_ids": ["a", "x"]}\n{"id": "q2", "chunk_ids": []}\n',
    )
    src = Sources(tmp_path)
    assert agreement.round3_added(src) == {"q1": ["a", "x"]}
    sizes = agreement.round3_sizes(src)
    assert list(sizes) == ["fact", "noisy"]
    assert (sizes["fact"].questions, sizes["fact"].decisions) == (1, 3)
    assert (sizes["noisy"].questions, sizes["noisy"].decisions) == (1, 2)
