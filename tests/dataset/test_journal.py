"""Tests for `crossvenue_rag.dataset.journal`: the typed judgment journal and its file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.dataset.journal import (
    Correction,
    Journal,
    Judgment,
    load_judged_sets,
    load_rounds,
    read_journal,
    rounds_path,
    save_rounds,
)
from crossvenue_rag.errors import UserError
from tests.conftest import adjudication, augmentation, correction, relabel
from tests.published import ROOT

COMMITTED = sorted(
    [
        ROOT / "evals/annotations/questions.rounds.json",
        *(ROOT / "evals/datasets/annotations").glob("*.rounds.json"),
    ]
)


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: p.name)
def test_every_committed_journal_reads_and_writes_back_byte_identical(path: Path):
    assert read_journal(path).to_text() == path.read_text(encoding="utf-8")


def test_a_record_keeps_the_key_order_it_was_read_in_and_new_keys_follow_it():
    journal = Journal.from_json(
        {"q": {"1": {"pool": ["a"], "relevant": [], "migrated_from": {
            "pool": [], "relevant": [], "index_version": None}, "index_version": "v"}}}
    )  # fmt: skip
    first = journal.first("q")
    assert first is not None
    first.corrections = [Correction.model_validate(correction([]))]
    first.venue_coverage = {"bybit": "covered"}
    assert list(journal.model_dump()["q"]["1"]) == [
        "pool", "relevant", "migrated_from", "index_version", "corrections", "venue_coverage",
    ]  # fmt: skip
    # A field always written keeps its null; an optional one set to None is left out.
    assert journal.model_dump()["q"]["1"]["migrated_from"]["index_version"] is None
    first.venue_coverage = None
    assert "venue_coverage" not in journal.model_dump()["q"]["1"]


def test_a_judgment_covers_its_pool_then_the_augmentations_and_knows_its_first_pass():
    entry = Judgment.model_validate(
        {
            "pool": ["a", "b", "c"],
            "relevant": ["b", "x"],
            "corrections": [correction(["a"]), correction(["b"])],
            "augmentations": [augmentation(["x", "a"], ["x"])],
        }
    )
    assert entry.judged_ids() == ["a", "b", "c", "x"]
    assert entry.first_pass_relevant() == {"a", "x"}
    assert Judgment(relevant=["d"]).first_pass_relevant() == {"d"}


def test_gold_resolves_from_round_1_the_adjudication_and_a_relabel():
    first = {"pool": ["a", "b", "c"], "relevant": ["a", "b"]}
    journal = Journal.from_json(
        {
            "q1": {"1": first},
            "q2": {"1": first, "adjudicated": adjudication({"b": False, "c": True, "a": True})},
            "q3": {"1": first, "relabeled": relabel(["a", "x"], ["x"])},
        }
    )
    assert journal.get("q1").resolved_relevant() == ["a", "b"]
    assert journal.get("q2").resolved_relevant() == ["a", "c"]
    q2 = journal.first("q2")
    assert q2 is not None and q2.relevant == ["a", "b"]  # round 1 is read, never changed
    assert journal.get("q3").resolved_relevant() == ["x"]
    assert journal.get("absent").resolved_relevant() == []


def test_entries_list_every_key_with_its_index_version():
    journal = Journal.from_json(
        {"q": {"1": {"index_version": "v1"}, "2": {}, "relabeled": relabel([], [])}}
    )
    assert journal.get("q").entries() == [("1", "v1"), ("2", None), ("relabeled", "v")]


def test_the_judged_sets_include_what_a_relabel_showed(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    assert load_judged_sets(path) is None
    save_rounds(
        path,
        Journal.from_json(
            {
                "q1": {"1": {"pool": ["a"], "relevant": []}, "relabeled": relabel(["a", "x"], [])},
                "q2": {"1": {"pool": ["b"], "relevant": ["b"]}},
                "q3": {"2": {"pool": ["c"], "relevant": []}},
            }
        ),
    )
    assert load_judged_sets(path) == {"q1": {"a", "x"}, "q2": {"b"}}


def test_a_missing_journal_reads_as_empty_and_a_damaged_one_is_a_user_error(tmp_path: Path):
    path = tmp_path / "q.jsonl"
    assert load_rounds(path).questions == {}
    sidecar = rounds_path(path)
    sidecar.parent.mkdir()
    for text, problem in (
        ("{not json", "Expecting property name"),
        ("[1, 2]", "holds a JSON list where an object of rounds by question id is expected"),
        ('{"q1": {"1": {"pool": "a"}}}', "q1.rounds.1.pool: Input should be a valid list"),
        ('{"q1": {"1": {"surprise": 1}}}', "q1.rounds.1.surprise: Extra inputs are not permitted"),
    ):
        sidecar.write_text(text, encoding="utf-8")
        with pytest.raises(UserError, match=problem):
            load_rounds(path)


def test_a_journal_saved_and_read_back_is_the_same_journal(tmp_path: Path):
    data = {"q1": {"1": {"pool": ["a"], "relevant": ["a"], "augmentations": [augmentation(["b"])]}}}
    save_rounds(tmp_path / "q.jsonl", Journal.from_json(data))
    assert json.loads(rounds_path(tmp_path / "q.jsonl").read_text()) == data


def _with_every_entry() -> Journal:
    return Journal.from_json(
        {
            "q": {
                "1": {"pool": ["a", "b"], "relevant": ["a"], "index_version": "v"},
                "2": {"pool": ["a", "b"], "relevant": ["b"], "shuffle_seed": 1},
                "3": {"pool": ["b", "a"], "relevant": ["b"], "criterion": "2026-10-10"},
                "adjudicated": adjudication({"a": False}),
                "relabeled": relabel(["a"], ["a"]),
            }
        }
    )


def test_round_3_replaces_every_earlier_entry_in_the_resolved_gold():
    assert _with_every_entry().get("q").resolved_relevant() == ["b"]


def test_a_journal_with_round_3_reads_and_writes_back_identical():
    journal = _with_every_entry()
    text = journal.to_text()
    assert Journal.from_json(json.loads(text)).to_text() == text
    assert list(json.loads(text)["q"]) == ["1", "2", "3", "adjudicated", "relabeled"]


def test_an_empty_flag_and_criterion_are_not_written():
    entry = json.loads(_with_every_entry().to_text())["q"]
    assert "flagged" not in entry["3"] and "criterion" not in entry["2"]


def test_a_failed_write_leaves_the_last_complete_journal(tmp_path, monkeypatch):
    path = tmp_path / "q.jsonl"
    save_rounds(path, _with_every_entry())
    before = rounds_path(path).read_bytes()

    def fail(self: Path, *_: object, **__: object) -> None:
        self.write_bytes(b"{ cut short")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="disk full"):
        save_rounds(path, Journal())
    assert rounds_path(path).read_bytes() == before
