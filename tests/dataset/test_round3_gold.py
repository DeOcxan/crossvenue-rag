"""Tests for `crossvenue_rag.dataset.round3_gold`: the gold set built from round 3 alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crossvenue_rag.dataset.records import load_records
from crossvenue_rag.dataset.round3_gold import build_round3_gold
from crossvenue_rag.errors import UserError
from tests.conftest import write_journal

COMMON = {"annotator": "me", "annotated_at": "2026-10-31T12:00:00Z", "index_version": "v"}


def _row(qid: str, **fields: object) -> dict[str, object]:
    return {
        "id": qid,
        "question": "which endpoint sets the leverage on bybit",
        "tier": "gold",
        "venues": ["bybit"],
        "qtype": "fact",
        "category": "fact",
        "difficulty": "easy",
        "is_answerable": True,
        "expected_behavior": "answer",
        "gold_chunk_ids": ["old"],
        "annotation_round": 1,
        "notes": "kept",
    } | fields


UNANSWERABLE = {
    "category": "unanswerable",
    "is_answerable": False,
    "expected_behavior": "refuse",
    "gold_chunk_ids": [],
}


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _earlier() -> dict[str, dict[str, Any]]:
    first = {"pool": ["old", "new"], "relevant": ["old"], "index_version": "v"}
    return {"q1": {"1": dict(first)}, "q2": {"1": dict(first, relevant=[])}}


def _setup(tmp_path: Path, third: dict[str, dict[str, Any]]) -> tuple[Path, str, Path]:
    """A base version with its frozen journal, and a working journal with `third` added."""
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    base = datasets / "base.jsonl"
    _write(base, [_row("q1"), _row("q2", **UNANSWERABLE)])
    write_journal(base, _earlier())
    working = tmp_path / "questions.jsonl"
    journal = _earlier()
    for qid, entry in third.items():
        journal[qid]["3"] = dict(COMMON, pool=["old", "new"]) | entry
    write_journal(working, journal)
    return base, str(working), datasets / "out.jsonl"


FULL: dict[str, dict[str, Any]] = {
    "q1": {"relevant": ["new"], "venue_coverage": {"bybit": "covered"}},
    "q2": {"relevant": [], "flagged": True},
}


def test_only_the_label_fields_come_from_round_3(tmp_path):
    base, working, out = _setup(tmp_path, FULL)
    built = build_round3_gold(base, working, out)
    assert built.written == 2
    assert built.flagged == ['q2 "which endpoint sets the leverage on"']
    before, after = load_records(base), load_records(out)
    assert after[0].gold_chunk_ids == ["new"] and after[1].gold_chunk_ids == []
    assert after[0].venue_coverage == {"bybit": "covered"}
    assert after[0].annotation_round == 3 and after[0].annotated_at == COMMON["annotated_at"]
    assert (after[1].category, after[1].notes) == (before[1].category, before[1].notes)
    label_fields = {
        "gold_chunk_ids", "venue_coverage", "annotator", "annotated_at", "annotation_round",
    }  # fmt: skip
    assert before[0].model_dump(exclude=label_fields) == after[0].model_dump(exclude=label_fields)


def test_an_unfinished_round_is_refused(tmp_path):
    base, working, out = _setup(tmp_path, {"q1": FULL["q1"]})
    with pytest.raises(UserError, match="without a round-3 entry: q2"):
        build_round3_gold(base, working, out)
    assert not out.exists()


def test_a_changed_earlier_entry_is_refused(tmp_path):
    base, working, out = _setup(tmp_path, FULL)
    journal = _earlier()
    journal["q1"]["1"]["relevant"] = ["new"]
    for qid, entry in FULL.items():
        journal[qid]["3"] = dict(COMMON, pool=["old", "new"]) | entry
    write_journal(working, journal)
    with pytest.raises(UserError, match="differ from the frozen journal: q1"):
        build_round3_gold(base, working, out)
    assert not out.exists()


def test_an_existing_version_is_never_rewritten(tmp_path):
    base, working, out = _setup(tmp_path, FULL)
    out.write_text("frozen\n", encoding="utf-8")
    with pytest.raises(UserError, match="exists"):
        build_round3_gold(base, working, out)
    assert out.read_text(encoding="utf-8") == "frozen\n"
