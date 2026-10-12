"""Tests for `crossvenue_rag.dataset.questions`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.dataset.journal import Journal
from crossvenue_rag.dataset.questions import label_version_problems, save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.models import Question
from tests.conftest import write_journal


def test_labels_from_another_index_are_reported(tmp_path: Path):
    labelled = Question("q1", "a?", "gold", ["bybit"], "fact", gold_chunk_ids=["c1"])
    unlabelled = Question("q2", "b?", "gold", ["bybit"], "fact")
    rounds = Journal.from_json(
        {"q1": {"1": {"index_version": "old", "pool": ["c1"], "relevant": ["c1"]}}}
    )
    assert label_version_problems([labelled, unlabelled], rounds, "new") == [
        "q1 round 1: labeled against old"
    ]
    assert label_version_problems([labelled], rounds, "old") == []
    assert label_version_problems([labelled], Journal(), "new") == [
        "q1 round 1: no index_version recorded"
    ]


def test_a_frozen_dataset_is_checked_against_its_meta(tmp_path: Path):
    from crossvenue_rag.dataset.questions import require_label_versions

    path = tmp_path / "v9-gold.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact", gold_chunk_ids=["c1"])])
    meta = tmp_path / "v9-gold.meta.json"
    snapshot = {
        "manifest_path": "data/manifest.csv",
        "manifest_sha256": "0" * 64,
        "documents": 1,
        "recorded_at": "2026-01-01T00:00:00Z",
        "index_version": "v2",
    }
    fields = {"dataset_version": "v9-gold", "created_at": "2026-01-01T00:00:00Z"}
    meta.write_text(
        json.dumps(
            {**fields, "corpus_snapshot": snapshot, "sources": [], "generator": {}, "counts": {}}
        )
    )
    require_label_versions(path, "v2")  # no sidecar needed
    with pytest.raises(UserError, match="frozen against index v2"):
        require_label_versions(path, "v3")


def test_a_gold_id_missing_from_the_corpus_stops_the_evaluation():
    from crossvenue_rag.dataset.questions import require_gold_resolves

    questions = [
        Question("cv-1", "q", "gold", ["binance"], "fact", gold_chunk_ids=["a", "b"]),
        Question("cv-2", "q", "gold", ["bybit"], "fact"),
    ]
    require_gold_resolves(questions, {"a", "b"})
    with pytest.raises(UserError, match=r"1 gold chunk id.*cv-1:b"):
        require_gold_resolves(questions, {"a"})


def test_a_question_line_missing_a_field_is_a_user_error_naming_the_line(tmp_path: Path):
    from crossvenue_rag.dataset.questions import load_questions

    path = tmp_path / "questions.jsonl"
    save_questions(path, [Question("cv-0001", "q one", "gold", ["bybit"], "fact")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "cv-0002", "question": "q two", "venues": ["bybit"]}) + "\n")
    with pytest.raises(UserError, match=f"{path}:2: the record has no tier and qtype") as refused:
        load_questions(path)
    assert refused.value.hint == "fix that line of the file"


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"{not json", "invalid"), (b'{"dataset_version": 1}', "invalid"), (b"\xff\xfe", "UTF-8")],
)
def test_a_damaged_dataset_meta_is_a_user_error_naming_it(tmp_path: Path, content, message):
    from crossvenue_rag.dataset.questions import require_label_versions

    path = tmp_path / "v9-gold.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact", gold_chunk_ids=["c1"])])
    (tmp_path / "v9-gold.meta.json").write_bytes(content)
    with pytest.raises(
        UserError, match=rf"{message}.*v9-gold\.meta\.json|v9-gold\.meta\.json.*{message}"
    ):
        require_label_versions(path, "v2")


def test_the_index_check_refuses_labels_of_another_index_and_gold_the_index_lacks(
    tmp_path: Path,
):
    from crossvenue_rag.dataset.questions import index_check

    path = tmp_path / "q.jsonl"
    question = Question("cv-1", "q", "gold", ["binance"], "fact", gold_chunk_ids=["a"])
    save_questions(path, [question])
    write_journal(path, {"cv-1": {"1": {"index_version": "v1", "pool": ["a"], "relevant": ["a"]}}})
    index_check(path, [question])("v1", {"a"})
    # Without questions only the labels are checked.
    index_check(path)("v1", set())
    with pytest.raises(UserError, match="not made against the current index v2"):
        index_check(path)("v2", {"a"})
    with pytest.raises(UserError, match="1 gold chunk id not in the current corpus"):
        index_check(path, [question])("v1", set())


def test_v0_2_2_gold_is_v0_2_1_with_resolved_gold_and_the_d23_review():
    """The committed v0.2.2-gold is v0.2.1-gold with each gold list resolved from its journal.

    Beyond gold, only the three questions of deviation D23 differ: two recategorized to
    fact, and the notes of all three.
    """
    from dataclasses import fields

    from crossvenue_rag.dataset.journal import load_rounds
    from crossvenue_rag.dataset.questions import load_questions
    from tests.published import ROOT

    base = load_questions(ROOT / "evals/datasets/v0.2.1-gold.jsonl")
    built = load_questions(ROOT / "evals/datasets/v0.2.2-gold.jsonl")
    rounds = load_rounds(ROOT / "evals/datasets/v0.2.2-gold.jsonl")
    assert [q.id for q in built] == [q.id for q in base]
    recategorized = {"cv-0115", "cv-0121"}
    for old, new in zip(base, built, strict=True):
        entries = rounds.get(new.id)
        resolved = entries.resolved_relevant() if entries.first else old.gold_chunk_ids
        assert new.gold_chunk_ids == resolved
        differ = {
            f.name
            for f in fields(old)
            if f.name != "gold_chunk_ids" and getattr(new, f.name) != getattr(old, f.name)
        }
        if new.id in recategorized:
            assert differ == {"category", "is_answerable", "expected_behavior", "notes"}
            assert (new.category, new.is_answerable, new.expected_behavior) == (
                "fact",
                True,
                "answer",
            )
        elif new.id == "cv-0106":
            assert differ == {"notes"} and new.category == "unanswerable"
        else:
            assert differ == set(), new.id
