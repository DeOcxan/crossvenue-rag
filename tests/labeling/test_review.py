"""Tests for `crossvenue_rag.labeling.review`."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.review import review_session
from crossvenue_rag.labeling.session import LabelTarget
from tests.conftest import journal_dict, write_journal
from tests.labeling.conftest import VERSION, Script, ThreeChunkRetriever, comparison, quiet


def setup_labelled(tmp_path: Path) -> str:
    path = tmp_path / "questions.jsonl"
    question = comparison(
        gold_chunk_ids=["c1"],
        venue_coverage={"binance": "covered", "bybit": "absent_in_docs"},
        annotator="me",
        annotated_at="2026-09-24T21:00:00Z",
        annotation_round=1,
    )
    save_questions(path, [question])
    write_journal(
        path,
        {
            "cv-0001": {
                "1": {
                    "annotator": "me",
                    "annotated_at": "2026-09-24T21:00:00Z",
                    "index_version": VERSION,
                    "pool": ["c1", "c2", "c3"],
                    "relevant": ["c1"],
                    "venue_coverage": {"binance": "covered", "bybit": "absent_in_docs"},
                }
            }
        },
    )
    return str(path)


def test_review_records_a_round_1_correction_and_keeps_the_previous_set(cfg, tmp_path):
    path = setup_labelled(tmp_path)
    console = quiet()
    changed = review_session(
        cfg,
        ThreeChunkRetriever(cfg),
        LabelTarget(path, VERSION, "me"),
        ["cv-0001"],
        console=console,
        prompt=Script("1 2"),
    )
    assert changed == 1

    rounds = journal_dict(path)
    assert set(rounds["cv-0001"]) == {"1"}  # a correction adds no round 2
    entry = rounds["cv-0001"]["1"]
    assert entry["relevant"] == ["c1", "c2"]
    assert entry["annotated_at"] == "2026-09-24T21:00:00Z"  # original time kept
    [correction] = entry["corrections"]
    assert correction["previous_relevant"] == ["c1"]
    assert correction["previous_venue_coverage"] == {
        "binance": "covered",
        "bybit": "absent_in_docs",
    }
    assert correction["relevant"] == ["c1", "c2"]
    assert correction["index_version"] == VERSION
    assert correction["at"].endswith("Z")

    [question] = load_questions(path)
    assert question.gold_chunk_ids == ["c1", "c2"]
    assert question.venue_coverage == {"binance": "covered", "bybit": "covered"}
    # The screen showed the current marks.
    assert "*1" in console.export_text()


def test_review_shows_the_recorded_pool_and_writes_nothing_when_unchanged(cfg, tmp_path):
    path = setup_labelled(tmp_path)
    before = (Path(path).read_bytes(), journal_dict(path))
    changed = review_session(
        cfg,
        ThreeChunkRetriever(cfg),
        LabelTarget(path, VERSION, "me"),
        ["cv-0001"],
        console=quiet(),
        prompt=Script("1", "a"),
    )
    assert changed == 0
    assert (Path(path).read_bytes(), journal_dict(path)) == before


def test_review_refuses_unknown_or_unlabelled_ids(cfg, tmp_path):
    path = setup_labelled(tmp_path)
    with pytest.raises(UserError, match="not in"):
        review_session(
            cfg,
            ThreeChunkRetriever(cfg),
            LabelTarget(path, VERSION, "me"),
            ["cv-9999"],
            console=quiet(),
        )
    questions = [*load_questions(path), comparison(id="cv-0002")]
    save_questions(path, questions)
    with pytest.raises(UserError, match="no round-1 judgment"):
        review_session(
            cfg,
            ThreeChunkRetriever(cfg),
            LabelTarget(path, VERSION, "me"),
            ["cv-0002"],
            console=quiet(),
        )
