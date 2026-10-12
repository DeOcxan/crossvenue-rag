"""Tests for `crossvenue_rag.labeling.coverage`."""

from __future__ import annotations

from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.models import Question
from tests.conftest import journal_dict, write_journal
from tests.labeling.conftest import VERSION, comparison

VENUES = {"c1": "binance", "c2": "bybit", "c3": "binance"}


def test_derivation_fills_only_what_the_recorded_data_settles(tmp_path):
    from crossvenue_rag.labeling.coverage import derive_coverage

    path = tmp_path / "questions.jsonl"
    three = ["binance", "bybit", "bingx"]
    questions = [
        comparison(id="q-full", gold_chunk_ids=["c1", "c2"], annotated_at="t"),
        # bybit had a candidate in the pool (c2) and none was marked: a judgment.
        Question(
            "q-part",
            "x y z?",
            "silver",
            three,
            "comparison",
            category="comparison",
            gold_chunk_ids=["c1"],
            annotated_at="t",
        ),
        comparison(id="q-skip", gold_chunk_ids=["c1", "c2"], annotated_at="t"),
        comparison(id="q-new"),  # never labeled
    ]
    save_questions(path, questions)
    entry = {"annotated_at": "t", "index_version": VERSION, "pool": ["c1", "c2"]}
    write_journal(
        path,
        {
            "q-full": {"1": {**entry, "relevant": ["c1", "c2"]}},
            "q-part": {"1": {**entry, "relevant": ["c1"]}},
            "q-skip": {"1": {**entry, "relevant": ["c1", "c2"]}},
        },
    )
    changed = derive_coverage(path, VENUES, VERSION, exclude={"q-skip"})
    assert [(d.question_id, d.coverage, d.undecided) for d in changed] == [
        ("q-full", {"binance": "covered", "bybit": "covered"}, []),
        ("q-part", {"binance": "covered", "bingx": "missing_from_pool"}, ["bybit"]),
    ]
    by_id = {q.id: q for q in load_questions(path)}
    assert by_id["q-skip"].venue_coverage == {}
    assert by_id["q-new"].venue_coverage == {}
    [correction] = journal_dict(path)["q-part"]["1"]["corrections"]
    assert correction["derived"] is True
    assert "annotator" not in correction
    assert correction["previous_venue_coverage"] == {}
    assert correction["relevant"] == correction["previous_relevant"] == ["c1"]
    # Idempotent: a second run changes nothing and journals nothing.
    assert derive_coverage(path, VENUES, VERSION, exclude={"q-skip"}) == []
    assert len(journal_dict(path)["q-part"]["1"]["corrections"]) == 1
