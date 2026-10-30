"""Tests for `crossvenue_rag.labeling.session`."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from crossvenue_rag.dataset.agreement import compare_rounds, judgements_from_rounds
from crossvenue_rag.dataset.journal import Judgment, load_rounds
from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.dataset.sampling import read_sample_file, stratified_sample, write_sample_file
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.pool import RETEST_SHUFFLE_SEED, retest_candidates
from crossvenue_rag.labeling.session import (
    LabelPlan,
    LabelTarget,
    QueueSpec,
    label_session,
    select_queue,
)
from crossvenue_rag.models import Question
from tests.conftest import augmentation, journal_dict, write_journal
from tests.dataset.factories import gold_like, question
from tests.labeling.conftest import VERSION, Script, ThreeChunkRetriever, comparison, quiet


def test_label_session_refuses_to_start_on_mismatched_labels(cfg, tmp_path: Path):
    path = tmp_path / "questions.jsonl"
    save_questions(path, [Question("q1", "a?", "gold", ["bybit"], "fact", gold_chunk_ids=["c1"])])
    write_journal(path, {"q1": {"1": {"index_version": "old", "pool": [], "relevant": ["c1"]}}})
    with pytest.raises(UserError, match="not made against the current index new"):
        # The version check refuses before the retriever is asked for anything.
        label_session(cfg, ThreeChunkRetriever(cfg), LabelPlan(LabelTarget(str(path), "new")))


def test_progress_reports_what_is_left_not_what_is_done():
    """The decision at the keyboard is "stop now or push on", so the estimate has
    to come from this session's pace and the remaining categories.
    """
    from crossvenue_rag.labeling.session import progress_table
    from crossvenue_rag.models import Question, QuestionCategory

    categories: list[QuestionCategory] = ["fact", "fact", "comparison", "multihop", "comparison"]
    queue = [
        Question(f"q{i}", f"question number {i}", "gold", ["bybit"], "fact", category=cat)
        for i, cat in enumerate(categories)
    ]
    console = Console(width=100, no_color=True)

    with console.capture() as capture:
        console.print(progress_table(queue, queue, handled=2, judged=1, elapsed_seconds=120.0))
    out = capture.get()
    assert "1 recorded, 1 skipped, 3 left in queue" in out
    assert "60s per question" in out  # pace counts the skip, which took reading time
    assert "3m" in out  # 3 remaining x 60s
    # Remaining shape, not the whole queue: the two handled items are excluded.
    assert "comparison=2" in out and "multihop=1" in out and "fact=" not in out

    with console.capture() as capture:
        console.print(progress_table(queue, queue, handled=0, judged=0, elapsed_seconds=0.0))
    assert "needs one question first" in capture.get()


def test_round_1_labeling_writes_venue_coverage(cfg, tmp_path):
    path = tmp_path / "questions.jsonl"
    save_questions(path, [comparison()])
    label_session(
        cfg,
        ThreeChunkRetriever(cfg),
        LabelPlan(LabelTarget(str(path), VERSION, "me")),
        console=quiet(),
        prompt=Script("2", "m"),
    )
    [question] = load_questions(path)
    assert question.gold_chunk_ids == ["c2"]
    assert question.venue_coverage == {"binance": "missing_from_pool", "bybit": "covered"}
    entry = journal_dict(path)["cv-0001"]["1"]
    assert entry["venue_coverage"] == question.venue_coverage
    assert entry["index_version"] == VERSION


def test_round_2_session_judges_round_1_material_and_kappa_covers_those_pairs(cfg, tmp_path):
    path = tmp_path / "questions.jsonl"
    question = Question(
        "cv-0001",
        "how long does binance keep my order",
        "gold",
        ["binance"],
        "fact",
        category="fact",
        gold_chunk_ids=["c1"],
        annotator="me",
        annotated_at="2026-09-24T12:00:00Z",
        annotation_round=1,
    )
    save_questions(path, [question])
    first = {
        "annotator": "me",
        "annotated_at": "2026-09-24T12:00:00Z",
        "index_version": VERSION,
        "pool": ["c1", "c3"],
        "relevant": ["c1"],
        "augmentations": [augmentation(["c2"])],
    }
    write_journal(path, {"cv-0001": {"1": first}})
    expected = retest_candidates(
        ThreeChunkRetriever(cfg).by_id,
        "cv-0001",
        Judgment.model_validate(first),
        RETEST_SHUFFLE_SEED,
    )
    pick = 1 + [r.chunk_id for r in expected].index("c1")

    console = quiet()
    label_session(
        cfg,
        ThreeChunkRetriever(cfg),
        LabelPlan(
            LabelTarget(str(path), VERSION, "me"),
            QueueSpec(annotation_round=2, sample_ids=["cv-0001"]),
        ),
        console=console,
        prompt=Script(str(pick)),
    )
    second = journal_dict(path)["cv-0001"]["2"]
    assert second["pool"] == [r.chunk_id for r in expected]
    assert sorted(second["pool"]) == sorted(Judgment.model_validate(first).judged_ids())
    assert second["shuffle_seed"] == RETEST_SHUFFLE_SEED
    assert second["relevant"] == ["c1"]
    assert load_questions(path)[0].gold_chunk_ids == ["c1"]  # round 1 untouched
    assert "found by" not in console.export_text()

    report = compare_rounds(
        judgements_from_rounds(load_rounds(path), 1), judgements_from_rounds(load_rounds(path), 2)
    )
    assert report.pairs == 3  # every candidate both rounds judged, augmentation included


def test_a_sample_file_drives_the_round_2_queue(tmp_path: Path):
    questions = gold_like()
    path = tmp_path / "round2_sample.txt"
    write_sample_file(path, stratified_sample(questions, 25, seed=20260924), seed=20260924)
    ids = read_sample_file(path)
    queue = select_queue(questions, QueueSpec(annotation_round=2, sample_ids=ids))
    assert [q.id for q in queue] == ids


def test_a_sample_file_naming_an_unjudged_question_is_refused():
    questions = [*gold_like(), question(999, "fact", judged=False)]
    with pytest.raises(UserError, match="round 1 never judged: cv-0999"):
        select_queue(questions, QueueSpec(annotation_round=2, sample_ids=["cv-0999"]))


def test_round_2_with_a_count_uses_the_stratified_sample():
    questions = gold_like()
    queue = select_queue(questions, QueueSpec(annotation_round=2, sample=25, seed=20260924))
    assert [q.id for q in queue] == [q.id for q in stratified_sample(questions, 25, 20260924)]


def test_the_progress_table_shows_the_count_a_round_passes():
    from crossvenue_rag.labeling.session import progress_table

    queue = [Question("q1", "a?", "gold", ["bybit"], "fact", category="fact")]
    console = Console(record=True, width=120)
    console.print(progress_table(queue, queue, 0, 0, 0.0, labelled=0))
    assert "0/1" in console.export_text()
