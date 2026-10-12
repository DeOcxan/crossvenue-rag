"""Tests for `crossvenue_rag.labeling.augment`."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from crossvenue_rag.dataset.agreement import judgements_from_rounds
from crossvenue_rag.dataset.journal import Judgment, load_rounds
from crossvenue_rag.dataset.questions import load_questions
from crossvenue_rag.errors import UserError
from crossvenue_rag.labeling.augment import augment_session
from crossvenue_rag.labeling.candidates import load_candidates, save_candidates
from crossvenue_rag.labeling.session import LabelTarget
from tests.conftest import journal_dict
from tests.labeling.conftest import (
    VERSION,
    ByIdRetriever,
    Script,
    setup_augment,
)


def test_augmentation_extends_round_1_and_keeps_existing_labels(tmp_path):
    path, candidates = setup_augment(tmp_path)
    console = Console(record=True, width=160)
    # n2 is #1, n1 is #2. The script marks n1 (bybit) and answers y to the coverage
    # question.
    recorded = augment_session(
        ByIdRetriever(),
        LabelTarget(path, VERSION, "me"),
        candidates,
        console=console,
        prompt=Script("2", "y"),
    )
    assert recorded == 1

    entry = journal_dict(path)["cv-0001"]["1"]
    assert set(journal_dict(path)["cv-0001"]) == {"1"}
    assert entry["pool"] == ["g1", "p1"]  # untouched
    assert entry["relevant"] == ["g1", "n1"]  # appended, g1 kept
    [aug] = entry["augmentations"]
    assert aug["shown"] == ["n2", "n1"]
    assert aug["relevant"] == ["n1"]
    assert aug["index_version"] == VERSION and aug["annotator"] == "me"
    assert aug["coverage_changes"] == [
        {
            "venue": "bybit",
            "from": "missing_from_pool",
            "to": "covered",
            "previous_assessment_wrong": False,
        }
    ]
    assert Judgment.model_validate(entry).judged_ids() == ["g1", "p1", "n2", "n1"]

    [question] = load_questions(path)
    assert question.gold_chunk_ids == ["g1", "n1"]
    assert question.venue_coverage == {"binance": "covered", "bybit": "covered"}
    out = console.export_text()
    assert "current gold g1" in out and "binance text of g1." in out
    # Question first, then the gold as context, then only the augmentation chunks.
    assert out.index("how do binance and bybit") < out.index("current gold g1")
    assert out.index("current gold g1") < out.index("first line")
    assert "source chunks in pool" not in out
    assert "oracle#1" in out and "split#4" in out  # both methods, told apart

    # Everything listed is judged after one pass: a second pass shows nothing, records nothing.
    before = Path(path).read_bytes()
    assert (
        augment_session(
            ByIdRetriever(),
            LabelTarget(path, VERSION, "me"),
            candidates,
            console=Console(),
            prompt=Script(),
        )
        == 0
    )
    assert Path(path).read_bytes() == before

    # Agreement counts augmented chunks as judged: n2 irrelevant, n1 relevant.
    labels = judgements_from_rounds(load_rounds(path), 1).labels
    assert labels[("cv-0001", "n2")] is False and labels[("cv-0001", "n1")] is True


def test_flipping_absent_in_docs_is_recorded_as_a_wrong_earlier_assessment(tmp_path):
    path, candidates = setup_augment(tmp_path, bybit="absent_in_docs")
    augment_session(
        ByIdRetriever(),
        LabelTarget(path, VERSION, "me"),
        candidates,
        console=Console(),
        prompt=Script("2", "y"),
    )
    [aug] = journal_dict(path)["cv-0001"]["1"]["augmentations"]
    assert aug["coverage_changes"][0]["from"] == "absent_in_docs"
    assert aug["coverage_changes"][0]["previous_assessment_wrong"] is True


def test_declining_the_coverage_change_goes_back_to_selection(tmp_path):
    path, candidates = setup_augment(tmp_path)
    # The script marks bybit's n1, declines, then marks only binance's n2 instead.
    augment_session(
        ByIdRetriever(),
        LabelTarget(path, VERSION, "me"),
        candidates,
        console=Console(),
        prompt=Script("2", "b", "1"),
    )
    entry = journal_dict(path)["cv-0001"]["1"]
    assert entry["augmentations"][0]["relevant"] == ["n2"]
    assert entry["augmentations"][0]["coverage_changes"] == []
    assert load_questions(path)[0].venue_coverage["bybit"] == "missing_from_pool"


def test_none_relevant_records_the_shown_chunks_and_skip_records_nothing(tmp_path):
    path, candidates = setup_augment(tmp_path)
    before = Path(path).read_bytes()
    augment_session(
        ByIdRetriever(),
        LabelTarget(path, VERSION, "me"),
        candidates,
        console=Console(),
        prompt=Script(""),
    )
    assert Path(path).read_bytes() == before
    assert "augmentations" not in journal_dict(path)["cv-0001"]["1"]

    augment_session(
        ByIdRetriever(),
        LabelTarget(path, VERSION, "me"),
        candidates,
        console=Console(),
        prompt=Script("n"),
    )
    entry = journal_dict(path)["cv-0001"]["1"]
    assert entry["augmentations"][0]["shown"] == ["n2", "n1"]
    assert entry["augmentations"][0]["relevant"] == []
    assert entry["relevant"] == ["g1"]


def test_a_candidate_list_from_another_index_is_refused(tmp_path):
    path, candidates = setup_augment(tmp_path)
    entries = load_candidates(candidates)
    entries[0] = entries[0].model_copy(update={"index_version": "other"})
    save_candidates(candidates, entries)
    with pytest.raises(UserError, match="lists chunks from index"):
        augment_session(
            ByIdRetriever(),
            LabelTarget(path, VERSION, "me"),
            candidates,
            console=Console(),
            prompt=Script(),
        )
