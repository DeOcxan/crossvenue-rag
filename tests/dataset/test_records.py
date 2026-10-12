"""Tests for `crossvenue_rag.dataset.records`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossvenue_rag.dataset.records import DatasetMeta, as_question
from tests.dataset.factories import record


def test_answerability_fields_must_agree():
    """A refusal benchmark whose unanswerable rows carry gold chunks measures nothing."""
    with pytest.raises(ValidationError, match="is_answerable"):
        record(category="unanswerable", is_answerable=True, expected_behavior="refuse")
    with pytest.raises(ValidationError, match="expected_behavior=refuse"):
        record(category="unanswerable", is_answerable=False, expected_behavior="answer")
    with pytest.raises(ValidationError, match="carries gold chunks"):
        record(
            category="unanswerable",
            is_answerable=False,
            expected_behavior="refuse",
            gold_chunk_ids=["c1"],
        )
    # The consistent form is accepted.
    assert (
        record(
            category="unanswerable", is_answerable=False, expected_behavior="refuse"
        ).is_answerable
        is False
    )


def test_a_record_breaking_its_categorys_rule_is_rejected():
    with pytest.raises(ValidationError, match="at least 2 venues"):
        record(category="comparison", qtype="comparison", venues=["bybit"])
    with pytest.raises(ValidationError, match="adversarial_type"):
        record(category="adversarial")
    with pytest.raises(ValidationError, match="ask_clarification"):
        record(category="ambiguous", expected_behavior="answer")


def test_the_newest_pool_is_found_by_version_not_by_string(tmp_path: Path):
    """`v0.1.10` sorts before `v0.1.2` as a string, which would silently point
    selection at a superseded pool."""
    from crossvenue_rag.dataset.records import newest_candidate_pool

    for name in ("v0.1-candidates", "v0.1.2-candidates", "v0.1.10-candidates"):
        (tmp_path / f"{name}.jsonl").touch()
    assert newest_candidate_pool(tmp_path).name == "v0.1.10-candidates.jsonl"


def _meta(**patch) -> dict:
    return {
        "dataset_version": "v9-gold",
        "created_at": "2026-01-01T00:00:00Z",
        "corpus_snapshot": {
            "manifest_path": "data/manifest.csv",
            "manifest_sha256": "0" * 64,
            "documents": 2,
            "recorded_at": "2026-01-01T00:00:00Z",
        },
        "sources": [{"venue": "binance", "venue_type": "cex"}],
        "generator": {"model": "m", "prompt_hash": "abc"},
        "counts": {"total": 2, "by_venue": {"binance": 2}},
        **patch,
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"sources": [{"license": "MIT"}]},
        {"generator": {"model": "m", "temperature": 0.7}},
        {"counts": {"total": "two"}},
    ],
)
def test_a_meta_whose_sources_generator_or_counts_break_their_shape_is_rejected(patch):
    with pytest.raises(ValidationError):
        DatasetMeta(**_meta(**patch))


def test_a_meta_keeps_the_counts_of_a_labeling_pass_it_does_not_check():
    meta = DatasetMeta(**_meta(counts={"total": 2, "judged": 2}))
    assert meta.counts.total == 2 and meta.counts.by_category is None
    assert meta.counts.model_extra == {"judged": 2}


def test_as_question_carries_every_shared_field():
    source = record(
        id="q7",
        source_chunk_ids=["c1"],
        reference_answer="ref",
        notes="kept",
        annotator="me",
        annotation_round=1,
        generator="gen",
    )
    question = as_question(source)
    dumped = source.model_dump()
    assert all(getattr(question, name) == dumped[name] for name in question.__slots__)
    assert (question.id, question.source_chunk_ids, question.generator) == ("q7", ["c1"], "gen")
