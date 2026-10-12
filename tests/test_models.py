"""Tests for `crossvenue_rag.models`."""

from __future__ import annotations

import json

from crossvenue_rag.corpus.dedup import deduplicate
from crossvenue_rag.dataset.questions import load_questions, save_questions
from crossvenue_rag.models import Chunk, ChunkSource, Question
from tests.conftest import occurrence


def test_chunk_round_trips_through_json_with_its_sources():
    [chunk], _ = deduplicate([occurrence("binance", "a.md", "x", env="mainnet")])
    again = Chunk.from_dict(chunk.to_dict())
    assert again == chunk
    assert isinstance(again.sources[0], ChunkSource)


def test_questions_without_coverage_keep_their_bytes(tmp_path):
    path = tmp_path / "q.jsonl"
    save_questions(path, [Question("q1", "a question?", "gold", ["bybit"], "fact")])
    assert "venue_coverage" not in json.loads(path.read_text())
    again = tmp_path / "again.jsonl"
    save_questions(again, load_questions(path))
    assert again.read_bytes() == path.read_bytes()


def test_a_question_saved_without_a_category_falls_back_to_its_qtype_everywhere():
    bare = Question("cv-1", "x", "gold", ["binance", "bybit"], "comparison")
    labeled = Question("cv-2", "x", "gold", ["binance"], "fact", category="adversarial")
    assert bare.effective_category == "comparison" and bare.is_comparison
    assert bare.slice_labels("category") == ["comparison"]
    assert labeled.effective_category == "adversarial" and not labeled.is_comparison


def test_a_comparison_qtype_under_another_category_needs_coverage_but_is_no_comparison():
    q = Question("cv-3", "x", "gold", ["binance", "bybit"], "comparison", category="multihop")
    assert q.needs_venue_coverage and not q.is_comparison


def test_a_question_is_judged_by_its_gold_or_by_an_explicit_empty_verdict():
    assert Question("cv-4", "x", "gold", ["binance"], "fact", ["c1"]).is_judged
    assert Question("cv-5", "x", "gold", ["binance"], "fact", annotated_at="t").is_judged
    assert not Question("cv-6", "x", "gold", ["binance"], "fact").is_judged


def test_the_retrieval_steps_are_the_timed_fields_in_pipeline_order():
    from crossvenue_rag.models import RETRIEVAL_STEPS, Timings

    assert RETRIEVAL_STEPS == ("encode", "search", "fuse", "rerank")
    timings = Timings(encode_ms=1.0, search_ms=2.0, fuse_ms=3.0, rerank_ms=4.0)
    assert timings.as_dict() == {
        "encode_ms": 1.0,
        "search_ms": 2.0,
        "fuse_ms": 3.0,
        "rerank_ms": 4.0,
        "total_ms": 10.0,
    }


def test_the_required_fields_of_a_record_are_those_without_a_default():
    from crossvenue_rag.models import fields_without_default, missing_fields

    assert fields_without_default(ChunkSource) == [
        "venue", "doc_path", "heading_path", "source_url", "position"
    ]  # fmt: skip
    assert fields_without_default(Question) == ["id", "question", "tier", "venues", "qtype"]
    record = {"venue": "binance", "doc_path": "a.md", "source_url": "u", "extra": 1}
    assert missing_fields(record, ChunkSource) == ["heading_path", "position"]


def test_a_wrongly_typed_field_is_named_with_why_and_nothing_is_converted():
    from crossvenue_rag.models import wrongly_typed_fields

    record = {"venue": "binance", "doc_path": 3, "heading_path": "API", "position": "1", "x": 1}
    assert list(wrongly_typed_fields(record, ChunkSource)) == [
        "doc_path", "heading_path", "position"
    ]  # fmt: skip
    assert wrongly_typed_fields({"venue": "binance", "position": 1}, ChunkSource) == {}
    assert "valid string" in wrongly_typed_fields({"venue": None}, ChunkSource)["venue"]
