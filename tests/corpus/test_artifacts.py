"""Tests for `crossvenue_rag.corpus.artifacts`."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_term_counting_is_chunks_case_insensitive(cfg):
    """One convention, because two of them produced 157 and 176 for the same term
    in the same corpus, and those numbers are quoted in notes as evidence."""
    from crossvenue_rag.corpus.artifacts import count_term, save_chunks
    from tests.conftest import make_chunk

    save_chunks(
        cfg,
        [
            make_chunk("a", "binance", "recvWindow cannot exceed 60000. recvWindow again."),
            make_chunk("b", "binance", "The RECVWINDOW parameter is optional."),
            make_chunk("c", "bybit", "No such parameter here.", venue_type="cex"),
        ],
    )

    counts = count_term(cfg, "recvWindow", None)
    # Chunks, not occurrences: chunk "a" says it twice and counts once.
    assert counts.by_venue["binance"] == 2
    assert counts.total == 2
    assert "bybit" not in counts.by_venue

    # Case-insensitive, so the shouted spelling in chunk "b" is the same term.
    assert count_term(cfg, "RECVWINDOW", None).total == 2
    assert count_term(cfg, "recvwindow", None).total == 2

    # A venue filter narrows the corpus before counting.
    assert count_term(cfg, "recvWindow", venue="bybit").total == 0
    assert count_term(cfg, "parameter", venue="bybit").total == 1


LOCAL_STATS = Path(__file__).resolve().parents[2] / "data" / "processed" / "corpus_stats.json"


@pytest.mark.skipif(not LOCAL_STATS.exists(), reason="no local ingest")
def test_the_local_corpus_statistics_are_written_back_with_the_same_bytes():
    """data/processed/ is not committed, so this runs where an ingest has."""
    from crossvenue_rag.corpus.artifacts import CorpusStats, corpus_stats_json

    text = LOCAL_STATS.read_text(encoding="utf-8")
    assert corpus_stats_json(CorpusStats.model_validate_json(text)) == text


def test_venue_chunk_counts_count_each_venues_chunks(cfg):
    from collections import Counter

    from crossvenue_rag.corpus.artifacts import save_chunks, venue_chunk_counts
    from tests.conftest import build_corpus

    chunks = build_corpus(cfg)
    save_chunks(cfg, chunks)
    assert venue_chunk_counts(cfg) == dict(Counter(c.venue for c in chunks))


def test_a_chunk_record_without_an_id_is_a_user_error_naming_its_line_and_the_fix(tmp_path):
    from crossvenue_rag.corpus.artifacts import read_chunk_field
    from crossvenue_rag.errors import UserError

    path = tmp_path / "chunks.jsonl"
    path.write_text('{"chunk_id": "a", "venue": "x"}\n{"venue": "y"}\n', encoding="utf-8")
    with pytest.raises(
        UserError, match=r"chunks\.jsonl:2: the record lacks a string chunk_id"
    ) as refused:
        read_chunk_field(path, "venue")
    assert "crossvenue ingest" in refused.value.hint


@pytest.mark.parametrize(
    ("record", "lacking"),
    [
        ({"chunk_id": ["a"], "venue": "x"}, "chunk_id"),
        ({"chunk_id": 7, "venue": "x"}, "chunk_id"),
        ({"chunk_id": "a", "venue": None}, "venue"),
        ({"chunk_id": "a", "venue": 3}, "venue"),
    ],
)
def test_a_chunk_record_with_a_field_that_is_no_string_is_a_user_error_naming_its_line(
    tmp_path, record, lacking
):
    import json

    from crossvenue_rag.corpus.artifacts import read_chunk_field
    from crossvenue_rag.errors import UserError

    path = tmp_path / "chunks.jsonl"
    good = {"chunk_id": "b", "venue": "y"}
    path.write_text(f"{json.dumps(good)}\n{json.dumps(record)}\n", encoding="utf-8")
    with pytest.raises(
        UserError, match=rf"chunks\.jsonl:2: the record lacks a string {lacking}$"
    ) as refused:
        read_chunk_field(path, "venue")
    assert refused.value.hint == "run `crossvenue ingest` again"


def test_a_missing_chunk_file_is_a_user_error_naming_ingest_for_every_field_reader(tmp_path):
    from crossvenue_rag.corpus.artifacts import read_chunk_field, read_chunk_venues
    from crossvenue_rag.errors import UserError

    missing = tmp_path / "chunks.jsonl"
    for read in (lambda: read_chunk_field(missing, "text"), lambda: read_chunk_venues(missing)):
        with pytest.raises(UserError, match=r"chunks\.jsonl not found") as refused:
            read()
        assert refused.value.hint == "run `crossvenue ingest` first"


def test_a_chunk_record_that_is_not_a_chunk_is_a_user_error_naming_its_line_and_the_fix(tmp_path):
    import json

    from crossvenue_rag.corpus.artifacts import read_chunks
    from crossvenue_rag.errors import UserError
    from tests.conftest import make_chunk

    good = make_chunk("a", "binance", "Some text.").to_dict()
    no_id = {k: v for k, v in good.items() if k != "chunk_id"}
    extra = {**good, "colour": "red"}
    bad_source = {**good, "sources": [{"venue": "binance"}]}
    cases = [
        (no_id, r"chunks\.jsonl:2: the record has no chunk_id"),
        (extra, r"chunks\.jsonl:2: the record has 1 unknown field colour"),
        (bad_source, r"chunks\.jsonl:2: the record has a source that has no doc_path"),
    ]
    path = tmp_path / "chunks.jsonl"
    for record, message in cases:
        lines = [json.dumps(good), json.dumps(record)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(UserError, match=message) as refused:
            read_chunks(path)
        assert "crossvenue ingest" in refused.value.hint


SOURCE = {"venue": "binance", "doc_path": "d.md", "heading_path": [], "source_url": "u"}


@pytest.mark.parametrize(
    ("field", "value", "refusal"),
    [
        ("text", None, "the record has a wrong text: "),
        ("chunk_id", 7, "the record has a wrong chunk_id: "),
        ("token_count", "12", "the record has a wrong token_count: "),
        (
            "sources",
            [{**SOURCE, "position": "1"}],
            "the record has a source that has a wrong position: ",
        ),
    ],
)
def test_a_chunk_record_with_a_wrongly_typed_field_is_a_user_error_naming_it(
    tmp_path, field, value, refusal
):
    import json

    from crossvenue_rag.corpus.artifacts import read_chunks
    from crossvenue_rag.errors import UserError
    from tests.conftest import make_chunk

    record = {**make_chunk("a", "binance", "Some text.").to_dict(), field: value}
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(UserError, match=r"chunks\.jsonl:1: " + refusal) as refused:
        read_chunks(path)
    assert "crossvenue ingest" in refused.value.hint


def test_chunks_are_keyed_by_their_id_in_corpus_order(cfg):
    from crossvenue_rag.corpus.artifacts import chunks_by_id
    from tests.conftest import build_corpus

    chunks = build_corpus(cfg)
    by_id = chunks_by_id(chunks)
    assert list(by_id) == [c.chunk_id for c in chunks]
    assert all(by_id[c.chunk_id] is c for c in chunks)
