"""Tests for `crossvenue_rag.corpus.dedup`."""

from __future__ import annotations

from crossvenue_rag.corpus.dedup import deduplicate, primary_key
from crossvenue_rag.models import ChunkSource
from tests.conftest import occurrence


def test_identical_text_within_a_venue_becomes_one_chunk_with_every_source():
    chunks, report = deduplicate(
        [
            occurrence("binance", "testnet/rest-api.md", "limit 1200", 4, env="testnet"),
            occurrence("binance", "rest-api.md", "limit 1200", 4, env="mainnet"),
            occurrence("binance", "rest-api.md", "limit 6000", 5, env="mainnet"),
        ]
    )
    assert len(chunks) == 2
    merged = chunks[0]
    # Cited as the mainnet page even though the testnet copy came first.
    assert merged.doc_path == "rest-api.md"
    assert merged.environment == "mainnet"
    assert [s.doc_path for s in merged.sources] == ["rest-api.md", "testnet/rest-api.md"]
    assert report.groups == 1
    assert report.occurrences_in_groups == 2
    assert report.testnet_mirror_groups == 1
    assert report.chunks_before - report.chunks_after == 1


def test_primary_source_is_deterministic():
    """Non-testnet first, then doc path, then position, whatever the input order."""
    sources = [
        ChunkSource("b", "z.md", [], "", 0, environment=None),
        ChunkSource("b", "a.md", [], "", 3, environment="testnet"),
        ChunkSource("b", "a.md", [], "", 1, environment=None),
        ChunkSource("b", "m.md", [], "", 0, environment="mainnet"),
    ]
    ordered = sorted(sources, key=primary_key)
    assert [(s.doc_path, s.position) for s in ordered] == [
        ("a.md", 1),
        ("m.md", 0),
        ("z.md", 0),
        ("a.md", 3),
    ]


def test_identical_text_on_two_venues_is_kept_apart_and_reported():
    chunks, report = deduplicate(
        [occurrence("binance", "a.md", "same words"), occurrence("bybit", "b.md", "same words")]
    )
    assert len(chunks) == 2
    assert report.groups == 0
    assert report.cross_venue_groups == 1
    assert report.cross_venue_examples[0].venues == ["binance", "bybit"]
