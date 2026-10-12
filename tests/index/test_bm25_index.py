"""Tests for `crossvenue_rag.index.bm25_index`."""

from __future__ import annotations

import json

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.index.bm25_index import IDS_FILENAME, BM25Index
from tests.conftest import occurrence


def test_bm25_venue_filter_excludes_other_venues(bm25: BM25Index):
    assert {
        cid for cid, _ in bm25.search("rate limits", k=5, venues=["bybit"], allowed_ids=None)
    } == {"c-bybit"}
    assert {
        cid
        for cid, _ in bm25.search(
            "fee tiers", k=5, venues=["uniswap", "pancakeswap"], allowed_ids=None
        )
    } == {
        "c-uniswap",
        "c-pancakeswap",
    }


def test_bm25_index_round_trips_through_disk(tmp_path, bm25: BM25Index):
    bm25.save(tmp_path / "bm25")
    loaded = BM25Index.load(tmp_path / "bm25")
    assert loaded.chunk_ids == bm25.chunk_ids
    assert loaded.search("request weight", k=1, venues=None, allowed_ids=None)[0][0] == "c-binance"


def test_an_ids_file_without_venues_is_a_user_error_naming_the_file(tmp_path, bm25: BM25Index):
    bm25.save(tmp_path / "bm25")
    ids = tmp_path / "bm25" / IDS_FILENAME
    ids.write_text(json.dumps({"chunk_ids": bm25.chunk_ids}), encoding="utf-8")
    with pytest.raises(UserError, match="venues") as caught:
        BM25Index.load(tmp_path / "bm25")
    assert str(ids) in str(caught.value)


def test_bm25_id_filter_returns_k_hits_below_any_overfetch_depth():
    """A selective filter must not silently shrink k."""
    chunks = [occurrence("bybit", f"d{i}.md", f"rate limit rule {i}") for i in range(300)]
    # The tail chunk shares one query token, so it scores above 0 but below all 300 others.
    chunks.append(occurrence("bybit", "tail.md", "fees charged at a flat rate"))
    index = BM25Index.build(chunks)
    allowed = {chunks[-1].chunk_id}
    hits = index.search("rate limit", k=5, venues=None, allowed_ids=allowed)
    assert [chunk_id for chunk_id, _ in hits] == [chunks[-1].chunk_id]


def test_a_chunk_sharing_no_query_token_is_not_a_hit():
    chunks = [
        occurrence("bybit", "a.md", "order creation endpoint"),
        occurrence("bybit", "b.md", "websocket heartbeat interval"),
        occurrence("bybit", "c.md", "fee tiers for makers"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("order", k=2, venues=None, allowed_ids=None)
    assert [chunk_id for chunk_id, _ in hits] == [chunks[0].chunk_id]
    assert all(score > 0 for _, score in hits)
    assert index.search("order", k=2, venues=["bybit"], allowed_ids=None) == hits
