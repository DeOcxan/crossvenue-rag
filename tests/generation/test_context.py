"""Tests for `generation.context`: the chunk block and citation resolution."""

from __future__ import annotations

from crossvenue_rag.generation.context import QuestionContext, render_chunks, resolve_citation
from crossvenue_rag.models import ChunkSource, Result
from tests.generation.offline import (
    chunk,
    mainnet_testnet_chunk,
)


def test_question_context_reads_environment_product_and_version():
    ctx = QuestionContext.of("on binance spot testnet how long is an IP ban with the v3 api")
    assert (ctx.environment, ctx.product, ctx.api_version) == ("testnet", "spot", "v3")
    assert QuestionContext.of("how long is an ip ban") == QuestionContext(None, None, None)


def test_citation_resolves_to_the_occurrence_matching_the_question():
    result = Result.from_chunk(mainnet_testnet_chunk(), 1.0)
    testnet = resolve_citation("on binance testnet how long is a ban", 1, result)
    assert testnet.doc_path == "testnet/rest-api.md" and not testnet.is_primary
    assert testnet.matched == ["environment"]


def test_citation_falls_back_to_the_primary_occurrence():
    result = Result.from_chunk(mainnet_testnet_chunk(), 1.0)
    neutral = resolve_citation("how long is a binance ip ban", 1, result)
    assert neutral.doc_path == "rest-api.md" and neutral.is_primary and neutral.matched == []
    # A question about an environment no occurrence has also gets the primary.
    perp = resolve_citation("binance futures ban", 1, result)
    assert perp.is_primary


def test_citation_prefers_more_matching_fields():
    sources = [
        ChunkSource("bybit", "a.md", [], "", 0, environment="mainnet", product="spot"),
        ChunkSource("bybit", "b.md", [], "", 0, environment="testnet", product="spot"),
        ChunkSource("bybit", "c.md", [], "", 0, environment="testnet", product="perp"),
    ]
    result = Result.from_chunk(chunk("c", "bybit", "text", sources), 1.0)
    best = resolve_citation("bybit perp testnet order limits", 2, result)
    assert best.doc_path == "c.md" and sorted(best.matched) == ["environment", "product"]


def test_a_tie_between_two_occurrences_that_beat_the_primary_keeps_the_earlier_one():
    sources = [
        ChunkSource("bybit", "a.md", [], "", 0, environment="mainnet"),
        ChunkSource("bybit", "b.md", [], "", 0, environment="testnet"),
        ChunkSource("bybit", "c.md", [], "", 0, environment="testnet"),
    ]
    result = Result.from_chunk(chunk("c", "bybit", "text", sources), 1.0)
    tied = resolve_citation("bybit testnet order limits", 1, result)
    assert tied.doc_path == "b.md" and not tied.is_primary


def test_chunk_block_numbers_chunks_and_shows_their_metadata():
    results = [Result.from_chunk(mainnet_testnet_chunk(), 1.0)]
    block = render_chunks(results)
    assert block.startswith("[1] venue: binance | environment: mainnet, testnet")
    assert "IP bans scale" in block
