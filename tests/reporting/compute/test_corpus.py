"""Tests for `reporting.compute.corpus`, on artifacts written under tmp_path."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.models import ChunkSource
from crossvenue_rag.reporting.compute import corpus
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks, write_file


def source(venue, doc_path):
    return ChunkSource(venue, doc_path, ["Limits"], f"https://example.invalid/{doc_path}", 0)


def test_the_chunk_token_budget_is_read_from_the_default_config(tmp_path):
    write_file(tmp_path, "configs/default.yaml", "chunking:\n  max_tokens: 450\n")
    assert corpus.chunk_token_budget(Sources(tmp_path)) == 450


def test_the_index_version_is_read_from_the_index_meta(tmp_path):
    write_file(tmp_path, "data/index/index_meta.json", json.dumps({"index_version": "v1"}))
    assert corpus.index_version(Sources(tmp_path)) == "v1"
    with pytest.raises(
        UserError, match=r"no index is built: no index metadata at .*index_meta\.json"
    ) as raised:
        corpus.index_version(Sources(tmp_path / "unbuilt"))
    assert raised.value.hint == "build the index with `make ingest index`"


def test_sources_are_counted_by_kind_and_dex_venues_listed(tmp_path):
    write_file(
        tmp_path,
        "data/sources.yaml",
        "sources:\n"
        "  - {venue: binance, kind: git, venue_type: cex, license: x, repo: r}\n"
        "  - {venue: uniswap, kind: gitbook, venue_type: dex, license: x, llms_index: u}\n"
        "  - {venue: dydx, kind: gitbook, venue_type: dex, license: x, llms_index: u}\n",
    )
    src = Sources(tmp_path)
    assert corpus.sources_of_kind(src, "gitbook") == 2
    assert corpus.sources_of_kind(src, "git") == 1
    assert corpus.dex_venues(src) == ["dydx", "uniswap"]


def test_chunks_are_counted_the_count_term_way(tmp_path):
    write_chunks(
        tmp_path,
        [
            make_chunk("u1", "uniswap", "The 0.30% tier, and 0.3% again", "dex"),
            make_chunk("u2", "uniswap", "no fee here", "dex"),
            make_chunk("p1", "pancakeswap", "a 0.25% POOL", "dex"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus.chunk_count(src) == 3
    # Chunks containing the term, case-insensitive: one chunk however often it says so.
    assert corpus.term_chunks(src, "0.3%", "uniswap") == 1
    assert corpus.term_chunks(src, "0.25% pool", "pancakeswap") == 1
    assert corpus.term_chunks(src, "0.25%", "uniswap") == 0


def test_over_budget_chunks_split_into_prose_and_code(tmp_path):
    write_file(
        tmp_path,
        "configs/default.yaml",
        "chunking: {max_tokens: 3, overlap_tokens: 0, min_tokens: 0}\n",
    )
    write_chunks(
        tmp_path,
        [
            make_chunk("a", "binance", "one two three four"),
            make_chunk("b", "binance", "```http GET /api ```"),
            make_chunk("c", "binance", "one two three"),
        ],
    )
    assert corpus.over_budget_chunks(Sources(tmp_path)) == (1, 1)


def test_chunks_above_a_token_count_exclude_a_chunk_at_that_count(tmp_path):
    write_chunks(
        tmp_path,
        [
            make_chunk("a", "binance", "one two three four"),
            make_chunk("b", "bybit", "one two three"),
        ],
    )
    assert [c.chunk_id for c in corpus.chunks_above(Sources(tmp_path), 3)] == ["a"]


def test_filled_chunks_count_a_metadata_field_that_is_set(tmp_path):
    write_chunks(
        tmp_path,
        [
            replace(make_chunk("a", "bybit", "x"), doc_type="reference"),
            make_chunk("b", "bybit", "y"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus.filled_chunks(src, "doc_type") == 1
    assert corpus.filled_chunks(src, "product") == 0


def test_timing_security_compares_the_rest_and_websocket_sections(tmp_path):
    heading = ["General Info", "Timing security"]
    write_chunks(
        tmp_path,
        [
            replace(
                make_chunk("r", "binance", "abcd"), doc_path="rest-api.md", heading_path=heading
            ),
            replace(
                make_chunk("w", "binance", "abce"),
                doc_path="web-socket-api.md",
                heading_path=heading,
            ),
            replace(make_chunk("o", "binance", "abcd"), doc_path="rest-api.md"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus.timing_security_chunks(src) == 2
    assert corpus.timing_security_ratio(src) == pytest.approx(0.75)


def test_a_timing_security_section_missing_from_a_page_raises_a_value_error(tmp_path):
    heading = ["Timing security"]
    chunk = replace(
        make_chunk("r", "binance", "abcd"), doc_path="rest-api.md", heading_path=heading
    )
    write_chunks(tmp_path, [chunk])
    with pytest.raises(ValueError, match=r"chunks has no 'web-socket-api\.md'"):
        corpus.timing_security_ratio(Sources(tmp_path))


def test_duplicates_count_groups_occurrences_and_cross_venue_texts(tmp_path):
    mirrored = [source("binance", "rest-api.md"), source("binance", "testnet/rest-api.md")]
    write_chunks(
        tmp_path,
        [
            replace(make_chunk("g", "binance", "same text"), sources=mirrored),
            replace(make_chunk("s", "bybit", "shared"), sources=[source("bybit", "a.md")]),
            replace(make_chunk("t", "bingx", "shared"), sources=[source("bingx", "b.md")]),
        ],
    )
    dup = corpus.duplicates(Sources(tmp_path))
    assert dup.occurrences == 4
    assert dup.in_groups == 2
    assert dup.groups == 1
    assert dup.by_venue == {"binance": 1}
    assert dup.cross_venue_texts == 1
    assert dup.pages == {("rest-api.md", "testnet/rest-api.md"): 1}


def test_venues_are_those_with_a_chunk(tmp_path):
    write_chunks(
        tmp_path,
        [make_chunk("a", "bybit", "x"), make_chunk("b", "bybit", "y"), make_chunk("c", "okx", "z")],
    )
    assert corpus.venue_count(Sources(tmp_path)) == 2


def test_tokens_and_words_are_summed_over_the_chunks(tmp_path):
    write_chunks(
        tmp_path,
        [
            replace(make_chunk("a", "bybit", "one two three"), token_count=6),
            replace(make_chunk("b", "binance", "four"), token_count=2),
        ],
    )
    src = Sources(tmp_path)
    assert corpus.token_total(src) == 8
    assert corpus.word_total(src) == 4
    assert corpus.tokens_per_word(src) == 2.0
    assert [c.chunk_id for c in corpus.venue_chunks(src, "binance")] == ["b"]


def test_a_text_on_two_venues_is_cross_venue_and_a_term_absent_where_no_chunk_has_it(tmp_path):
    write_chunks(
        tmp_path,
        [make_chunk("a", "bybit", "BNB Chain fees"), make_chunk("b", "binance", "spot fees")],
    )
    src = Sources(tmp_path)
    assert corpus.no_cross_venue_texts(src)
    assert corpus.term_absent(src, "bnb chain", "binance")
    assert not corpus.term_absent(src, "bnb chain", "bybit")
    write_chunks(
        tmp_path,
        [make_chunk("a", "bybit", "same"), make_chunk("b", "binance", "same")],
    )
    assert not corpus.no_cross_venue_texts(Sources(tmp_path))


def test_testnet_copies_of_a_mainnet_page_are_found_byte_identical_or_normalized(tmp_path):
    main, testnet = source("binance", "rest-api.md"), source("binance", "testnet/rest-api.md")
    write_chunks(
        tmp_path,
        [
            # One chunk stored for both pages: a byte-identical copy.
            replace(make_chunk("a", "binance", "same text"), sources=[main, testnet]),
            # Identical once "testnet" is removed and whitespace collapsed.
            replace(make_chunk("b", "binance", "Base URL  is the api"), sources=[main]),
            replace(make_chunk("c", "binance", "Base URL is the Testnet api"), sources=[testnet]),
            replace(make_chunk("d", "binance", "mainnet only"), sources=[main]),
            replace(make_chunk("e", "bybit", "Base URL is the api"), sources=[main]),
        ],
    )
    copies = corpus.testnet_copies(Sources(tmp_path), "rest-api.md")
    assert copies == corpus.TestnetCopies(3, 1, 2)


def test_sources_with_a_term_are_the_venues_with_a_chunk_that_holds_it(tmp_path):
    write_chunks(
        tmp_path,
        [
            make_chunk("u1", "uniswap", "a funding note", "dex"),
            make_chunk("p1", "pancakeswap", "a Rate Limit here", "dex"),
            make_chunk("b1", "binance", "an ORDER and a rate limit"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus.sources_with_term(src, "rate limit") == ["binance", "pancakeswap"]
    assert corpus.sources_without_term(src, "rate limit") == ["uniswap"]
    assert corpus.sources_without_term(src, "order") == ["pancakeswap", "uniswap"]
