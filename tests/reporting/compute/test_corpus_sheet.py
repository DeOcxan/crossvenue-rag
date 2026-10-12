"""Tests for `reporting.compute.corpus_sheet`, on a corpus and manifest written under tmp_path."""

from __future__ import annotations

from dataclasses import replace

from crossvenue_rag.models import ChunkSource
from crossvenue_rag.reporting.compute import corpus_sheet
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks, write_file

MANIFEST_HEADER = "venue,venue_type,source_url,local_path,fetched_at,commit_sha,sha256\n"


def chunk(chunk_id, venue, text, *pages, tokens=None):
    """A chunk of `venue` occurring on each of `pages`, the first its primary page."""
    c = make_chunk(chunk_id, venue, text)
    sources = [ChunkSource(venue, p, ["H"], f"https://example.invalid/{p}", 0) for p in pages]
    return replace(c, doc_path=pages[0], sources=sources, token_count=tokens or c.token_count)


def binance_corpus(tmp_path):
    write_chunks(
        tmp_path,
        [
            # A mainnet page and its byte-identical testnet copy: one chunk.
            chunk("b1", "binance", "weight 20", "rest-api.md", "testnet/rest-api.md"),
            # The same text but for the word testnet: two chunks.
            chunk("b2", "binance", "host api.binance.com", "rest-api.md"),
            chunk("b3", "binance", "host testnet api.binance.com", "testnet/rest-api.md"),
            chunk("b4", "binance", "a change", "testnet/CHANGELOG.md"),
            chunk("b5", "binance", "terms", "testnet/TESTNET-TERMS-OF-USE.md"),
            chunk("b6", "binance", "only mainnet", "rest-api.md"),
        ],
    )
    return Sources(tmp_path)


def test_a_chunk_on_a_mainnet_page_and_its_testnet_copy_is_a_mirror_group(tmp_path):
    assert corpus_sheet.testnet_mirror_groups(binance_corpus(tmp_path)) == 1


def test_testnet_only_chunks_are_split_into_reference_terms_and_changelog(tmp_path):
    assert corpus_sheet.testnet_only(binance_corpus(tmp_path)) == corpus_sheet.OnlyOnTestnet(
        3, 1, 1, 1
    )


def test_chunks_are_counted_on_every_page_they_occur_on(tmp_path):
    pages = corpus_sheet.page_chunks(binance_corpus(tmp_path), "binance")
    assert pages["rest-api.md"] == 3
    assert pages["testnet/rest-api.md"] == 2
    src = binance_corpus(tmp_path)
    assert corpus_sheet.terms_pages(src) == ["testnet/TESTNET-TERMS-OF-USE.md"]
    assert corpus_sheet.page_chunk_count(src, "binance", corpus_sheet.terms_pages(src)) == 1
    assert (
        corpus_sheet.page_chunk_count(src, "binance", ["rest-api.md", "testnet/rest-api.md"])
        is None
    )


def test_a_chunk_with_the_term_but_not_the_other_is_named(tmp_path):
    write_chunks(
        tmp_path,
        [
            chunk("h1", "hyperliquid", "Funding is paid every hour", "a.md"),
            chunk("h2", "hyperliquid", "Interest shows EVERY HOUR", "b.md"),
            chunk("d1", "dydx", "every hour", "c.md"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus_sheet.chunk_without(src, "hyperliquid", "every hour", "funding") == "h2"


def test_chunks_longer_than_the_window_are_counted_with_whether_all_hold_code(tmp_path):
    write_chunks(
        tmp_path,
        [
            chunk("c1", "ccxt", "```py\nx\n```", "a.md", tokens=600),
            chunk("c2", "ccxt", "prose", "a.md", tokens=520),
            chunk("c3", "ccxt", "short", "a.md", tokens=100),
        ],
    )
    assert corpus_sheet.chunks_longer_than(Sources(tmp_path), 510) == (2, False)


def test_documents_without_a_chunk_and_url_sections_are_read_from_the_manifest(tmp_path):
    rows = [
        "pancakeswap,dex,https://docs.example.invalid/trade/a,pancakeswap/trade/a.md,"
        "2026-09-20T00:00:00Z,,x",
        "pancakeswap,dex,https://docs.example.invalid/trade/b,pancakeswap/trade/b.md,"
        "2026-09-20T00:00:00Z,,x",
        "pancakeswap,dex,https://docs.example.invalid/earn/c,pancakeswap/earn/c.md,"
        "2026-09-21T00:00:00Z,,x",
    ]
    write_file(tmp_path, "data/manifest.csv", MANIFEST_HEADER + "\n".join(rows) + "\n")
    write_chunks(tmp_path, [chunk("p1", "pancakeswap", "swap", "trade/a.md")])
    src = Sources(tmp_path)
    assert corpus_sheet.url_sections(src, "pancakeswap") == {"trade": 2, "earn": 1}
    assert corpus_sheet.documents_without_chunks(src) == {"pancakeswap": 2}
    assert corpus_sheet.fetch_dates(src) == {"2026-09-20", "2026-09-21"}
    assert corpus_sheet.manifest_documents_under(src, "pancakeswap", "trade/") == 2


def test_sources_without_a_license_and_documents_per_kind_are_counted(tmp_path):
    write_file(
        tmp_path,
        "data/sources.yaml",
        "sources:\n"
        "  - {venue: binance, kind: git, venue_type: cex, license: no license found, repo: r}\n"
        "  - {venue: uniswap, kind: git, venue_type: dex, license: MIT, repo: r}\n"
        "  - {venue: dydx, kind: gitbook, venue_type: dex, license: no license found, "
        "llms_index: u}\n",
    )
    write_file(
        tmp_path,
        "data/manifest.csv",
        MANIFEST_HEADER + "dydx,dex,https://d.invalid/a,dydx/a.md,2026-09-20T00:00:00Z,,x\n",
    )
    src = Sources(tmp_path)
    assert corpus_sheet.sources_without_license(src) == (2, 3)
    assert corpus_sheet.kind_documents(src, "gitbook") == 1
