"""Tests for `crossvenue_rag.corpus.ingest`."""

from __future__ import annotations

import json

from crossvenue_rag.config import Config
from crossvenue_rag.corpus.artifacts import corpus_stats_json, read_corpus_stats
from crossvenue_rag.corpus.ingest import doc_path_of
from crossvenue_rag.corpus.manifest import ManifestRow
from crossvenue_rag.models import VenueType
from tests.conftest import CORPUS, build_corpus


def test_ingestion_is_deterministic(cfg: Config):
    first = [c.chunk_id for c in build_corpus(cfg)]
    second = [c.chunk_id for c in build_corpus(cfg)]
    assert first == second


def test_ingestion_merges_a_testnet_mirror_into_one_mainnet_cited_chunk(cfg: Config):
    """433 of the real corpus's 536 identical-text groups were Binance mainnet
    pages against their testnet/ copies. After ingestion each is one chunk that
    cites the mainnet page and keeps the testnet page as a second source."""
    mainnet_text = CORPUS["binance/rest-api.md"][2]
    extra: dict[str, tuple[str, VenueType, str]] = {
        "binance/testnet/rest-api.md": ("binance", "cex", mainnet_text),
        # Same words on another venue: never merged across the venue boundary.
        "bybit/copied.md": ("bybit", "cex", mainnet_text),
    }
    chunks = build_corpus(cfg, extra)

    binance = [c for c in chunks if c.venue == "binance"]
    assert binance and all(len(c.sources) == 2 for c in binance)
    for chunk in binance:
        assert chunk.doc_path == "rest-api.md"
        assert chunk.environment == "mainnet"
        assert [s.environment for s in chunk.sources] == ["mainnet", "testnet"]
        assert [s.doc_path for s in chunk.sources] == ["rest-api.md", "testnet/rest-api.md"]

    bybit_copies = [c for c in chunks if c.venue == "bybit" and c.doc_path == "copied.md"]
    assert {c.text for c in bybit_copies} == {c.text for c in binance}
    assert not {c.chunk_id for c in bybit_copies} & {c.chunk_id for c in binance}

    stats = json.loads((cfg.paths.processed / "corpus_stats.json").read_text(encoding="utf-8"))
    assert stats["dedup"]["groups"] == len(binance)
    assert stats["dedup"]["testnet_mirror_groups"] == len(binance)
    assert stats["dedup"]["cross_venue_groups"] == len(binance)
    assert stats["totals"]["chunks"] == len(chunks)
    text = (cfg.paths.processed / "corpus_stats.json").read_text(encoding="utf-8")
    assert corpus_stats_json(read_corpus_stats(cfg)) == text


def test_the_ingest_log_line_formats_its_counts_and_plurals(cfg: Config, caplog):
    import logging
    import re

    with caplog.at_level(logging.INFO, logger="crossvenue_rag.corpus.ingest"):
        chunks = build_corpus(cfg)
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("ingested")]
    assert re.fullmatch(
        r"ingested [\d,]+ documents? into [\d,]+ chunks? \([\d,]+ before exact dedup, "
        r"[\d,]+ groups?, [\d,]+ cross-venue groups? left separate\)",
        line,
    ), line
    assert f"into {len(chunks)} chunk" in line


def test_a_document_path_drops_the_venue_segment_of_the_local_path():
    def row(local_path: str) -> ManifestRow:
        return ManifestRow("bybit", "cex", "https://example.invalid", local_path, "", "", "")

    assert doc_path_of(row("bybit/v5/order/create.md")) == "v5/order/create.md"
    assert doc_path_of(row("create.md")) == "create.md"
