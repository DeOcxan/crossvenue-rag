"""Tests for `reporting.compute.corpus_changes`, on a corpus written under tmp_path."""

from __future__ import annotations

import json
from dataclasses import replace

from crossvenue_rag.models import ChunkSource
from crossvenue_rag.reporting.compute import corpus_changes
from crossvenue_rag.reporting.run_ids import DEDUP_IDMAP
from crossvenue_rag.reporting.sources import Sources
from tests.conftest import make_chunk
from tests.reporting.conftest import write_chunks, write_file


def occurrence(doc_path):
    return ChunkSource("binance", doc_path, ["Limits"], f"https://example.invalid/{doc_path}", 0)


def test_the_id_map_gives_old_ids_and_their_new_chunks(tmp_path):
    idmap = {
        "old_ids": 3,
        "map": {"o1": "n1", "o2": "n1", "o3": "n2"},
        "old_chunks_sha256": "",
        "to_index_version": "",
    }
    write_file(tmp_path, DEDUP_IDMAP, json.dumps(idmap))
    src = Sources(tmp_path)
    assert corpus_changes.dedup_old_ids(src) == 3
    assert corpus_changes.dedup_idmap(src) == {"o1": "n1", "o2": "n1", "o3": "n2"}


def test_the_old_ids_are_counted_from_the_map_whatever_count_the_file_records(tmp_path):
    write_file(
        tmp_path,
        DEDUP_IDMAP,
        json.dumps(
            {"old_ids": 5, "map": {"o1": "n1"}, "old_chunks_sha256": "", "to_index_version": ""}
        ),
    )
    assert corpus_changes.dedup_old_ids(Sources(tmp_path)) == 1


def test_merging_removes_the_copies_and_their_tokens(tmp_path):
    copies = [occurrence("rest-api.md"), occurrence("testnet/rest-api.md")]
    write_chunks(
        tmp_path,
        [
            replace(make_chunk("g", "binance", "a b c"), sources=copies),
            replace(make_chunk("s", "binance", "d"), sources=[occurrence("x.md")]),
        ],
    )
    src = Sources(tmp_path)
    # Three occurrences stored as two chunks; the second copy's 3 tokens are gone.
    assert corpus_changes.dedup_chunk_change(src) == -1
    assert corpus_changes.dedup_token_change(src) == -3


def test_a_heading_led_chunk_misses_its_parent_heading_unless_its_text_holds_it(tmp_path):
    write_chunks(
        tmp_path,
        [
            replace(make_chunk("a", "bybit", "## Limits\ntext"), heading_path=["API", "Limits"]),
            replace(make_chunk("b", "bybit", "# API\n## Limits"), heading_path=["API", "Limits"]),
            replace(make_chunk("c", "bybit", "  # Top"), heading_path=["Top"]),
            make_chunk("d", "bybit", "no heading"),
        ],
    )
    src = Sources(tmp_path)
    assert corpus_changes.heading_led_chunks(src) == 3
    assert corpus_changes.missing_parent_heading(src) == 1


def test_binance_removals_count_the_three_kinds_of_page(tmp_path):
    # A changelog under another testnet directory is not the testnet changelog.
    paths = [
        "testnet/CHANGELOG.md",
        "x/testnet/CHANGELOG.md",
        "demo-mode/general.md",
        "Terms-of-use.md",
        "rest-api.md",
    ]
    chunks = [replace(make_chunk(f"b{i}", "binance", "x"), doc_path=p) for i, p in enumerate(paths)]
    chunks.append(replace(make_chunk("y", "bybit", "x"), doc_path="terms.md"))
    write_chunks(tmp_path, chunks)
    assert corpus_changes.binance_removals(Sources(tmp_path)) == corpus_changes.BinanceRemovals(
        1, 1, 1
    )


def test_quoted_phrases_and_archive_pages_of_a_venue(tmp_path):
    archive = "https://docs.pancakeswap.invalid/archive/"
    write_chunks(
        tmp_path,
        [
            make_chunk("p1", "pancakeswap", "Querying This Documentation a b", "dex"),
            make_chunk("p2", "pancakeswap", "querying this documentation", "dex"),
            replace(
                make_chunk("p3", "pancakeswap", "old", "dex"),
                source_url=archive + "v1",
                doc_path="archive/v1.md",
            ),
            replace(
                make_chunk("p4", "pancakeswap", "old too", "dex"),
                source_url=archive + "v1#b",
                doc_path="archive/v1.md",
            ),
            make_chunk("u1", "uniswap", "Querying This Documentation", "dex"),
        ],
    )
    src = Sources(tmp_path)
    phrase = "Querying This Documentation"
    # Case-sensitive, and the tokens are those of the matching chunks.
    assert corpus_changes.chunks_quoting(src, "pancakeswap", phrase) == (1, 5)
    assert corpus_changes.archive_chunks(src, "pancakeswap") == (2, 1)
