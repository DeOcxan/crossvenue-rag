"""Changes to the corpus: the exact deduplication made, the corrections deferred.

The deduplication is recorded by its id map (`run_ids.DEDUP_IDMAP`), which maps
each chunk id of the corpus before it to the id of the chunk that holds its text
after it. The corrections are those the next corpus version applies (ADR 0012).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...index.migrate import IdMap
from ...models import Chunk
from ..run_ids import DEDUP_IDMAP
from ..sources import Sources
from .corpus import (
    duplicates,
    is_terms_page,
    is_testnet_changelog,
    quoting_chunks,
    venue_chunks,
)

# The Binance pages the next corpus version removes, as ADR 0012 lists them: the testnet
# changelog (`TESTNET_CHANGELOG`), the `demo-mode/` pages and the `TERMS` pages.
BINANCE_DEMO_MODE = "demo-mode/"


def dedup_idmap(src: Sources) -> dict[str, str]:
    """Chunk id before deduplication -> the id of the chunk that holds its text after it."""
    return IdMap.load(src.path(DEDUP_IDMAP)).mapping


def dedup_old_ids(src: Sources) -> int:
    """Chunk ids the corpus had before deduplication: the ids the id map maps."""
    return len(dedup_idmap(src))


def dedup_chunk_change(src: Sources) -> int:
    """Chunks the index holds minus the chunk occurrences before merging (<= 0)."""
    return len(src.chunks) - int(duplicates(src).occurrences)


def dedup_token_change(src: Sources) -> int:
    """Tokens removed by merging, negated: each merged copy's tokens after the first (<= 0)."""
    return -sum(c.token_count * (len(c.sources) - 1) for c in src.chunks)


def _heading_led(src: Sources) -> list[Chunk]:
    """Chunks whose text starts with a Markdown heading."""
    return [c for c in src.chunks if c.text.lstrip().startswith("#")]


def heading_led_chunks(src: Sources) -> int:
    """Chunks whose text starts with a Markdown heading."""
    return len(_heading_led(src))


def missing_parent_heading(src: Sources) -> int:
    """Heading-led chunks two or more levels deep whose text lacks a parent heading."""
    return sum(
        len(c.heading_path) > 1 and not all(h in c.text for h in c.heading_path[:-1])
        for c in _heading_led(src)
    )


@dataclass(frozen=True)
class BinanceRemovals:
    """Binance chunks of the pages the next version removes, by page."""

    changelog: int  # of the testnet changelog
    demo_mode: int  # of `demo-mode/` pages
    terms: int  # of pages with `terms` in their path, case-insensitive


def binance_removals(src: Sources) -> BinanceRemovals:
    """Binance chunks of the pages the next version removes."""
    binance = venue_chunks(src, "binance")
    return BinanceRemovals(
        sum(is_testnet_changelog(c.doc_path) for c in binance),
        sum(BINANCE_DEMO_MODE in c.doc_path for c in binance),
        sum(is_terms_page(c.doc_path) for c in binance),
    )


def chunks_quoting(src: Sources, venue: str, phrase: str) -> tuple[int, int]:
    """(chunks of `venue` containing `phrase` verbatim, case-sensitive, their tokens)."""
    found = quoting_chunks(src, venue, phrase)
    return len(found), sum(c.token_count for c in found)


def archive_chunks(src: Sources, venue: str) -> tuple[int, int]:
    """(chunks of `venue` from an `/archive/` URL, the pages they come from)."""
    found = [c for c in venue_chunks(src, venue) if "/archive/" in c.source_url]
    return len(found), len({c.doc_path for c in found})
