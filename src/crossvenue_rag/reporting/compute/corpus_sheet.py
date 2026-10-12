"""The corpus as its datasheet (docs/reference/corpus.md) describes it.

The per-venue table is recomputed by the function `ingest` writes corpus_stats.json with.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from ...corpus.artifacts import CorpusStats, VenueStats
from ...corpus.ingest import corpus_stats, doc_path_of, documents_by_venue
from ...corpus.normalize import has_code_fence
from ..sources import Sources
from .corpus import (
    chunked_documents,
    chunks_above,
    is_terms_page,
    is_testnet_changelog,
    is_testnet_page,
    source_entries,
    venue_chunks,
)
from .distinct import shared_value
from .memo import per_sources

# The license data/sources.yaml records for a source that states none.
NO_LICENSE = "no license found"


@per_sources
def corpus_statistics(src: Sources) -> CorpusStats:
    """The corpus statistics `ingest` writes, recomputed from the chunks and the manifest."""
    return corpus_stats(src.chunks, documents_by_venue(src.manifest))


def smallest_venue(src: Sources) -> VenueStats:
    """The statistics of the venue with the fewest chunks; ValueError if there is no venue."""
    return min(corpus_statistics(src).by_venue, key=lambda v: v.chunks)


def testnet_mirror_groups(src: Sources) -> int:
    """Binance chunks stored once for a mainnet page and its byte-identical testnet copy."""
    return sum(
        1
        for c in venue_chunks(src, "binance")
        if any(is_testnet_page(o.doc_path) for o in c.sources)
        and any(not is_testnet_page(o.doc_path) for o in c.sources)
    )


@dataclass(frozen=True)
class OnlyOnTestnet:
    """Binance chunks that occur only on testnet pages, by the page they come from."""

    chunks: int
    mirrored: int  # of the mirrored reference pages
    terms: int  # of the testnet terms page
    changelog: int  # of `testnet/CHANGELOG.md`


def testnet_only(src: Sources) -> OnlyOnTestnet:
    """Binance chunks that occur only on testnet pages."""
    only = [
        c.doc_path
        for c in venue_chunks(src, "binance")
        if all(is_testnet_page(o.doc_path) for o in c.sources)
    ]
    changelog = sum(is_testnet_changelog(p) for p in only)
    terms = sum(is_terms_page(p) for p in only)
    return OnlyOnTestnet(len(only), len(only) - changelog - terms, terms, changelog)


def page_chunks(src: Sources, venue: str) -> Counter[str]:
    """Chunks per page of `venue`, a chunk counted on every page it occurs on."""
    return Counter(o.doc_path for c in venue_chunks(src, venue) for o in c.sources)


def terms_pages(src: Sources) -> list[str]:
    """The Binance pages with `terms` in their path (`corpus.is_terms_page`), sorted."""
    return sorted(page for page in page_chunks(src, "binance") if is_terms_page(page))


def page_chunk_count(src: Sources, venue: str, pages: Sequence[str]) -> int | None:
    """The chunks each of `pages` of `venue` has (`page_chunks`); None unless they all match."""
    counts = page_chunks(src, venue)
    return shared_value(counts[page] for page in pages)


def manifest_documents_under(src: Sources, venue: str, directory: str) -> int:
    """Manifest documents of `venue` whose path is under `directory` (`docs/v3/`)."""
    return sum(1 for r in src.manifest if r.venue == venue and doc_path_of(r).startswith(directory))


def url_sections(src: Sources, venue: str) -> Counter[str]:
    """Manifest documents of `venue` per first segment of their URL's path."""
    return Counter(
        urlparse(r.source_url).path.strip("/").split("/")[0]
        for r in src.manifest
        if r.venue == venue
    )


def chunks_longer_than(src: Sources, tokens: int) -> tuple[int, bool]:
    """(chunks of more than `tokens` tokens, whether each holds a code fence)."""
    long = chunks_above(src, tokens)
    return len(long), all(has_code_fence(c.text) for c in long)


def documents_without_chunks(src: Sources) -> Counter[str]:
    """Manifest documents that yield no chunk, per venue."""
    chunked = chunked_documents(src)
    return Counter(r.venue for r in src.manifest if (r.venue, doc_path_of(r)) not in chunked)


def fetch_dates(src: Sources) -> set[str]:
    """The dates (YYYY-MM-DD) the manifest's documents were fetched on."""
    return {r.fetched_at.partition("T")[0] for r in src.manifest}


def chunk_without(src: Sources, venue: str, term: str, other: str) -> str | None:
    """The id of `venue`'s one chunk with `term` but not `other`, case-insensitive; else None."""
    term, other = term.lower(), other.lower()
    found = [
        c.chunk_id
        for c in venue_chunks(src, venue)
        if term in c.text.lower() and other not in c.text.lower()
    ]
    return found[0] if len(found) == 1 else None


def sources_without_license(src: Sources) -> tuple[int, int]:
    """(sources data/sources.yaml records with no license, sources)."""
    entries = source_entries(src)
    return sum(e.license == NO_LICENSE for e in entries), len(entries)


def kind_documents(src: Sources, kind: str) -> int:
    """Manifest documents of the venues whose source is fetched as `kind`."""
    venues = {e.venue for e in source_entries(src) if e.kind == kind}
    return sum(1 for r in src.manifest if r.venue in venues)
