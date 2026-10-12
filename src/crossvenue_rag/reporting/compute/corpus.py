"""The corpus as the documents describe it: chunks, documents, duplicates and sources.

The repository's own files are in `repository`.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ...config import SCHEMA_DEFAULTS
from ...corpus.artifacts import count_term_in
from ...corpus.normalize import has_code_fence
from ...corpus.sources import SourceSpec, load_sources
from ...evaluation.metrics import WRAPPER_VENUES
from ...index_version import index_meta_path, require_built_index_version
from ...models import Chunk, MetadataField
from ..patterns import collapse
from ..sources import Sources
from .lookup import entry
from .memo import per_sources

# Venues a chunk text must occur on to be a cross-venue duplicate.
CROSS_VENUE_VENUES = 2
# The directory Binance's testnet copies of its pages are under.
TESTNET = "testnet/"
# The Binance pages that are not a mirrored reference page: the testnet changelog, and the
# pages with this word in their path, case-insensitive.
TESTNET_CHANGELOG = "testnet/CHANGELOG.md"
TERMS = "terms"
# Binance's REST and WebSocket pages, whose "Timing security" sections are compared.
TIMING_PAGES = ("rest-api.md", "web-socket-api.md")


def is_testnet_page(doc_path: str) -> bool:
    """Whether a Binance document path is a testnet page: one under `TESTNET`."""
    return doc_path.startswith(TESTNET)


def is_testnet_changelog(doc_path: str) -> bool:
    """Whether a Binance document path is the testnet changelog, `TESTNET_CHANGELOG`."""
    return doc_path == TESTNET_CHANGELOG


def is_terms_page(doc_path: str) -> bool:
    """Whether a document path holds `TERMS`, case-insensitive."""
    return TERMS in doc_path.lower()


def chunk_token_budget(src: Sources) -> int:
    """`chunking.max_tokens` of configs/default.yaml."""
    return src.default_config.chunking.max_tokens


def index_version(src: Sources) -> str:
    """The `index_version` of the built index, from its index_meta.json."""
    return require_built_index_version(src.path(index_meta_path(SCHEMA_DEFAULTS)))


@per_sources
def source_entries(src: Sources) -> list[SourceSpec]:
    """The sources of data/sources.yaml, read by the fetcher's reader (`load_sources`)."""
    return load_sources(src.path(SCHEMA_DEFAULTS.paths.sources))


def sources_of_kind(src: Sources, kind: str) -> int:
    """Entries of data/sources.yaml fetched as `kind` (`git`, `gitbook`, ...)."""
    return sum(1 for e in source_entries(src) if e.kind == kind)


def dex_venues(src: Sources) -> list[str]:
    """Venues data/sources.yaml marks `venue_type: dex`, sorted."""
    return sorted({e.venue for e in source_entries(src) if e.venue_type == "dex"})


def chunk_count(src: Sources) -> int:
    """Chunks in the corpus."""
    return len(src.chunks)


def manifest_count(src: Sources) -> int:
    """Documents in data/manifest.csv, each fetched from a source."""
    return len(src.manifest)


def chunked_documents(src: Sources) -> set[tuple[str, str]]:
    """The (venue, document path) pairs some chunk occurs in."""
    return {(o.venue, o.doc_path) for c in src.chunks for o in c.sources}


def chunked_document_count(src: Sources) -> int:
    """Documents of the manifest with at least one chunk: a (venue, path) some chunk occurs in."""
    return len(chunked_documents(src))


def venue_chunks(src: Sources, venue: str) -> list[Chunk]:
    """The chunks of `venue`, in corpus order."""
    return [c for c in src.chunks if c.venue == venue]


@dataclass(frozen=True)
class TestnetCopies:
    """The chunks of a Binance mainnet page, and how many its testnet copy repeats."""

    chunks: int
    identical: int  # with a byte-identical testnet occurrence
    # Identical to a testnet chunk once the word "testnet" is removed and whitespace collapsed;
    # a byte-identical copy counts here too.
    normalized: int


def without_testnet(text: str) -> str:
    """`text` with the word "testnet" removed, in any case, and whitespace collapsed."""
    return collapse(re.sub(r"(?i)testnet", "", text)).strip()


def testnet_copies(src: Sources, page: str) -> TestnetCopies:
    """The chunks of Binance's mainnet `page`, and those its testnet copy repeats."""

    def pages(c: Chunk) -> set[str]:
        return {s.doc_path for s in c.sources}

    binance = venue_chunks(src, "binance")
    copy = f"{TESTNET}{page}"
    main = [c for c in binance if page in pages(c)]
    identical = sum(1 for c in main if copy in pages(c))
    testnet = {without_testnet(c.text) for c in binance if copy in pages(c)}
    normalized = sum(1 for c in main if without_testnet(c.text) in testnet)
    return TestnetCopies(len(main), identical, normalized)


def quoting_chunks(src: Sources, venue: str, phrase: str) -> list[Chunk]:
    """The chunks of `venue` containing `phrase` verbatim, case-sensitive."""
    return [c for c in venue_chunks(src, venue) if phrase in c.text]


def sources_with_term(src: Sources, term: str) -> list[str]:
    """The venues with a chunk containing `term`, by the counting convention of `count-term`."""
    return sorted(count_term_in(src.lowered_texts, term).by_venue)


def venues_with_spellings(
    src: Sources, spellings: Sequence[str], venues: Sequence[str]
) -> dict[str, list[str]]:
    """Each spelling, with those of `venues` that have a chunk containing it (`count-term`)."""
    return {s: [v for v in sources_with_term(src, s) if v in venues] for s in spellings}


def sources_without_term(src: Sources, term: str) -> list[str]:
    """The venues of the corpus with no chunk containing `term`."""
    return sorted({venue for venue, _ in src.lowered_texts} - set(sources_with_term(src, term)))


def token_total(src: Sources) -> int:
    """Tokens of every chunk (`token_count`, counted with the embedder's tokenizer)."""
    return sum(c.token_count for c in src.chunks)


def word_total(src: Sources) -> int:
    """Whitespace-separated words of every chunk."""
    return sum(len(c.text.split()) for c in src.chunks)


def tokens_per_word(src: Sources) -> float:
    """`token_total` over `word_total`."""
    return token_total(src) / word_total(src)


def term_chunks(src: Sources, term: str, venue: str | None = None) -> int:
    """Chunks of `venue`, or of every venue, containing `term`, case-insensitive.

    The counting convention of `crossvenue count-term TERM [--venue VENUE]`.
    """
    return count_term_in(src.lowered_texts, term, venue).total


def term_absent(src: Sources, term: str, venue: str) -> bool:
    """Whether no chunk of `venue` contains `term` (`term_chunks` is 0)."""
    return term_chunks(src, term, venue) == 0


def absent_terms(src: Sources, terms: Sequence[str], venue: str) -> list[str]:
    """The `terms` no chunk of `venue` contains (`term_absent`)."""
    return [term for term in terms if term_absent(src, term, venue)]


def chunks_above(src: Sources, tokens: int) -> list[Chunk]:
    """The chunks of more than `tokens` tokens, in corpus order."""
    return [c for c in src.chunks if c.token_count > tokens]


def over_budget_chunks(src: Sources) -> tuple[int, int]:
    """(prose chunks, code chunks) above the chunk token budget.

    A chunk with a code fence is a code block kept whole.
    """
    over = chunks_above(src, chunk_token_budget(src))
    code = sum(has_code_fence(c.text) for c in over)
    return len(over) - code, code


def filled_chunks(src: Sources, field: MetadataField) -> int:
    """Chunks whose metadata `field` (`doc_type`, `product`, ...) is set."""
    return sum(getattr(c, field) is not None for c in src.chunks)


def null_chunks_by_venue(src: Sources, field: MetadataField) -> dict[str, int]:
    """Per venue, the chunks whose metadata `field` is null."""
    return dict(Counter(c.venue for c in src.chunks if getattr(c, field) is None))


def _timing_security(src: Sources) -> list[Chunk]:
    """Binance's REST and WebSocket chunks headed "Timing security"."""
    return [
        c
        for c in venue_chunks(src, "binance")
        if c.doc_path in TIMING_PAGES
        and c.heading_path
        and c.heading_path[-1].lower() == "timing security"
    ]


def timing_security_chunks(src: Sources) -> int:
    """Chunks of Binance's REST and WebSocket "Timing security" sections."""
    return len(_timing_security(src))


def timing_security_ratio(src: Sources) -> float:
    """The difflib ratio of Binance's REST and WebSocket "Timing security" chunks."""
    texts = {c.doc_path: c.text for c in _timing_security(src)}
    rest, web_socket = (entry(texts, p, "the Timing security chunks") for p in TIMING_PAGES)
    return difflib.SequenceMatcher(None, rest, web_socket).ratio()


def _merged(src: Sources) -> list[Chunk]:
    """Chunks stored once for several verbatim occurrences."""
    return [c for c in src.chunks if len(c.sources) > 1]


def _cross_venue_texts(src: Sources) -> int:
    """Chunk texts that two or more venues hold."""
    by_text: dict[str, set[str]] = defaultdict(set)
    for c in src.chunks:
        by_text[c.text].add(c.venue)
    return sum(1 for venues in by_text.values() if len(venues) >= CROSS_VENUE_VENUES)


def _binance_pages(merged: list[Chunk]) -> Counter[tuple[str, ...]]:
    """Binance's merged chunks per set of pages they occur on."""
    return Counter(
        tuple(sorted({s.doc_path for s in c.sources})) for c in merged if c.venue == "binance"
    )


@dataclass(frozen=True)
class Duplicates:
    """Exact deduplication in the corpus.

    `occurrences`: chunk occurrences before merging; `in_groups`: those in a
    merged group; `groups`: chunks with more than one occurrence; `by_venue`:
    those groups per venue; `cross_venue_texts`: texts held by two venues;
    `pages`: Binance groups per set of pages.
    """

    occurrences: int
    in_groups: int
    groups: int
    by_venue: Counter[str]
    cross_venue_texts: int
    pages: Counter[tuple[str, ...]]


@per_sources
def duplicates(src: Sources) -> Duplicates:
    """Exact deduplication in the corpus (`Duplicates` names each count)."""
    merged = _merged(src)
    return Duplicates(
        occurrences=sum(len(c.sources) for c in src.chunks),
        in_groups=sum(len(c.sources) for c in merged),
        groups=len(merged),
        by_venue=Counter(c.venue for c in merged),
        cross_venue_texts=_cross_venue_texts(src),
        pages=_binance_pages(merged),
    )


def no_cross_venue_texts(src: Sources) -> bool:
    """Whether no chunk text is held by two venues."""
    return int(duplicates(src).cross_venue_texts) == 0


def duplicate_groups_by_venue(src: Sources) -> list[tuple[str, int]]:
    """(venue, groups of byte-identical occurrences) per venue, in corpus order."""
    return list(duplicates(src).by_venue.items())


def binance_page_groups(src: Sources, pages: tuple[str, ...]) -> int:
    """Binance's groups of byte-identical occurrences on exactly the set of `pages`."""
    return int(duplicates(src).pages[tuple(sorted(pages))])


def venue_count(src: Sources) -> int:
    """Trading venues with at least one chunk in the corpus; a client library is no venue."""
    return len({c.venue for c in src.chunks} - set(WRAPPER_VENUES))
