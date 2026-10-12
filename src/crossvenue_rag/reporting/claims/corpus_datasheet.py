"""What docs/reference/corpus.md claims.

Each function checks one section of the document or a part of one, in document order.
"""

from __future__ import annotations

from ... import formatting
from ...corpus.ingest import LENGTH_BINS
from ...models import MetadataField
from ..check import Doc
from ..compute import corpus, corpus_changes, corpus_sheet
from ..compute.corpus import TESTNET_CHANGELOG
from ..compute.corpus_changes import BINANCE_DEMO_MODE
from ..outcomes import SKIP
from ..sources import Sources
from .adr.adr_0012 import PANCAKESWAP_FURNITURE
from .comparisons import optional_count
from .reasons import (
    BGE_SPECIAL_TOKENS,
    BGE_TOKENIZER,
    BGE_WINDOW,
    CITED,
    COLLAPSED_FENCES,
    COLLAPSED_FENCES_REASON,
)

# The first cell of each venue's row of the composition table.
VENUE_CELLS = {
    "ccxt": "[CCXT](https://github.com/ccxt/ccxt)",
    "bybit": "[Bybit](https://github.com/bybit-exchange/docs)",
    "binance": "[Binance](https://github.com/binance/binance-spot-api-docs)",
    "bingx": "[BingX](https://github.com/BingX-API/api-ai-skills)",
    "uniswap": "[Uniswap](https://github.com/Uniswap/docs)",
    "dydx": "[dYdX v4](https://github.com/dydxprotocol/v4-documentation)",
    "hyperliquid": "[Hyperliquid](https://hyperliquid.gitbook.io/hyperliquid-docs)",
    "pancakeswap": "[PancakeSwap](https://docs.pancakeswap.finance/)",
}
# The PancakeSwap URL sections the datasheet names, in its order.
PANCAKESWAP_SECTIONS = ("welcome-to-pancakeswap", "trade", "earn", "archive")
# The tokens of text the bi-encoder reads: its window less its special tokens.
ENCODER_TEXT_TOKENS = BGE_WINDOW - BGE_SPECIAL_TOKENS
RERANKER_TOKENIZER = (
    "the model_max_length of the bge-reranker-base tokenizer and the four special tokens it "
    "adds to a (query, chunk) pair, read from the Hugging Face cache; no saved file holds them"
)
CCXT_SIZE = (
    "the `size` of ccxt/ccxt from the GitHub API, retrieved 2026-09-26; no saved file holds it"
)
QUOTED = "quoted documentation"
# The metadata fields in the order the datasheet writes them (`models.METADATA_FIELDS`).
FIELD_ORDER: tuple[MetadataField, ...] = ("doc_type", "api_version", "product", "environment")


def corpus_datasheet(doc: Doc, src: Sources) -> None:
    """States the claims of docs/reference/corpus.md."""
    _header(doc, src)
    _composition(doc, src)
    _included(doc, src)
    _gaps(doc, src)
    _limitations(doc, src)
    _collection(doc, src)
    _preprocessing(doc, src)
    _counting(doc, src)
    _distribution(doc, src)


def _header(doc: Doc, src: Sources) -> None:
    doc.definition("(Gebru et al., 2021, [doi:10.1145/3458723]", reason=CITED)
    doc.ref("ingested at `index_version` `{}`", corpus.index_version(src))
    doc.check(
        "The documentation of the {} venues and the CCXT library",
        formatting.count(corpus.venue_count(src)),
    )


def _composition(doc: Doc, src: Sources) -> None:
    stats = corpus_sheet.corpus_statistics(src)
    rows: dict[str, list[object]] = {
        VENUE_CELLS[v.venue]: [
            SKIP,
            SKIP,
            SKIP,
            *(formatting.count(n) for n in (v.documents, v.chunks, v.tokens_total)),
            *(formatting.count(n) for n in (v.tokens_p50, v.tokens_p95)),
        ]
        for v in stats.by_venue
    }
    t = stats.totals
    rows["total"] = [
        SKIP,
        SKIP,
        SKIP,
        *(formatting.count(n) for n in (t.documents, t.chunks, t.tokens, t.tokens_p50)),
        formatting.count(t.tokens_p95),
    ]
    doc.table("| Venue | Type |", rows)
    bins: list[str] = []
    values: list[str] = []
    counts = stats.chunk_length_histogram.values()
    for (low, high), n in zip(LENGTH_BINS, counts, strict=True):
        if high is not None:
            bins.append("{}--{}: {}")
            values += [formatting.count(low), formatting.count(high - 1)]
        else:
            bins.append("{} and over: {}")
            values.append(formatting.count(low))
        values.append(formatting.count(n))
    doc.check(
        "as `make stats` bins them (lower bound included, upper bound excluded): "
        + "; ".join(bins)
        + ".",
        *values,
    )
    prose, code = corpus.over_budget_chunks(src)
    doc.check(
        "Over the {}-token budget are {} prose chunks and {} fenced code blocks",
        formatting.count(corpus.chunk_token_budget(src)),
        formatting.count(prose),
        formatting.count(code),
    )
    chunks = corpus.chunk_count(src)
    doc.check(
        "`doc_type` {}%, `api_version` {}%, `product` {}%, `environment` {}%",
        *[
            formatting.percent_figure(corpus.filled_chunks(src, field), chunks)
            for field in FIELD_ORDER
        ],
    )
    dup = corpus.duplicates(src)
    doc.check(
        "{} chunk occurrences are stored as {} chunks: {} groups of byte-identical text within "
        "a venue, {} of them Binance mainnet pages",
        formatting.count(dup.occurrences),
        formatting.count(chunks),
        formatting.count(dup.groups),
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
    )
    doc.claim(
        "No text occurs on {} venues ([findings]",
        formatting.count(corpus.CROSS_VENUE_VENUES),
        holds=corpus.no_cross_venue_texts(src),
        what=f"no text on {formatting.count(corpus.CROSS_VENUE_VENUES)} venues",
    )


def _included(doc: Doc, src: Sources) -> None:
    doc.definition("the two are one chunk", reason="a mainnet page and its testnet copy")
    doc.check(
        "with the testnet page in `sources` ({} chunks)",
        formatting.count(corpus_sheet.testnet_mirror_groups(src)),
    )
    only = corpus_sheet.testnet_only(src)
    doc.check(
        "{} Binance chunks occur only on testnet pages: {} from the mirrored reference pages, "
        "{} from the testnet terms page and {} from `testnet/CHANGELOG.md`",
        *[formatting.count(n) for n in (only.chunks, only.mirrored, only.terms, only.changelog)],
    )
    pages = corpus_sheet.page_chunks(src, "binance")
    demo = [f"{BINANCE_DEMO_MODE}{page}" for page in ("CHANGELOG.md", "general-info.md")]
    terms = corpus_sheet.terms_pages(src)
    # Pages written with one count: a second count fails the comparison.
    demo_count = corpus_sheet.page_chunk_count(src, "binance", demo)
    terms_count = corpus_sheet.page_chunk_count(src, "binance", terms)
    doc.check(
        "`testnet/CHANGELOG.md` ({} chunks), `demo-mode/CHANGELOG.md` and "
        "`demo-mode/general-info.md` ({} each), and the {} terms-of-use pages ({} each)",
        formatting.count(pages[TESTNET_CHANGELOG]),
        optional_count(demo_count),
        formatting.count(len(terms)),
        optional_count(terms_count),
    )
    doc.check(
        "superseded by v5 and {} documents wide",
        formatting.count(corpus_sheet.manifest_documents_under(src, "bybit", "docs/v3/")),
    )


def _gaps(doc: Doc, src: Sources) -> None:
    doc.definitions(("Hyperliquid's is 1 hour", '"funding interval is 1 hour"'), reason=QUOTED)
    interest = corpus_sheet.chunk_without(src, "hyperliquid", "every hour", "funding")
    doc.check(
        ': {} chunk; "every hour": {}, one of them,',
        formatting.count(corpus.term_chunks(src, "funding interval is 1 hour", "hyperliquid")),
        formatting.count(corpus.term_chunks(src, "every hour", "hyperliquid")),
    )
    doc.ref("one of them, `{}`, about the interest", interest or "")
    doc.check(
        '("every hour": {}). Bybit',
        formatting.count(corpus.term_chunks(src, "every hour", "dydx")),
    )
    doc.check(
        "`fundingInterval`. BingX's {} chunks containing",
        formatting.count(corpus.term_chunks(src, "funding", "bingx")),
    )
    smallest = corpus_sheet.smallest_venue(src)
    doc.claim(
        "It is the smallest venue, at {} chunks.",
        formatting.count(smallest.chunks),
        holds=smallest.venue == "bingx",
        what="BingX has the fewest chunks of any venue",
    )


def _limitations(doc: Doc, src: Sources) -> None:
    missing = formatting.count(corpus_changes.missing_parent_heading(src))
    doc.check(
        "{} of the {} chunks that start at a heading",
        missing,
        formatting.count(corpus_changes.heading_led_chunks(src)),
    )
    doc.check(
        'The {} Binance "Timing security" chunks',
        formatting.count(corpus.timing_security_chunks(src)),
    )
    doc.check("the id, of those {} chunks", missing)
    sections = corpus_sheet.url_sections(src, "pancakeswap")
    doc.check(
        "`welcome-to-pancakeswap` ({} pages), `trade` ({}), `earn` ({}), `archive` ({})",
        *[formatting.count(sections[s]) for s in PANCAKESWAP_SECTIONS],
    )
    archived, archive_pages = corpus_changes.archive_chunks(src, "pancakeswap")
    doc.check(
        "The {} chunks of the {} archived pages",
        formatting.count(archived),
        formatting.count(archive_pages),
    )
    doc.check("the metadata of those {} chunks", formatting.count(archived))
    furniture, tokens = corpus_changes.chunks_quoting(src, "pancakeswap", PANCAKESWAP_FURNITURE)
    doc.check(
        "{} PancakeSwap chunks ({} tokens) are the",
        formatting.count(furniture),
        formatting.count(tokens),
    )
    doc.check("Stripping them removes the {} chunks.", formatting.count(furniture))
    over, code = corpus_sheet.chunks_longer_than(src, ENCODER_TEXT_TOKENS)
    doc.claim(
        "- {} chunks exceed",
        formatting.count(over),
        holds=code,
        what="every chunk longer than the bi-encoder reads is a code block kept whole",
    )
    doc.given(
        "exceed {} tokens of text, the reading window of the bi-encoder ({} with its {} special",
        formatting.count(ENCODER_TEXT_TOKENS),
        formatting.count(BGE_WINDOW),
        formatting.count(BGE_SPECIAL_TOKENS),
        reason=BGE_TOKENIZER,
    )
    doc.given(
        "reads the query and the chunk in one window of {} tokens with {} special tokens",
        "512",
        "4",
        reason=RERANKER_TOKENIZER,
    )
    empty = corpus_sheet.documents_without_chunks(src)
    documents = corpus.manifest_count(src)
    doc.claim(
        "- {} dYdX pages normalize to headings alone or to nothing and yield no chunk, so {} "
        "of the {} "
        "documents are in the index.",
        formatting.count(empty["dydx"]),
        formatting.count(corpus.chunked_document_count(src)),
        formatting.count(documents),
        holds=empty.keys() == {"dydx"},
        what="every document without a chunk is a dYdX page",
    )
    doc.given("inside {} fenced code blocks", COLLAPSED_FENCES, reason=COLLAPSED_FENCES_REASON)


def _collection(doc: Doc, src: Sources) -> None:
    doc.definition("Two acquisition modes", reason="the two modes listed after it")
    doc.definitions(("a clone at depth 1", "by its sha at depth 1"), reason="a git clone depth")
    doc.given("is {} KB by the GitHub API's `size` field", "8,537,212", reason=CCXT_SIZE)
    doc.claim(
        "for all {} documents.",
        formatting.count(corpus.manifest_count(src)),
        holds=corpus_sheet.fetch_dates(src) == {"2026-09-20"},
        what="every document of the manifest was fetched on 2026-09-20",
    )


def _preprocessing(doc: Doc, src: Sources) -> None:
    doc.given(
        "blank-line runs collapsed in {} blocks", COLLAPSED_FENCES, reason=COLLAPSED_FENCES_REASON
    )
    chunking = src.default_config.chunking
    doc.check(
        "once a chunk reaches {} tokens and keeps fenced code atomic up to {} tokens",
        formatting.count(chunking.min_tokens),
        formatting.count(chunking.max_code_block_tokens),
    )
    doc.check(
        "and holds the rendered chunk to {} tokens",
        formatting.count(corpus.chunk_token_budget(src)),
    )
    doc.check(
        "after the last code block, whole and up to {} tokens, into the next chunk",
        formatting.count(chunking.overlap_tokens),
    )


def _counting(doc: Doc, src: Sources) -> None:
    doc.definition("so two people counting", reason="any two people")
    doc.check(
        "`GTX` = {} chunks, both a substring",
        formatting.count(corpus.term_chunks(src, "GTX")),
    )


def _distribution(doc: Doc, src: Sources) -> None:
    doc.check(
        "from the {} git sources at their recorded commits",
        formatting.count(corpus.sources_of_kind(src, "git")),
    )
    doc.check(
        "The {} GitBook sources (Hyperliquid, PancakeSwap: {} documents)",
        formatting.count(corpus.sources_of_kind(src, "gitbook")),
        formatting.count(corpus_sheet.kind_documents(src, "gitbook")),
    )
    doc.check(
        "{} of the {} sources record no license",
        *[formatting.count(n) for n in corpus_sheet.sources_without_license(src)],
    )
