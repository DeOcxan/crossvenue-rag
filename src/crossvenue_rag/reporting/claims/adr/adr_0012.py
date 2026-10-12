"""What docs/adr/0012-deferred-corpus-changes.md claims.

Each deferred correction, and the fenced code blocks it changes.
"""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import corpus, corpus_changes
from ...sources import Sources
from ..reasons import COLLAPSED_FENCES, COLLAPSED_FENCES_REASON

# The navigation text PancakeSwap's pages repeat, which the next version removes.
PANCAKESWAP_FURNITURE = "Querying This Documentation"
THE_TABLE = "the table that follows lists them"


def deferred_corpus_changes_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0012-deferred-corpus-changes.md."""
    doc.ref("refers to index_version `{}`", corpus.index_version(src))
    doc.definition("found five defects", reason=THE_TABLE)
    doc.definition("The five corrections wait", reason=THE_TABLE)
    doc.definition("applies all five together", reason="the five rows of the table")
    _headings(doc, src)
    _removals(doc, src)
    doc.given("inside {} fenced code blocks", COLLAPSED_FENCES, reason=COLLAPSED_FENCES_REASON)
    archive = corpus_changes.archive_chunks(src, "pancakeswap")
    doc.check(
        "metadata of the {} chunks from {} `archive/` pages",
        *[formatting.count(v) for v in archive],
    )


def _headings(doc: Doc, src: Sources) -> None:
    """Decision: heading-led chunks that lack a parent heading."""
    missing = formatting.count(corpus_changes.missing_parent_heading(src))
    doc.check(
        "{} of the {} heading-led chunks miss a parent heading",
        missing,
        formatting.count(corpus_changes.heading_led_chunks(src)),
    )
    doc.check("text and id of {} chunks", missing)


def _removals(doc: Doc, src: Sources) -> None:
    """Decision: the Binance pages and PancakeSwap furniture removed."""
    removed = corpus_changes.binance_removals(src)
    doc.check(
        "removes `testnet/CHANGELOG.md` ({} chunks), `demo-mode/` ({}) and the terms pages ({})",
        *[formatting.count(v) for v in (removed.changelog, removed.demo_mode, removed.terms)],
    )
    furniture = corpus_changes.chunks_quoting(src, "pancakeswap", PANCAKESWAP_FURNITURE)
    doc.check(
        "removes {} PancakeSwap chunks ({} tokens)",
        *[formatting.count(v) for v in furniture],
    )
