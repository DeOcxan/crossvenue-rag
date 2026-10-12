"""docs/results/findings.md: the corpus sections."""

from __future__ import annotations

from .... import formatting
from ...check import Doc
from ...compute import corpus, corpus_pancakeswap
from ...sources import Sources
from ..comparisons import duplicate_share_text, fee_tier_texts

# The best-match cosines the PancakeSwap table counts chunks at.
THRESHOLDS = (0.9, 0.7, 0.6, 0.5, 0.4)
# How the duplicate table names each venue.
VENUE_LABELS = {"binance": "binance (all mainnet/testnet)"}


def pancakeswap(doc: Doc, src: Sources) -> None:
    """PancakeSwap against Uniswap."""
    doc.check(
        "each of the {} PancakeSwap chunks against its best match among the {} Uniswap chunks",
        formatting.count(len(corpus.venue_chunks(src, "pancakeswap"))),
        formatting.count(len(corpus.venue_chunks(src, "uniswap"))),
    )
    doc.table(
        "| threshold |",
        {
            f">= {formatting.own_form(m.threshold, formatting.COSINE_THRESHOLD_DECIMALS)}": [
                f"{formatting.count(m.matched)} of {formatting.count(m.chunks)}"
            ]
            for m in corpus_pancakeswap.matches_at_least(src, THRESHOLDS)
        },
    )
    median = formatting.metric(corpus_pancakeswap.pancakeswap_median_cosine(src))
    doc.check("## PancakeSwap's median best-match cosine to Uniswap is {}", median)
    doc.check("Median best-match cosine: {}.", median)
    fees = fee_tier_texts()
    doc.claim(
        "different fee tiers ({}% vs {}%)",
        fees["uniswap"],
        fees["pancakeswap"],
        holds=corpus_pancakeswap.fee_tiers_stated(src),
        what=f"Uniswap's documentation states a {fees['uniswap']}% tier and PancakeSwap's a "
        f"{fees['pancakeswap']}% tier",
    )


def duplicate_claims(doc: Doc, src: Sources) -> None:
    """Duplicate text in the corpus."""
    dup = corpus.duplicates(src)
    doc.check(
        "Of {} chunk occurrences, {} ({}%) fall into {} groups of byte-identical text; {} of the "
        "groups are Binance",
        formatting.count(dup.occurrences),
        formatting.count(dup.in_groups),
        duplicate_share_text(src),
        formatting.count(dup.groups),
        formatting.count(dup.by_venue["binance"]),
    )
    doc.claim(
        "and no text occurs on {} venues",
        formatting.count(corpus.CROSS_VENUE_VENUES),
        holds=corpus.no_cross_venue_texts(src),
        what=f"no text on {formatting.count(corpus.CROSS_VENUE_VENUES)} venues",
    )
    doc.table(
        "| venue | groups |",
        {
            VENUE_LABELS.get(v, v): [formatting.count(k)]
            for v, k in corpus.duplicate_groups_by_venue(src)
        },
    )
    doc.check(
        "`testnet/web-socket-api.md` <-> `web-socket-api.md` ({} groups), `rest-api.md` <-> "
        "`testnet/rest-api.md` ({}) and `fix-api.md` <-> `testnet/fix-api.md` ({})",
        formatting.count(
            corpus.binance_page_groups(src, ("testnet/web-socket-api.md", "web-socket-api.md"))
        ),
        formatting.count(corpus.binance_page_groups(src, ("rest-api.md", "testnet/rest-api.md"))),
        formatting.count(corpus.binance_page_groups(src, ("fix-api.md", "testnet/fix-api.md"))),
    )
    doc.check("as one of {} chunks that keeps every", formatting.count(corpus.chunk_count(src)))
