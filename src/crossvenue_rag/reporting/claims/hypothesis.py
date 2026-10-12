"""The hypothesis paragraph that README.md shares with findings.md.

Both documents state what the retrieval run and the judged answers show
against the hypothesis in the same words, so one function checks both copies.
"""

from __future__ import annotations

from ... import formatting
from ...evaluation.comparisons import TOP_ONE_OFF_VENUE
from ..check import Doc
from ..compute import answers, corpus
from ..compute import retrieval_off_venue as off_venue
from ..compute.retrieval import METHODS, VENUE_FILTERED
from ..run_ids import JUDGED_V1, JUDGED_V2, RETRIEVAL_RUN
from ..sources import Sources
from .comparisons import coverage_parts, top_k_text

# The methods without a venue filter.
UNFILTERED = tuple(m for m in METHODS if m not in VENUE_FILTERED)


def hypothesis(doc: Doc, src: Sources) -> None:
    """The hypothesis paragraph: what the retrieval run and the judged answers show against it."""
    k = top_k_text(src)
    reranked = ("dense", "hybrid_rerank", "bm25")
    _sources_with_terms(doc, src)
    doc.ref("The retrieval run `results/{}` on `v0.2.1-gold`", RETRIEVAL_RUN)
    doc.check(
        "A CCXT page comes first on dense {}/{}, hybrid_rerank {}/{} and BM25 {}/{}.",
        *[
            formatting.count(v)
            for v in off_venue.first_rank_counts(
                src, off_venue.TOP_ONE_VIA_CCXT, "comparison", reranked
            )
        ],
    )
    doc.check(
        "a CCXT page comes first on BM25 {}/{}, dense {}/{}, hybrid {}/{} and hybrid_rerank {}/{}.",
        *[
            formatting.count(v)
            for v in off_venue.first_rank_counts(
                src, off_venue.TOP_ONE_VIA_CCXT, "adversarial", UNFILTERED
            )
        ],
    )
    doc.claim(
        "is rare on the {} comparison questions for dense ({}/{}) and hybrid_rerank ({}/{}); "
        "BM25 substitutes on {}/{}.",
        formatting.count(
            off_venue.first_rank_count(src, TOP_ONE_OFF_VENUE, "bm25", "comparison")[1]
        ),
        *[
            formatting.count(v)
            for v in off_venue.first_rank_counts(src, TOP_ONE_OFF_VENUE, "comparison", reranked)
        ],
        holds=off_venue.first_rank_at_most(
            src, TOP_ONE_OFF_VENUE, "comparison", ("dense", "hybrid_rerank"), 1
        ),
        what="dense and hybrid_rerank put an unnamed venue first on at most one comparison "
        "question",
    )
    doc.claim(
        "On the {} adversarial questions every method without a venue filter substitutes on "
        "{} or {}: BM25 {}/{}, dense {}/{}, hybrid {}/{}, hybrid_rerank {}/{};",
        formatting.count(
            off_venue.first_rank_count(src, TOP_ONE_OFF_VENUE, "bm25", "adversarial")[1]
        ),
        *[
            formatting.count(v)
            for v in off_venue.first_rank_spread(src, TOP_ONE_OFF_VENUE, "adversarial", UNFILTERED)
        ],
        *[
            formatting.count(v)
            for v in off_venue.first_rank_counts(src, TOP_ONE_OFF_VENUE, "adversarial", UNFILTERED)
        ],
        holds=off_venue.first_rank_adjacent(src, TOP_ONE_OFF_VENUE, "adversarial", UNFILTERED),
        what="the methods without a venue filter substitute on adjacent counts",
    )
    _missing_venue(doc, src, k)


def _sources_with_terms(doc: Doc, src: Sources) -> None:
    """The opening sentence: which sources have chunks with each of three terms."""
    doc.claim(
        'Of the {} documentation sources, all have chunks that contain "order", {} that contain '
        '"rate limit" (every source but PancakeSwap) and {} that contain "funding" (every source '
        "but Binance), counted with `crossvenue count-term`.",
        formatting.count(len(corpus.sources_with_term(src, "order"))),
        formatting.count(len(corpus.sources_with_term(src, "rate limit"))),
        formatting.count(len(corpus.sources_with_term(src, "funding"))),
        holds=corpus.sources_without_term(src, "order") == []
        and corpus.sources_without_term(src, "rate limit") == ["pancakeswap"]
        and corpus.sources_without_term(src, "funding") == ["binance"],
        what="every source has an order chunk; only PancakeSwap lacks rate limit, only Binance "
        "funding",
    )


def _missing_venue(doc: Doc, src: Sources, k: str) -> None:
    """The hypothesis paragraph, continued: a named venue missing from the top k."""
    doc.check("a named venue missing from the top {}:", k)
    doc.check(
        "hybrid_rerank's venue_coverage@{} on the comparison questions is {} ({}/{})",
        k,
        *coverage_parts(src, "hybrid_rerank"),
    )
    methods = ("dense", "hybrid", "hybrid_rerank")
    doc.claim(
        "more of the comparison top {} comes from CCXT's pages than from unnamed exchanges: "
        "{} against {}, {} against {} and {} against {}, a descriptive contrast",
        k,
        *[formatting.metric(v) for v in off_venue.top_k_shares(src, "comparison", methods)],
        holds=off_venue.via_ccxt_exceeds_off_venue(src, "comparison", methods),
        what="for dense, hybrid and hybrid_rerank the via-ccxt share exceeds the off-venue share",
    )
    doc.ref("In the judged answers (`results/{}` and `results/{}`)", JUDGED_V1, JUDGED_V2)
    doc.claim(
        "all {} claims Opus marks misattributed in both matrices cite a CCXT page for an "
        "exchange, the wrapper kind",
        formatting.count(answers.misattributed_count(src)),
        holds=answers.misattributions_are_wrappers(src),
        what="no cross-exchange or other misattribution in either judged matrix",
    )
