"""What docs/adr/0007-venue-split.md claims.

The coverage failure of the first run on `v0.2-gold` (`run_ids.FIRST_GOLD_RUN`), and the BNB
Chain counts, by the count-term convention.
"""

from __future__ import annotations

from .... import formatting
from ....models import MULTI_VENUE
from ...check import Doc
from ...compute import corpus, datasets, retrieval, retrieval_earlier
from ...run_ids import FIRST_GOLD_RUN
from ...sources import Sources
from ..comparisons import top_k_text

BNB_CHAIN = "BNB Chain"


def venue_split_record(doc: Doc, src: Sources) -> None:
    """States the claims of docs/adr/0007-venue-split.md."""
    _coverage(doc, src)
    _bnb_chain(doc, src)
    doc.check("With fewer than {} venues named", formatting.count(MULTI_VENUE))
    doc.check(
        "Each detected venue's first chunk enters the top {}",
        top_k_text(src),
    )


def _share(src: Sources, method: str) -> list[str]:
    """A method's coverage on `v0.2-gold`, for the holes of `0.52 (26/50)`."""
    return [
        formatting.fixed(
            retrieval_earlier.first_gold_coverage(src, method), formatting.SHARE_DECIMALS
        ),
        formatting.count(retrieval_earlier.first_gold_covered(src, method)),
        formatting.count(datasets.comparison_count_before_augmentation(src)),
    ]


def _coverage(doc: Doc, src: Sources) -> None:
    """Context: no method covered every named venue on most comparison questions."""
    doc.ref("(`v0.2-gold`, `results/{}`)", FIRST_GOLD_RUN)
    doc.check(
        "none of the {} methods put a chunk from every named venue into the top {} of more "
        "than {} of the {} comparison questions",
        formatting.count(len(retrieval.ROUND1_POOL)),
        top_k_text(src),
        formatting.count(retrieval_earlier.first_gold_covered_most(src)),
        formatting.count(datasets.comparison_count_before_augmentation(src)),
    )
    doc.check(
        "venue_coverage@10 was {} ({}/{}) for bm25 and hybrid_rerank, {} ({}/{}) for hybrid "
        "and {} ({}/{}) for dense",
        *_share(src, "bm25"),
        *_share(src, "hybrid"),
        *_share(src, "dense"),
    )
    doc.claim(
        "venue_coverage@10 was {} ({}/{}) for bm25 and hybrid_rerank",
        *_share(src, "hybrid_rerank"),
        holds=retrieval_earlier.first_gold_coverage_equal(src, "bm25", "hybrid_rerank"),
        what="bm25 and hybrid_rerank have the same coverage",
    )
    doc.check(
        "The annotator recorded {} comparison venues as `missing_from_pool`",
        formatting.count(datasets.missing_from_pool_before_augmentation(src)),
    )


def _bnb_chain(doc: Doc, src: Sources) -> None:
    """Decision: why detection is a dictionary of venue names."""
    doc.check(
        '"BNB Chain" occurs in {} PancakeSwap chunks, {} Uniswap chunks and no Binance chunk',
        formatting.count(corpus.term_chunks(src, BNB_CHAIN, "pancakeswap")),
        formatting.count(corpus.term_chunks(src, BNB_CHAIN, "uniswap")),
    )
    doc.claim(
        "and no Binance chunk",
        holds=corpus.term_absent(src, BNB_CHAIN, "binance"),
        what='no Binance chunk contains "BNB Chain"',
    )
