"""PancakeSwap against Uniswap: how close the two forks' documentation is.

Each PancakeSwap chunk is matched against every Uniswap chunk by TF-IDF cosine
over the two venues' chunks; its best match is the highest cosine. The fee
tiers each venue states are counted the `crossvenue count-term` way.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from ...formatting import FEE_TIER_DECIMALS, own_form
from ...stats import P50, percentile
from ..sources import Sources
from .corpus import term_chunks, venue_chunks
from .lookup import entry
from .memo import per_sources

# The TF-IDF best-match cosine against Uniswap that no PancakeSwap chunk reaches.
PANCAKESWAP_COSINE_BOUND = 0.7
# The PancakeSwap chunks a best-match cosine is computed for at once.
BATCH = 200
# The TF-IDF vocabulary: terms in at least two chunks, the 60,000 most frequent; the
# parameters behind the documents' best-match cosines (docs/how-to/reproduce.md).
MIN_DOCUMENT_FREQUENCY = 2
MAX_FEATURES = 60000
# The fee tier, in percent, each venue's documentation states.
FEE_TIERS = {"uniswap": 0.30, "pancakeswap": 0.25}


@per_sources
def best_match(src: Sources) -> NDArray[np.float64]:
    """TF-IDF best-match cosine of each PancakeSwap chunk against Uniswap's chunks."""
    uni = [c.text for c in venue_chunks(src, "uniswap")]
    pan = [c.text for c in venue_chunks(src, "pancakeswap")]
    vectorizer = TfidfVectorizer(
        min_df=MIN_DOCUMENT_FREQUENCY, max_features=MAX_FEATURES, stop_words="english"
    )
    x = vectorizer.fit_transform(uni + pan)
    u, p = x[: len(uni)], x[len(uni) :]
    best: NDArray[np.float64] = np.concatenate(
        [(p[i : i + BATCH] @ u.T).toarray().max(axis=1) for i in range(0, p.shape[0], BATCH)]
    )
    return best


def pancakeswap_median_cosine(src: Sources) -> float:
    """Median TF-IDF best-match cosine of a PancakeSwap chunk against Uniswap's chunks.

    The median is the nearest-rank one every report states (`stats.P50`).
    """
    return percentile([float(v) for v in best_match(src)], P50)


def pancakeswap_below_bound(src: Sources) -> bool:
    """Whether every PancakeSwap chunk's best-match cosine is below the bound."""
    return bool((best_match(src) < PANCAKESWAP_COSINE_BOUND).all())


@dataclass(frozen=True)
class ThresholdMatches:
    """The PancakeSwap chunks whose best Uniswap match reaches a threshold."""

    threshold: float
    matched: int
    chunks: int  # every PancakeSwap chunk


def matches_at_least(src: Sources, thresholds: Sequence[float]) -> list[ThresholdMatches]:
    """Per threshold, the PancakeSwap chunks whose best match reaches it."""
    best = best_match(src)
    return [ThresholdMatches(t, int((best >= t).sum()), len(best)) for t in thresholds]


def fee_tier_text(tier: float) -> str:
    """A fee tier as the documents write it, without the percent sign: `0.30`."""
    return own_form(tier, FEE_TIER_DECIMALS)


def fee_tier_terms(tier: float) -> set[str]:
    """A fee tier as a documentation writes it: `0.30%` and its shortest form, `0.3%`."""
    return {fee_tier_text(tier) + "%", f"{tier:g}%"}


def fee_tier_stated(src: Sources, venue: str) -> bool:
    """Whether a chunk of `venue` states its tier of `FEE_TIERS` (`fee_tier_terms`).

    Raises:
        ValueError: `venue` is not a venue of `FEE_TIERS`.
    """
    terms = fee_tier_terms(entry(FEE_TIERS, venue, "FEE_TIERS"))
    return sum(term_chunks(src, term, venue) for term in terms) > 0


def fee_tiers_stated(src: Sources) -> bool:
    """Whether each venue of `FEE_TIERS` has a chunk stating its tier (`fee_tier_stated`)."""
    return all(fee_tier_stated(src, venue) for venue in FEE_TIERS)
