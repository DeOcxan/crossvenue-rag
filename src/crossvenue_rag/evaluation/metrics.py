"""Retrieval metrics under binary relevance: a chunk contains the answer or not, as a human judged.

The rank metrics take the ranked ids and the gold set and return a float in [0, 1]; the
venue metrics take the venues of the ranked chunks and of the question.
An empty gold set leaves a rank metric undefined, and the harness skips the question rather
than average in a 0.0 that would raise or lower a method's mean at random.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

from .. import formatting
from ..models import MULTI_VENUE


def _dedupe(ids: Sequence[str]) -> list[str]:
    """The ids in rank order with every repeat after the first removed.

    A rank list must not contain a document twice; a duplicate would inflate
    precision and recall for free.
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of the gold set found in the top k.

    Raises:
        ValueError: `k` is not positive, or the gold set is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {formatting.count(k)}")
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall is undefined for an empty gold set")
    top = set(_dedupe(retrieved)[:k])
    return len(top & gold_set) / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of the top k that is relevant.

    The denominator is k, not `len(retrieved[:k])`: a method that returns three
    results where ten were asked for does not get the precision of a method
    that returned ten good ones.

    Raises:
        ValueError: `k` is not positive, or the gold set is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {formatting.count(k)}")
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("precision is undefined for an empty gold set")
    top = _dedupe(retrieved)[:k]
    return sum(1 for doc_id in top if doc_id in gold_set) / k


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first relevant result, or 0.0 if none of the gold set is retrieved.

    Raises:
        ValueError: The gold set is empty.
    """
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("reciprocal rank is undefined for an empty gold set")
    for position, doc_id in enumerate(_dedupe(retrieved), start=1):
        if doc_id in gold_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """nDCG@k with binary gains.

    Raises:
        ValueError: `k` is not positive, or the gold set is empty.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {formatting.count(k)}")
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("nDCG is undefined for an empty gold set")

    top = _dedupe(retrieved)[:k]
    dcg = sum(1.0 / log2(i + 1) for i, doc_id in enumerate(top, start=1) if doc_id in gold_set)
    ideal_hits = min(k, len(gold_set))
    idcg = sum(1.0 / log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def ndcg_condensed_at_k(
    retrieved: Sequence[str], gold: Sequence[str], judged: set[str], k: int
) -> float:
    """nDCG'@k on the condensed list (Sakai 2007).

    Unjudged documents are removed from the ranking before nDCG@k is computed.
    Sakai 2007's full entry is under References in evals/methodology.md.

    Plain nDCG scores an unjudged document as irrelevant; the condensed list
    skips it instead, so the gap between the two is how much the incomplete
    judgments lower a method's nDCG. Gold documents are judged by definition,
    so they are never removed.

    The list condensed is the one given: the harness retrieves `max(k_values)`
    deep, so with unjudged documents in the top 10 the condensed list is
    shorter than 10 rather than refilled from further down.

    Raises:
        ValueError: `k` is not positive, or the gold set is empty.
    """
    kept = judged | set(gold)
    return ndcg_at_k([d for d in retrieved if d in kept], gold, k)


def evaluate(
    retrieved: Sequence[str], gold: Sequence[str], k_values: Sequence[int]
) -> dict[str, float]:
    """All metrics at every k, keyed `recall@5`, `ndcg@10`, ... plus `mrr`.

    Raises:
        ValueError: The gold set is empty.
    """
    scores: dict[str, float] = {"mrr": reciprocal_rank(retrieved, gold)}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    return scores


def venue_coverage_at_k(
    retrieved_venues: Sequence[str], question_venues: Sequence[str], k: int
) -> float:
    """1.0 if the top k hold a chunk from every venue the question names, else 0.0.

    Needs no gold labels: it measures only whether retrieval returned each venue
    at all, which is the precondition for answering a comparison. Defined for
    questions naming two or more venues; a one-venue question would measure
    plain venue precision instead.

    Raises:
        ValueError: The question names fewer than two venues.
    """
    if len(set(question_venues)) < MULTI_VENUE:
        raise ValueError(
            "venue_coverage is defined for questions naming "
            f"{formatting.count(MULTI_VENUE)} or more venues"
        )
    return 1.0 if set(question_venues) <= set(retrieved_venues[:k]) else 0.0


# Venues that document other venues' APIs: a chunk from one of them on a
# question that does not name it is counted as "via" the wrapper, apart from
# off-venue chunks (methodology section 9b).
WRAPPER_VENUES = ("ccxt",)


def off_venue_at_k(
    retrieved_venues: Sequence[str], question_venues: Sequence[str], k: int
) -> tuple[float, float]:
    """(off-venue share, via-wrapper share) of the top k (methodology section 9b).

    Off-venue: the chunk's venue is not among the venues the question names,
    and it is not a wrapper venue. Via wrapper: a chunk from a wrapper venue
    (ccxt) the question does not name. Both are shares of the chunks retrieved
    in the top k, so at k = 1 the off-venue share is the top-1 off-venue rate.
    Needs no gold labels. Defined for questions naming at least one venue.

    Raises:
        ValueError: The question names no venue.
    """
    named = set(question_venues)
    if not named:
        raise ValueError("off-venue share is defined for questions naming a venue")
    top = list(retrieved_venues[:k])
    if not top:
        return 0.0, 0.0
    via = sum(1 for v in top if v not in named and v in WRAPPER_VENUES)
    off = sum(1 for v in top if v not in named and v not in WRAPPER_VENUES)
    return off / len(top), via / len(top)
