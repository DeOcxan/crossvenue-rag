"""Reciprocal Rank Fusion: ranked lists combined by rank, `score(d) = sum_i 1 / (k + rank_i(d))`.

BM25 scores are unbounded and corpus-dependent and cosine similarities are in
[-1, 1], so normalizing one onto the other would assume something about their
distributions; ranks assume nothing. The constant `k` (60 in Cormack et al.
2009, full entry under References in evals/methodology.md, and in
configs/default.yaml) damps the top of each list so a single method cannot
dominate on its first hit alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import formatting


@dataclass(frozen=True)
class Fused:
    """One id of a fused list: its RRF score and the 1-based rank each method gave it."""

    chunk_id: str
    score: float
    ranks: dict[str, int]


def reciprocal_rank_fusion(ranked_lists: Mapping[str, Sequence[str]], k: int) -> list[Fused]:
    """Named ranked lists of ids fused into one, by descending score.

    Ties break on id so the output is deterministic: an ablation table whose rows
    shuffle between runs is not evidence of anything.

    Raises:
        ValueError: `k` is not positive.
    """
    if k <= 0:
        raise ValueError(f"RRF constant must be positive, got {formatting.count(k)}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for method, ids in ranked_lists.items():
        for position, doc_id in enumerate(ids, start=1):
            # A method that returns the same id twice must not count it twice.
            if method in ranks.get(doc_id, {}):
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(doc_id, {})[method] = position

    return sorted(
        (Fused(doc_id, score, ranks[doc_id]) for doc_id, score in scores.items()),
        key=lambda item: (-item.score, item.chunk_id),
    )
