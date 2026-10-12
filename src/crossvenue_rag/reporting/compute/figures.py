"""The values the findings figures draw.

scripts/figures.py draws these and checks each mean against the number its run reports
before drawing it; the bootstrap intervals, which no run reports, are drawn unchecked.
"""

from __future__ import annotations

from ...evaluation.metrics import off_venue_at_k
from ...evaluation.paired import paired_bootstrap
from ...evaluation.results import PerQueryRow
from ...stats import mean
from ..sources import Sources
from . import retrieval
from .estimates import Estimate
from .lookup import entry
from .retrieval import METHODS, RankMetric, chunk_venues, top_k


def category_per_query(src: Sources, category: str) -> dict[str, list[PerQueryRow]]:
    """The retrieval run's per-query rows of the questions of `category`, by method."""
    rows: dict[str, list[PerQueryRow]] = {m: [] for m in METHODS}
    for r in src.retrieval.per_query:
        if entry(src.questions, r.question_id, "the gold set").effective_category == category:
            entry(rows, r.method, "METHODS").append(r)
    return rows


def bootstrap_mean_interval(values: list[float], src: Sources) -> Estimate:
    """The mean of `values` with its bootstrap interval.

    With the retrieval run's resamples and seed; the interval is the paired
    bootstrap's against a zero baseline, so it is the interval of the mean.
    """
    cfg = src.retrieval.config.eval
    return Estimate.of_bootstrap(
        paired_bootstrap(values, [0.0] * len(values), cfg.bootstrap_resamples, cfg.bootstrap_seed)
    )


def labelled_values(src: Sources, rows: list[PerQueryRow], metric: RankMetric) -> list[float]:
    """`metric` at the run's top k for each of `rows` whose question has gold chunks.

    `retrieval.labelled_scores` against the gold set.
    """
    return retrieval.labelled_scores(src, rows, metric, src.questions)


def off_venue_shares(src: Sources, rows: list[PerQueryRow]) -> tuple[float, float]:
    """(off-venue share, via-ccxt share) of the top k, averaged over `rows`.

    From each row's venues (`PerQueryRow.retrieval`), scored with `off_venue_at_k`.
    """
    k, venue_of = top_k(src), chunk_venues(src)
    shares = [
        off_venue_at_k(
            r.retrieval(venue_of).venues,
            entry(src.questions, r.question_id, "the gold set").venues,
            k,
        )
        for r in rows
    ]
    return mean([s[0] for s in shares]), mean([s[1] for s in shares])
