"""Retrieval runs made on an earlier corpus or gold set than the documents report.

The run made before exact deduplication (`run_ids.PRE_DEDUP_RUN`), whose top 10s
held copies of one text, and the first run on `v0.2-gold`
(`run_ids.FIRST_GOLD_RUN`), whose venue coverage motivated venue_split (ADR
0007). Both are read at the retrieval run's deepest cutoff (`top_k`); the run that measured
the coverage failure (`run_ids.COVERAGE_FAILURE_RUN`) at the top 10.
"""

from __future__ import annotations

from ... import formatting
from ...config_eval import LABELLED_COMPARISON_SCOPE
from ...evaluation.metrics import venue_coverage_at_k
from ...evaluation.results import PER_QUERY_FILE
from ...run_files import PAIRED_FILE
from ...stats import mean
from ..run_ids import (
    COVERAGE_FAILURE_RUN,
    FIRST_GOLD_RUN,
    PRE_DEDUP_RUN,
)
from ..sources import Sources
from . import datasets
from .corpus_changes import dedup_idmap
from .estimates import Estimate
from .lookup import entry
from .retrieval import ROUND1_POOL, at_top_k, top_k


def copy_slots(src: Sources) -> dict[str, list[int]]:
    """Per method, the slots of each question's top k held by a copy of a text ranked above it.

    In the run made before exact deduplication: two of its chunk ids are copies when
    the id map maps them to the same chunk.
    """
    idmap = dedup_idmap(src)
    k = top_k(src)
    slots: dict[str, list[int]] = {}
    for row in src.run(PRE_DEDUP_RUN).per_query:
        seen: set[str] = set()
        copies = 0
        for old in row.ids[:k]:
            new = idmap.get(old, old)
            copies += new in seen
            seen.add(new)
        slots.setdefault(row.method, []).append(copies)
    return slots


def copy_slot_questions(src: Sources) -> int:
    """Questions per method in the run made before deduplication.

    Raises:
        ValueError: the run has no row.
    """
    slots = next(iter(copy_slots(src).values()), None)
    if slots is None:
        raise ValueError(f"{PRE_DEDUP_RUN} has no row in {PER_QUERY_FILE}")
    return len(slots)


def _copy_slot_means(src: Sources) -> dict[str, float]:
    return {m: mean(v) for m, v in copy_slots(src).items()}


def copy_slot_mean_range(src: Sources) -> tuple[float, float]:
    """(smallest, largest) mean of copy slots per question over the methods."""
    means = _copy_slot_means(src).values()
    return min(means), max(means)


def copy_slot_means(src: Sources) -> list[float]:
    """The mean copy slots per question of each round-1 pool method, in `ROUND1_POOL` order."""
    means = _copy_slot_means(src)
    return [entry(means, m, f"{PRE_DEDUP_RUN}'s {PER_QUERY_FILE}") for m in ROUND1_POOL]


def copy_question_range(src: Sources) -> tuple[int, int]:
    """(fewest, most) questions with at least one copy slot, over the methods."""
    affected = [sum(x > 0 for x in v) for v in copy_slots(src).values()]
    return min(affected), max(affected)


def copy_slot_most(src: Sources) -> int:
    """The most copy slots in one question's top k, over every method."""
    return max(max(v) for v in copy_slots(src).values())


def first_gold_coverage(src: Sources, method: str) -> float:
    """Mean venue_coverage@k of `method` on `v0.2-gold`'s comparison questions.

    Recomputed from the first run on that set, with each retrieved chunk's venue as the
    row saved it, or else from the corpus (`PerQueryRow.retrieval`).
    """
    questions = src.questions_before_augmentation
    k = top_k(src)
    rows = [
        (r, entry(questions, r.question_id, "v0.2-gold"))
        for r in src.run(FIRST_GOLD_RUN).per_query
        if r.method == method
    ]
    values = [
        venue_coverage_at_k(r.retrieval(src.venue_of).venues, q.venues, k)
        for r, q in rows
        if q.is_comparison
    ]
    return mean(values)


def first_gold_covered(src: Sources, method: str) -> int:
    """`v0.2-gold` comparison questions with every named venue in `method`'s top k."""
    total = datasets.comparison_count_before_augmentation(src)
    return round(first_gold_coverage(src, method) * total)


def first_gold_covered_most(src: Sources) -> int:
    """The most `first_gold_covered` of any round-1 pool method."""
    return max(first_gold_covered(src, m) for m in ROUND1_POOL)


def first_gold_coverage_equal(src: Sources, a: str, b: str) -> bool:
    """Whether a and b cover as many questions, their coverage equal as a share is written."""

    def written(method: str) -> str:
        return formatting.fixed(first_gold_coverage(src, method), formatting.SHARE_DECIMALS)

    return written(a) == written(b) and first_gold_covered(src, a) == first_gold_covered(src, b)


def first_run_unjudged(src: Sources, method: str) -> float:
    """Mean share of `method`'s top 10 on the comparison questions that round 1 never judged.

    In the run that measured the coverage failure, against the `v0.2-gold`
    journal; the run predates `unjudged@k`, so it is recomputed from its saved
    retrievals.
    """
    rounds = src.rounds_before_augmentation
    before = src.questions_before_augmentation
    shares = []
    for row in src.run(COVERAGE_FAILURE_RUN).per_query:
        q = row.question_id
        first = rounds.first(q)
        if row.method != method or not before[q].is_comparison or first is None:
            continue
        seen = set(first.judged_ids())
        top = row.ids[: top_k(src)]
        shares.append(mean([c not in seen for c in top]))
    return mean(shares)


def earlier_gold_gap(src: Sources) -> Estimate:
    """venue_split - hybrid_rerank on comparison recall at the top k, in the run on `v0.2-gold`.

    The paired row of `COVERAGE_FAILURE_RUN`, with its interval.
    """
    key = ("venue_split", "hybrid_rerank", at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE)
    return Estimate.of(entry(src.run(COVERAGE_FAILURE_RUN).paired, key, PAIRED_FILE))


def earlier_gold_run_matches(src: Sources) -> bool:
    """Whether the run on `v0.2-gold` holds the retrieval run's top k, question by method."""
    earlier = {(r.method, r.question_id): r.ids for r in src.run(COVERAGE_FAILURE_RUN).per_query}
    return earlier == {(r.method, r.question_id): r.ids for r in src.retrieval.per_query}
