"""The retrieval results' tables: per method and per comparison, as findings.md states them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ... import formatting
from ...config_eval import COMPARISON_SCOPE, LABELLED_COMPARISON_SCOPE
from ...evaluation.results import PER_QUERY_FILE, breakdown_file
from ...formatting import METRIC_DECIMALS
from ...models import COMPARISON, RETRIEVAL_STEPS
from ...run_files import SUMMARY_FILE
from ..sources import Sources
from .estimates import Estimate
from .lookup import entry
from .retrieval import (
    BASELINE,
    FIRST_STAGE,
    METHODS,
    at_top_k,
    category_metric,
    coverage_counts,
    covers_every_venue,
    paired_interval,
    paired_p,
    recall_against,
    step_p50,
)

# The largest gain of nDCG'@10 over nDCG@10 the documents allow any method.
NDCG_PRIME_GAIN_BOUND = 0.01


@dataclass(frozen=True)
class ComparisonCategoryRow:
    """One method on the comparison questions, at the run's top k."""

    method: str
    recall: float
    ndcg: float
    ndcg_prime: float
    coverage: float  # venue_coverage
    covered: int  # questions with every named venue in the top k
    questions: int


def comparison_rows(src: Sources) -> list[ComparisonCategoryRow]:
    """Per method, its row on the comparison questions."""
    rows: list[ComparisonCategoryRow] = []
    for m in METHODS:
        covered, questions = coverage_counts(src, m)
        ndcg_prime = entry(src.ndcg_prime.by_category, (COMPARISON, m), breakdown_file("category"))
        rows.append(
            ComparisonCategoryRow(
                m,
                category_metric(src, COMPARISON, m, at_top_k(src, "recall")),
                category_metric(src, COMPARISON, m, at_top_k(src, "ndcg")),
                ndcg_prime.value(at_top_k(src, "ndcg_prime")),
                category_metric(src, COMPARISON, m, at_top_k(src, "venue_coverage")),
                covered,
                questions,
            )
        )
    return rows


@dataclass(frozen=True)
class RerankerRow:
    """hybrid_rerank - `b` on recall and on venue_coverage, at the run's top k."""

    b: str
    recall: Estimate
    recall_holm: float
    coverage: Estimate
    coverage_mcnemar: float
    coverage_holm: float


def reranker_rows(src: Sources) -> list[RerankerRow]:
    """hybrid_rerank against each method of `FIRST_STAGE`, in reverse order."""
    rows: list[RerankerRow] = []
    recall_k, coverage_k = at_top_k(src, "recall"), at_top_k(src, "venue_coverage")
    for b in reversed(FIRST_STAGE):
        cell = (BASELINE, b)
        rows.append(
            RerankerRow(
                b,
                paired_interval(src, *cell, recall_k, LABELLED_COMPARISON_SCOPE),
                paired_p(src, *cell, recall_k, LABELLED_COMPARISON_SCOPE, "holm_p"),
                paired_interval(src, *cell, coverage_k, COMPARISON_SCOPE),
                paired_p(src, *cell, coverage_k, COMPARISON_SCOPE, "mcnemar_p"),
                paired_p(src, *cell, coverage_k, COMPARISON_SCOPE, "holm_p"),
            )
        )
    return rows


@dataclass(frozen=True)
class OracleGains:
    """Questions naming two or more venues that the oracle covers and venue_split does not."""

    questions: int
    naming_none: int  # of them, naming no venue in the text
    naming_one: int  # of them, naming one venue


def oracle_gains(src: Sources) -> OracleGains:
    """The questions the oracle covers and venue_split does not, at the run's top k."""
    covered = {
        (r.method, r.question_id): covers_every_venue(src, r) for r in src.retrieval.per_query
    }
    gained = [
        q
        for q, rec in src.questions.items()
        if rec.is_multi_venue
        and entry(covered, ("venue_split_oracle", q), PER_QUERY_FILE)
        and not entry(covered, ("venue_split", q), PER_QUERY_FILE)
    ]
    named = Counter(len(src.detector.detect(src.questions[q].question)) for q in gained)
    return OracleGains(len(gained), named[0], named[1])


def recall_gap(src: Sources, *, augmented: bool) -> float:
    """venue_split - hybrid_rerank on comparison recall@10, rescored against a gold set.

    The augmented gold set (`src.questions`) or the one before augmentation.
    """
    gold = src.questions if augmented else src.questions_before_augmentation
    return recall_against(src, "venue_split", gold) - recall_against(src, "hybrid_rerank", gold)


def rescored_gap_agrees(src: Sources) -> bool:
    """Whether the gap rescored against the augmented gold is the p-value run's, as written.

    The two are compared as the document writes a difference (`formatting.signed`).
    """
    diff = paired_interval(
        src, "venue_split", "hybrid_rerank", at_top_k(src, "recall"), LABELLED_COMPARISON_SCOPE
    ).value
    return formatting.signed(recall_gap(src, augmented=True)) == formatting.signed(diff)


@dataclass(frozen=True)
class CondensedNdcg:
    """One method's nDCG'@10 against its nDCG@10 on the labeled questions."""

    method: str
    difference: float  # nDCG'@10 - nDCG@10, of the values at `METRIC_DECIMALS`
    plain: float
    condensed: float


def condensed_ndcg(src: Sources) -> list[CondensedNdcg]:
    """Per method on the labeled questions: nDCG'@10 against nDCG@10."""
    rows: list[CondensedNdcg] = []
    for m in METHODS:
        row = entry(src.ndcg_prime.summary, m, SUMMARY_FILE)
        plain = row.value(at_top_k(src, "ndcg"))
        condensed = row.value(at_top_k(src, "ndcg_prime"))
        difference = round(condensed, METRIC_DECIMALS) - round(plain, METRIC_DECIMALS)
        rows.append(CondensedNdcg(m, difference, plain, condensed))
    return rows


def condensed_gains_below_bound(src: Sources) -> bool:
    """Whether every method's nDCG'@10 - nDCG@10 is below `NDCG_PRIME_GAIN_BOUND`."""
    return all(r.condensed - r.plain < NDCG_PRIME_GAIN_BOUND for r in condensed_ndcg(src))


def latency_rows(src: Sources) -> list[tuple[str, list[float]]]:
    """Per method: the p50 ms of each retrieval step, then p50 and p95 ms of the whole.

    Each step's p50 is the value the run's own table gives (`retrieval.step_p50`), and the
    document rounds that value once, where it shows it.
    """
    rows: list[tuple[str, list[float]]] = []
    for m in METHODS:
        summary = entry(src.retrieval.summary, m, SUMMARY_FILE)
        values = [step_p50(src, m, s) for s in RETRIEVAL_STEPS]
        rows.append((m, [*values, summary.p50_ms, summary.p95_ms]))
    return rows


def labelled_sizes(src: Sources, categories: Sequence[str]) -> list[int]:
    """Labeled questions of each of `categories` in turn."""
    by_category = src.retrieval.by_category
    return [
        entry(by_category, (c, "bm25"), breakdown_file("category")).labelled_questions
        for c in categories
    ]


def labelled_size_range(src: Sources, categories: Sequence[str]) -> tuple[int, int]:
    """(fewest, most) `labelled_sizes` over `categories`; ValueError if there are none."""
    sizes = labelled_sizes(src, categories)
    return min(sizes), max(sizes)
