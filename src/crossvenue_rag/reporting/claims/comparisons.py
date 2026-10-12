"""Formatted values that several claim modules state.

The comparisons the documents quote with their statistics, the annotator's kappa with its
interval, the sizes of the question sets and of the sensitivity runs, and the venues' fee tiers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ... import formatting, listing
from ...config_eval import COMPARISON_SCOPE, LABELLED_SCOPE, MULTI_VENUE_SCOPE
from ...dataset.agreement import AgreementVariant
from ...evaluation.paired import MIN_TESTED_QUESTIONS
from ...evaluation.power import SIGNIFICANCE_LEVEL
from ...run_files import file_sha256
from ...stats import INTERVAL_LEVEL_PERCENT
from ..compute import answers, corpus, datasets, datasets_rounds, retrieval, sensitivity
from ..compute.corpus_pancakeswap import FEE_TIERS, fee_tier_text
from ..compute.estimates import Estimate
from ..run_ids import SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from ..sources import Sources

# The retrieval comparisons the documents quote, a - b.
SPLIT = ("venue_split", "hybrid_rerank")
RERANKER = ("hybrid_rerank", "hybrid")
ORACLE = ("venue_split_oracle", "venue_split")


def estimate(difference: Estimate) -> str:
    """A difference with its interval: `+0.280 [+0.160, +0.400]`."""
    interval = formatting.interval(difference.low, difference.high)
    return f"{formatting.signed(difference.value)} {interval}"


def signed_parts(difference: Estimate) -> list[str]:
    """A difference and its bounds, each signed, for the three holes of a claim."""
    return [formatting.signed(v) for v in (difference.value, difference.low, difference.high)]


def signed_ms_parts(difference: Estimate) -> list[str]:
    """A difference in ms and its bounds, each signed, for the three holes of a claim."""
    return [formatting.signed_ms(v) for v in (difference.value, difference.low, difference.high)]


def level() -> str:
    """The level of the paired intervals, in percent: `95`."""
    return formatting.count(INTERVAL_LEVEL_PERCENT)


def kappa_bounds(reading: AgreementVariant) -> tuple[str, str]:
    """The bounds of a reading's kappa interval, formatted.

    Raises:
        ValueError: the reading has no interval.
    """
    ci = reading.interval
    if ci is None or ci.low is None or ci.high is None:
        raise ValueError(f"the {reading.reference} {reading.scope} reading has no kappa interval")
    return formatting.metric(ci.low), formatting.metric(ci.high)


def kappa_cell(reading: AgreementVariant) -> str:
    """A reading's kappa with its interval: `0.470 [0.317, 0.645]`.

    The kappa is given without an interval when no resample defined one.
    """
    kappa, ci = reading.report.kappa, reading.interval
    if ci is None or ci.low is None or ci.high is None:
        return formatting.metric(kappa)
    return f"{formatting.metric(kappa)} {formatting.bounds(ci.low, ci.high)}"


def significance() -> str:
    """The significance level of every comparison and registered test: `0.05`."""
    return formatting.design_parameter(SIGNIFICANCE_LEVEL)


def p_bound() -> str:
    """The bound a document writes for a p-value below it: `0.001`."""
    return formatting.fixed(formatting.SMALLEST_SHOWN_P, formatting.P_VALUE_DECIMALS)


def min_tested() -> str:
    """The fewest questions a comparison is tested on."""
    return formatting.count(MIN_TESTED_QUESTIONS)


def top_k_text(src: Sources) -> str:
    """The retrieval run's deepest cutoff, `retrieval.top_k`."""
    return formatting.count(retrieval.top_k(src))


def comparison_questions(src: Sources) -> str:
    """Comparison questions of the gold set."""
    return formatting.count(len(datasets.comparison_ids(src)))


def question_count_text(src: Sources) -> str:
    """Questions of the gold set."""
    return formatting.count(datasets.question_count(src))


def round2_sample_text(src: Sources) -> str:
    """Questions of the round-2 re-judging sample, counted only."""
    return formatting.count(len(datasets.round2_sample(src)))


def sensitivity_sizes(src: Sources) -> list[str]:
    """The comparison and multi-venue questions of the sensitivity runs, then their labeled ones.

    The labeled questions are those of `v0.2.2-gold`, then those of `v0.2.1-gold`.

    Raises:
        ValueError: The two runs pair different comparison or multi-venue questions.
    """
    scopes = (COMPARISON_SCOPE, MULTI_VENUE_SCOPE)
    shared, base = (
        [sensitivity.tested_questions(src, run, s) for s in scopes]
        for run in (SENSITIVITY_RUN, SENSITIVITY_BASE_RUN)
    )
    if shared != base:
        raise ValueError("the sensitivity runs pair different comparison or multi-venue questions")
    labeled = [
        sensitivity.tested_questions(src, run, LABELLED_SCOPE)
        for run in (SENSITIVITY_RUN, SENSITIVITY_BASE_RUN)
    ]
    return [formatting.count(n) for n in (*shared, *labeled)]


def better_worse(split: retrieval.QuestionSplit) -> list[str]:
    """The questions a comparison is better and worse on, for the two holes of a claim."""
    return [formatting.count(split.better), formatting.count(split.worse)]


def fee_tier_texts() -> dict[str, str]:
    """Each venue's fee tier (`corpus_pancakeswap.FEE_TIERS`) as the documents write it: `0.30`."""
    return {v: fee_tier_text(t) for v, t in FEE_TIERS.items()}


def change_text[T](before: T, after: T, fmt: Callable[[T], str]) -> str:
    """A value under prompt v1 and under v2, each through `fmt`: `0.120 to 0.040`."""
    return f"{fmt(before)} to {fmt(after)}"


def duplicate_share_text(src: Sources) -> str:
    """The percent of chunk occurrences in a group of byte-identical ones (`corpus.duplicates`)."""
    dup = corpus.duplicates(src)
    return formatting.percent_figure(dup.in_groups, dup.occurrences)


def optional_count(n: int | None) -> str:
    """A count (`formatting.count`), or `formatting.MISSING` when there is none."""
    return formatting.optional(n, lambda v: formatting.count(int(v)))


def file_short_hash(src: Sources, relative: str | Path) -> str:
    """The short sha256 of the file at `relative` under the root (`formatting.short_hash`)."""
    return formatting.short_hash(file_sha256(src.path(relative)))


def round2_category_text(src: Sources, category: str) -> str:
    """Questions of `category` in the round-2 re-judging sample, counted only."""
    return formatting.count(datasets_rounds.sample_categories(src)[category])


def retried_status_parts() -> tuple[str, list[str]]:
    """The statuses the model client retries, for a claim: their holes as prose, and the texts."""
    statuses = [formatting.own_form(status) for status in answers.retried_statuses()]
    return listing.prose(["{}"] * len(statuses)), statuses


def coverage_parts(src: Sources, method: str) -> list[str]:
    """`method`'s comparison coverage and its counts, for the holes of `0.52 (26/50)`."""
    share = formatting.fixed(retrieval.covered_share(src, method), formatting.SHARE_DECIMALS)
    return [share, *[formatting.count(v) for v in retrieval.coverage_counts(src, method)]]
