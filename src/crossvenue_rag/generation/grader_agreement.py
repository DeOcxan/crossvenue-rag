"""Agreement between two graders of the same answers: per dimension, with intervals.

Two gradings (a judge against the human, against another judge, or against
itself) are paired by answer; binary dimensions are compared by agreement and
Cohen's kappa, coverage by correlation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..config_generation import CalibrationDimension
from ..stats import Interval, cohens_kappa_or_none, mean
from .metrics import answer_metrics
from .schemas import Judgement

# The fewest paired values a correlation is defined on.
CORRELATION_POINTS = 2

# The binary dimensions of a judgment two graders are compared on: every
# `CalibrationDimension` but coverage.
BINARY_DIMENSIONS: tuple[CalibrationDimension, ...] = (
    "supported",
    "venue_correct",
    "behaviour_correct",
)


@dataclass(slots=True)
class DimensionAgreement:
    """How two graders agree on one dimension over `n` paired verdicts.

    A statistic that does not apply to the dimension, or is undefined on the
    verdicts, is None.
    """

    dimension: str
    n: int
    agreement: float | None  # binary: share of identical verdicts
    kappa: float | None  # binary: Cohen's kappa
    pearson: float | None  # coverage
    spearman: float | None  # coverage


def _rank(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = np.asarray(values)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def _pearson(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float | None:
    if len(x) < CORRELATION_POINTS or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def binary_agreement(dimension: str, a: Sequence[bool], b: Sequence[bool]) -> DimensionAgreement:
    """Share of identical verdicts and Cohen's kappa of the paired verdicts `a` and `b`.

    Raises:
        ValueError: `a` is not empty and differs from `b` in length.
    """
    if not a:
        return DimensionAgreement(dimension, 0, None, None, None, None)
    agreed = mean([x == y for x, y in zip(a, b, strict=True)])
    # Undefined, rather than 1.0, when both graders use one category throughout.
    kappa = cohens_kappa_or_none(list(zip(a, b, strict=True)))
    return DimensionAgreement(dimension, len(a), agreed, kappa, None, None)


def coverage_agreement(a: Sequence[float], b: Sequence[float]) -> DimensionAgreement:
    """Pearson and Spearman correlation of the paired coverage values `a` and `b`."""
    pearson = _pearson(a, b)
    spearman = _pearson(_rank(a), _rank(b)) if len(a) >= CORRELATION_POINTS else None
    return DimensionAgreement("coverage", len(a), None, None, pearson, spearman)


@dataclass(slots=True)
class PairedVerdicts:
    """One answer graded twice: the paired verdicts it contributes."""

    answer_id: str
    supported: list[tuple[bool, bool]]
    venue_correct: list[tuple[bool, bool]]
    behaviour_correct: tuple[bool, bool] | None  # None: excluded from this dimension
    coverage: tuple[float, float] | None


def paired_verdicts(
    first: Mapping[str, Judgement],
    second: Mapping[str, Judgement],
    exclude_behaviour: Collection[str] = (),
) -> list[PairedVerdicts]:
    """Per answer graded in both, the verdicts to compare.

    Claims are matched by index; coverage is compared where both graders give
    one. Answers in `exclude_behaviour` contribute no `behaviour_correct` pair.
    """
    out = []
    for answer_id in sorted(set(first) & set(second)):
        x, y = first[answer_id], second[answer_id]
        claims_x = {c.index: c for c in x.claims}
        claims_y = {c.index: c for c in y.claims}
        shared = sorted(set(claims_x) & set(claims_y))
        mx, my = answer_metrics(x).coverage, answer_metrics(y).coverage
        out.append(
            PairedVerdicts(
                answer_id=answer_id,
                supported=[(claims_x[i].supported, claims_y[i].supported) for i in shared],
                venue_correct=[
                    (claims_x[i].venue_correct, claims_y[i].venue_correct) for i in shared
                ],
                behaviour_correct=None
                if answer_id in exclude_behaviour
                else (x.behaviour_correct, y.behaviour_correct),
                coverage=(mx, my) if mx is not None and my is not None else None,
            )
        )
    return out


def dimension_verdicts(pairs: Sequence[PairedVerdicts], dimension: str) -> list[tuple[bool, bool]]:
    """The paired verdicts of one binary dimension, (first grader, second grader) each.

    There is one per answer for `behaviour_correct` and one per claim for the
    claim dimensions.

    Raises:
        ValueError: `dimension` is not one of `BINARY_DIMENSIONS`.
    """
    if dimension not in BINARY_DIMENSIONS:
        raise ValueError(f"{dimension} is not a binary dimension: {', '.join(BINARY_DIMENSIONS)}")
    if dimension == "behaviour_correct":
        return [p.behaviour_correct for p in pairs if p.behaviour_correct is not None]
    return [v for p in pairs for v in getattr(p, dimension)]


def agreement_from_pairs(pairs: Sequence[PairedVerdicts]) -> list[DimensionAgreement]:
    """Agreement per dimension over `pairs`: the binary dimensions, then coverage."""
    out = []
    for dimension in BINARY_DIMENSIONS:
        verdicts = dimension_verdicts(pairs, dimension)
        out.append(binary_agreement(dimension, [a for a, _ in verdicts], [b for _, b in verdicts]))
    cov = [p.coverage for p in pairs if p.coverage is not None]
    out.append(coverage_agreement([a for a, _ in cov], [b for _, b in cov]))
    return out


def compare_gradings(
    first: Mapping[str, Judgement],
    second: Mapping[str, Judgement],
    exclude_behaviour: Collection[str] = (),
) -> list[DimensionAgreement]:
    """Agreement over answers graded in both, dimension by dimension.

    The graders are the judge against a human, or the judge against itself.
    Each binary dimension gets percent agreement and Cohen's kappa, coverage
    gets Pearson and Spearman correlation. Keys are answer ids; values are
    judgments in the judge's schema (a human rating uses the same schema).
    Answers in `exclude_behaviour` are left out of `behaviour_correct` only, as
    `calibration.agreement_record` leaves them out (`paired_verdicts`).
    """
    return agreement_from_pairs(paired_verdicts(first, second, exclude_behaviour))


def agreement_intervals(
    pairs: Sequence[PairedVerdicts], resamples: int, seed: int
) -> dict[str, dict[str, Interval]]:
    """Percentile bootstrap intervals by dimension and statistic, resampling answers.

    The level is `stats.INTERVAL_LEVEL_PERCENT`, as for every interval.

    A claim's verdicts are resampled with its answer. The binary dimensions get
    agreement and kappa, coverage gets Spearman.
    """
    rng = np.random.default_rng(seed)
    draws: dict[tuple[str, str], list[float]] = {}
    undefined: dict[tuple[str, str], int] = {}
    for _ in range(resamples):
        sample = [pairs[i] for i in rng.integers(0, len(pairs), size=len(pairs))]
        for key, value in _statistics(sample):
            if value is None:
                undefined[key] = undefined.get(key, 0) + 1
            else:
                draws.setdefault(key, []).append(value)
    out: dict[str, dict[str, Interval]] = {}
    for key in set(draws) | set(undefined):
        dimension, statistic = key
        out.setdefault(dimension, {})[statistic] = Interval.from_draws(
            draws.get(key, []), undefined.get(key, 0), resamples
        )
    return out


def _statistics(sample: Sequence[PairedVerdicts]) -> list[tuple[tuple[str, str], float | None]]:
    """Every statistic of one resample (`agreement_from_pairs`), `None` where it is undefined.

    The binary dimensions give agreement and kappa, coverage gives Spearman, in that order.
    """
    out: list[tuple[tuple[str, str], float | None]] = []
    for d in agreement_from_pairs(sample):
        if d.dimension == "coverage":
            out.append((("coverage", "spearman"), d.spearman))
        else:
            out += [((d.dimension, "agreement"), d.agreement), ((d.dimension, "kappa"), d.kappa)]
    return out
