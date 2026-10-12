"""Statistics more than one layer computes: means, percentiles, bootstrap intervals, Cohen's kappa.

They are at the package root so corpus statistics, the evaluation harness, the
dataset tools and answer generation compute each one the same way without
importing one another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from . import formatting

# The percentiles every run and report states: the median and the 95th.
P50 = 0.50
P95 = 0.95

# The level of every bootstrap interval, in percent, and the percentiles that bound it.
INTERVAL_LEVEL_PERCENT = 95
INTERVAL_PERCENTILES = (
    (formatting.PER_CENT - INTERVAL_LEVEL_PERCENT) / 2,
    (formatting.PER_CENT + INTERVAL_LEVEL_PERCENT) / 2,
)


def mean(values: Sequence[float]) -> float:
    """The arithmetic mean of `values`.

    A caller for whom no values is a legitimate input says what it returns then.

    Raises:
        ValueError: There are no values.
    """
    if not values:
        raise ValueError("the mean of no values is undefined")
    return sum(values) / len(values)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of `values` at `fraction` (0.95 for p95).

    The smallest value at or above `fraction` of the sorted sample, so `p95` of
    four samples is the largest of them. It is not interpolated: latency
    samples here are few, and an interpolated p95 over eight queries is a number
    nobody measured. A caller for whom no values is a legitimate input says what
    it reports then, as for `mean`.

    Raises:
        ValueError: There are no values, or `fraction` is outside [0, 1].
    """
    if not values:
        raise ValueError("the percentile of no values is undefined")
    if not 0 <= fraction <= 1:
        raise ValueError(f"a percentile fraction lies in [0, 1], got {formatting.metric(fraction)}")
    ordered = sorted(values)
    rank = ceil(fraction * len(ordered))
    return ordered[min(len(ordered) - 1, max(0, rank - 1))]


def mean_of_present(values: Sequence[float | None]) -> float | None:
    """The mean of the values that are not None; None when no value is present."""
    present = [v for v in values if v is not None]
    return mean(present) if present else None


@dataclass(frozen=True)
class Kappa:
    """Cohen's kappa with the observed and the chance-expected agreement it is computed from."""

    kappa: float
    observed: float
    expected: float


def cohens_kappa(first: Sequence[bool], second: Sequence[bool]) -> Kappa:
    """Cohen's kappa for two binary label sequences.

    Kappa is undefined when both raters used one and the same category for
    everything; observed and expected agreement are then both 1.0, and kappa is
    reported as 1.0 rather than dividing by zero.

    Raises:
        ValueError: The sequences differ in length or are empty.
    """
    if len(first) != len(second):
        raise ValueError(
            f"{formatting.count(len(first))} labels vs {formatting.count(len(second))}"
        )
    n = len(first)
    if n == 0:
        raise ValueError("no overlapping judgments to compare")

    agree = sum(1 for a, b in zip(first, second, strict=True) if a == b)
    observed = agree / n

    p_first = sum(first) / n
    p_second = sum(second) / n
    expected = chance_agreement(p_first, p_second)

    if expected >= 1.0:
        # Both raters used one and the same label throughout, so they agreed.
        return Kappa(1.0, observed, expected)
    return Kappa(kappa_value(observed, expected), observed, expected)


def chance_agreement[N: (float, NDArray[np.float64])](p_first: N, p_second: N) -> N:
    """Kappa's expected agreement from each rater's share of positive labels, element-wise."""
    return p_first * p_second + (1 - p_first) * (1 - p_second)


def kappa_value[N: (float, NDArray[np.float64])](observed: N, expected: N) -> N:
    """Kappa from observed and expected agreement, element-wise.

    Raises:
        ValueError: An expected agreement is 1 or more, where kappa is undefined.
    """
    if np.any(np.asarray(expected) >= 1.0):
        raise ValueError("kappa is undefined where the expected agreement is 1")
    return (observed - expected) / (1 - expected)


def percentile_bounds(values: Sequence[float] | NDArray[np.float64]) -> tuple[float, float]:
    """The bounds of a percentile bootstrap interval at `INTERVAL_LEVEL_PERCENT`.

    Raises:
        ValueError: There are no values.
    """
    if len(values) == 0:
        raise ValueError("the percentile bounds of no values are undefined")
    low, high = np.percentile(values, INTERVAL_PERCENTILES)
    return float(low), float(high)


@dataclass(slots=True)
class Interval:
    """A percentile bootstrap interval; its bounds are None when no resample defined it."""

    low: float | None
    high: float | None
    undefined: int  # resamples in which the statistic was undefined
    resamples: int

    @classmethod
    def from_draws(
        cls, values: Sequence[float] | NDArray[np.float64], undefined: int, resamples: int
    ) -> Interval:
        """The interval of the statistic's values over the resamples that defined it."""
        if len(values) == 0:
            return cls(None, None, undefined, resamples)
        return cls(*percentile_bounds(values), undefined, resamples)


def cohens_kappa_or_none(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa of paired binary verdicts, None where it is undefined.

    Undefined means no pairs, or one and the same category used throughout, the
    case `cohens_kappa` reports as 1.0.
    """
    if not pairs:
        return None
    result = cohens_kappa([a for a, _ in pairs], [b for _, b in pairs])
    return None if result.expected >= 1.0 else result.kappa
