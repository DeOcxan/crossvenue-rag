"""Paired tests for two retrieval methods, or two answer configurations, on the same questions.

The tests are paired by question because an unpaired test would count the
question-to-question variance, far larger than the effect compared, as noise between the two.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Hashable, Sequence
from dataclasses import dataclass
from math import comb
from typing import Literal

import numpy as np
from pydantic import ConfigDict

from .. import formatting, listing
from ..run_files import CsvRow
from ..stats import mean, percentile_bounds

# A paired comparison's family: the pre-registered primary one, a secondary one, or none
# (too few questions for a p-value).
Role = Literal["primary", "secondary", "descriptive"]


class PairedEstimate(CsvRow):
    """The fields every paired comparison row starts with.

    They are the pair, the metric and scope, the number of paired questions,
    both means and the bootstrap estimate of a - b with its interval.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    a: str
    b: str
    metric: str
    scope: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    low: float
    high: float


# Methodology section 7: a per-category subset (a scope defined by a category
# or slice) of fewer than 50 questions carries no p-value; its comparisons are
# descriptive, an interval and the per-question counts.
MIN_TESTED_QUESTIONS = 50


# How far below the observed mean a flipped mean may fall and still count as extreme,
# so a flip that reproduces the observed mean exactly counts despite summation order.
FLIP_TOLERANCE = 1e-12


def tested(scope_questions: int, *, whole_set: bool = False) -> bool:
    """Whether a comparison over a scope gets a p-value.

    `scope_questions` is the number of questions the scope is defined over (the
    category or slice), whatever number of them carries a value for the metric. A
    registered family over the 50 comparison questions keeps its p-values when the
    answers of both configurations have claims on only 17 of them (34 answers). A
    scope that is the whole question set is never a subset, so it is always tested.
    """
    return whole_set or scope_questions >= MIN_TESTED_QUESTIONS


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> float:
    """The two-sided exact McNemar p-value of paired binary outcomes (0/1).

    Only the discordant pairs carry information; under the null each is a fair
    coin, so the p-value is an exact binomial tail. With a few dozen questions the
    discordant counts are too small for the chi-square approximation.

    Raises:
        ValueError: `a` and `b` differ in length.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return mcnemar_p(only_a, only_a + only_b)


def mcnemar_p(k: int, discordant: int) -> float:
    """The exact two-sided McNemar p of a split: k of `discordant` questions favor one side.

    1.0 with no discordant question.

    Raises:
        ValueError: `k` is not between 0 and `discordant`.
    """
    if not 0 <= k <= discordant:
        raise ValueError(
            f"a split of {formatting.count(discordant)} discordant questions cannot favor one "
            f"side {formatting.count(k)}"
        )
    if discordant == 0:
        return 1.0
    outcomes: int = 2**discordant
    tail = sum(comb(discordant, i) for i in range(min(k, discordant - k) + 1)) / outcomes
    return min(1.0, 2 * tail)


@dataclass(slots=True)
class BootstrapDiff:
    """The mean of a - b with its bootstrap interval and the per-question counts."""

    mean_diff: float
    low: float
    high: float
    better: int  # questions where a is better than b, in the metric's direction
    worse: int
    ties: int


def sign_flip_p(samples: np.ndarray, permutations: int, rng: np.random.Generator) -> np.ndarray:
    """Two-sided sign-flip p for each row of `samples` (rows x questions).

    Each permutation flips the sign of every question's difference at random and
    counts, per row, whether the flipped mean is as far from 0 as the observed one.
    p = (1 + extreme) / (1 + permutations), so a p-value is never 0 and the observed
    arrangement counts as one permutation. The signs are drawn one permutation at a
    time, so one row draws the same signs as a single draw of `permutations` rows.
    """
    rows, n = samples.shape
    observed = np.abs(samples.mean(axis=1))
    extreme = np.zeros(rows)
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(rows, n))
        extreme += np.abs((samples * signs).mean(axis=1)) >= observed - FLIP_TOLERANCE
    result: np.ndarray = (1 + extreme) / (1 + permutations)
    return result


def paired_randomisation(
    a: Sequence[float], b: Sequence[float], permutations: int, seed: int
) -> float:
    """Two-sided p-value of the mean of a - b under random sign flips.

    Under the null that the two methods are exchangeable, each question's
    difference is equally likely to have either sign. Smucker, Allan & Carterette
    (2007; full entry under References in evals/methodology.md) use this test as the
    reference test for IR evaluation. It is `sign_flip_p` on the one row a - b.

    Raises:
        ValueError: `permutations` is below 1, or `a` and `b` differ in length.
    """
    if permutations < 1:
        raise ValueError(
            f"paired_randomisation needs permutations >= 1, got {formatting.count(permutations)}"
        )
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0 or not diff.any():
        return 1.0
    return float(sign_flip_p(diff[np.newaxis, :], permutations, np.random.default_rng(seed))[0])


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order.

    Step-down: the i-th smallest p is multiplied by (m - i + 1), and adjusted
    values are made monotone so a larger raw p never gets a smaller adjusted one.

    Raises:
        ValueError: A p-value is outside [0, 1] or not a number, which would poison the
            running maximum.
    """
    outside = [p for p in p_values if not 0 <= p <= 1]
    if outside:
        raise ValueError(
            f"p-values lie in [0, 1], got {listing.prose([formatting.metric(p) for p in outside])}"
        )
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int,
    seed: int,
    *,
    higher_is_better: bool = True,
) -> BootstrapDiff:
    """Mean of a - b over questions with a percentile bootstrap interval (`INTERVAL_LEVEL_PERCENT`).

    Questions are resampled with each question's pair kept together; a fixed
    `seed` makes the interval reproducible.

    `better` counts the questions where a is better than b: a > b when higher
    is better, a < b when lower is better; the interval is of a - b either way.

    Raises:
        ValueError: `resamples` is below 1, or `a` and `b` differ in length.
    """
    if resamples < 1:
        raise ValueError(
            f"paired_bootstrap needs resamples >= 1, got {formatting.count(resamples)}"
        )
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0:
        return BootstrapDiff(0.0, 0.0, 0.0, 0, 0, 0)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, diff.size, size=(resamples, diff.size))].mean(axis=1)
    low, high = percentile_bounds(means)
    return BootstrapDiff(
        float(diff.mean()),
        low,
        high,
        int((diff > 0).sum() if higher_is_better else (diff < 0).sum()),
        int((diff < 0).sum() if higher_is_better else (diff > 0).sum()),
        int((diff == 0).sum()),
    )


# The tests `compare_pair` can run on a pair.
PairTestName = Literal["randomisation", "mcnemar"]


@dataclass(frozen=True)
class PairTest:
    """One pair on the same questions: a against b, the means, the bootstrap and the p-values.

    A p-value is None when its test was not asked for.
    """

    n: int
    mean_a: float
    mean_b: float
    boot: BootstrapDiff
    randomisation_p: float | None
    mcnemar_p: float | None


def pair_tests(*, randomisation: bool, mcnemar: bool) -> tuple[PairTestName, ...]:
    """The tests `compare_pair` runs: the paired randomization test, the exact McNemar test."""
    chosen: list[PairTestName] = []
    if randomisation:
        chosen.append("randomisation")
    if mcnemar:
        chosen.append("mcnemar")
    return tuple(chosen)


def compare_pair(
    a: Sequence[float],
    b: Sequence[float],
    *,
    resamples: int,
    seed: int,
    higher_is_better: bool,
    tests: Collection[PairTestName],
) -> PairTest:
    """A against b on the same questions: the bootstrap of a - b, and the tests in `tests`.

    `randomisation` is the paired randomization test and `mcnemar` the exact McNemar
    test, which needs binary values.

    Raises:
        ValueError: `a` is empty, `a` and `b` differ in length, or `resamples` is below 1.
    """
    return PairTest(
        len(a),
        mean(a),
        mean(b),
        paired_bootstrap(a, b, resamples, seed, higher_is_better=higher_is_better),
        paired_randomisation(a, b, resamples, seed) if "randomisation" in tests else None,
        mcnemar_exact(a, b) if "mcnemar" in tests else None,
    )


def holm_by_family[T](
    rows: Sequence[T],
    family: Callable[[T], Hashable | None],
    p_value: Callable[[T], float | None],
) -> list[float | None]:
    """The Holm-adjusted p of each row within its family, in the rows' order.

    `family` names a row's family, None for a row in none; a row without a p-value joins
    no family either. Those rows get None.
    """
    members: dict[Hashable, list[int]] = {}
    for i, row in enumerate(rows):
        key = family(row)
        if key is not None and p_value(row) is not None:
            members.setdefault(key, []).append(i)
    adjusted: list[float | None] = [None] * len(rows)
    for indices in members.values():
        ps = [p for p in (p_value(rows[i]) for i in indices) if p is not None]
        for i, p in zip(indices, holm(ps), strict=True):
            adjusted[i] = p
    return adjusted
