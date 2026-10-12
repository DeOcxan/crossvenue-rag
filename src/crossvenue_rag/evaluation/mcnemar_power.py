"""The analytic power model of the exact McNemar test (methodology section 7).

Questions are discordant with probability psi, and a discordant question favors
the better method with probability 1/2 + delta/(2 psi); the power is the chance
the exact two-sided test (`paired.mcnemar_p`) rejects at `power.SIGNIFICANCE_LEVEL`.
"""

from __future__ import annotations

import functools
import math

from .. import formatting
from .paired import mcnemar_p
from .power import SIGNIFICANCE_LEVEL, TARGET_POWER

# Below this probability a number of discordant questions adds nothing to the power.
NEGLIGIBLE = 1e-12
# The grid a detectable difference is searched on.
DELTA_STEP = 0.01
# The decimals of a grid value: rounding keeps 0.07 from becoming 0.07000000000000001.
DELTA_DECIMALS = 2
# Where the search for a sample size starts, and its first upper bound.
FEWEST_QUESTIONS = 10
FIRST_BOUND = 1000


def _require_model(size: int, psi: float, delta: float) -> None:
    """The model's domain: at least one question, 0 < psi <= 1 and 0 <= delta <= psi.

    A delta above psi would make a discordant question favor the better method with
    a probability above 1.

    Raises:
        ValueError: An argument is outside that domain.
    """
    if size < 1:
        raise ValueError(
            f"the McNemar power model needs at least one question, got {formatting.count(size)}"
        )
    if not 0 < psi <= 1:
        raise ValueError(f"psi is a probability above 0, got {formatting.design_parameter(psi)}")
    if not 0 <= delta <= psi:
        raise ValueError(
            f"delta must lie in [0, psi] = [0, {formatting.design_parameter(psi)}], "
            f"got {formatting.design_parameter(delta)}"
        )


def binomial_pmf(k: int, trials: int, prob: float) -> float:
    """P(k successes in `trials`) for success probability `prob` in [0, 1]."""
    if prob in (0.0, 1.0):
        certain = 0 if prob == 0.0 else trials
        return 1.0 if k == certain else 0.0
    # In log space: the binomial coefficient of a few thousand trials exceeds the
    # largest float, and multiplying it by prob**k raises OverflowError.
    log_pmf = (
        math.lgamma(trials + 1)
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + k * math.log(prob)
        + (trials - k) * math.log1p(-prob)
    )
    return math.exp(log_pmf)


@functools.cache
def rejections(discordant: int) -> tuple[int, ...]:
    """Splits k of `discordant` questions that the exact two-sided McNemar test rejects.

    The p of a split grows with min(k, discordant - k), so the rejected splits are
    both tails up to the last one within `SIGNIFICANCE_LEVEL`. The tail is summed as integers,
    as `paired.mcnemar_p` sums it, so each p is the same float.
    """
    outcomes: int = 2**discordant
    tail, coefficient, last = 0, 1, -1
    for k in range(discordant // 2 + 1):
        tail += coefficient
        coefficient = coefficient * (discordant - k) // (k + 1)
        if min(1.0, 2 * (tail / outcomes)) > SIGNIFICANCE_LEVEL:
            break
        last = k
    return tuple(k for k in range(discordant + 1) if min(k, discordant - k) <= last)


@functools.cache
def mcnemar_power(size: int, psi: float, delta: float) -> float:
    """Power of the exact two-sided McNemar test on `size` questions, at `SIGNIFICANCE_LEVEL`.

    Raises:
        ValueError: `size`, `psi` or `delta` is outside the model (`_require_model`).
    """
    _require_model(size, psi, delta)
    favor = 0.5 + delta / (2 * psi)
    total = 0.0
    for d in range(1, size + 1):
        weight = binomial_pmf(d, size, psi)
        if weight < NEGLIGIBLE:
            continue
        total += weight * sum(binomial_pmf(k, d, favor) for k in rejections(d))
    return total


@functools.cache
def detectable_delta(size: int, psi: float) -> float:
    """The smallest difference on a grid of 0.01 that `size` questions detect; nan if none.

    Detected means with `TARGET_POWER`. The grid ends at psi, the largest difference
    the model allows.

    Raises:
        ValueError: `size` or `psi` is outside the model.
    """
    _require_model(size, psi, 0.0)
    delta = 0.0
    while round(delta + DELTA_STEP, DELTA_DECIMALS) <= psi:
        delta = round(delta + DELTA_STEP, DELTA_DECIMALS)
        if mcnemar_power(size, psi, delta) >= TARGET_POWER:
            return delta
    return math.nan


@functools.cache
def questions_needed(psi: float, delta: float) -> int:
    """A number of questions (from 10) that detects `delta` with `TARGET_POWER`.

    Found by bisection between 10 and the first power of two times 1,000 that has the
    power. The exact test's power is not monotonic in n, so a slightly smaller n can
    reach the target too; at the values the methodology states (psi 0.25, delta 0.10)
    none does.

    Raises:
        ValueError: `psi` or `delta` is outside the model, or `delta` is 0 (no sample
            detects it).
    """
    _require_model(FEWEST_QUESTIONS, psi, delta)
    if delta == 0:
        raise ValueError("no number of questions detects a delta of 0")
    low, high = FEWEST_QUESTIONS, FIRST_BOUND
    while mcnemar_power(high, psi, delta) < TARGET_POWER:
        high *= 2
    while low < high:
        mid = (low + high) // 2
        if mcnemar_power(mid, psi, delta) >= TARGET_POWER:
            high = mid
        else:
            low = mid + 1
    return low


def expected_split(size: int, psi: float, delta: float) -> tuple[int, int]:
    """(for the better method, for the other) of the expected discordant questions, rounded.

    Raises:
        ValueError: `size`, `psi` or `delta` is outside the model (`_require_model`).
    """
    _require_model(size, psi, delta)
    discordant = size * psi
    wins = discordant * (0.5 + delta / (2 * psi))
    return round(wins), round(discordant - wins)


def expected_split_p(size: int, psi: float, delta: float) -> float:
    """The exact two-sided McNemar p of the expected wins among `round(size * psi)` discordant.

    The wins are those of `expected_split`; the discordant total is rounded on its own,
    so it can differ by one from the sum of the split's two rounded parts. The arguments
    lie in the model's domain: at least one question, 0 < psi <= 1 and 0 <= delta <= psi.

    Raises:
        ValueError: `size`, `psi` or `delta` is outside the model (`expected_split`).
    """
    return mcnemar_p(expected_split(size, psi, delta)[0], round(size * psi))
