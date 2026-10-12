"""The paired t-test sample size formula methodology section 7 compares the McNemar model with."""

from __future__ import annotations

import math

from .. import formatting

# (z_0.975 + z_0.80)^2 with the rounded z's of the textbook formula
# n = (z_0.975 + z_0.80)^2 / d^2 for a paired t-test at effect size d.
Z_FACTOR = (1.96 + 0.84) ** 2


def questions_for_effect(effect: float) -> int:
    """Questions a paired t-test needs to detect effect size d = `effect`, rounded up.

    Raises:
        ValueError: `effect` is not above 0.
    """
    if not effect > 0:
        raise ValueError(
            f"the effect size must be above 0, got {formatting.design_parameter(effect)}"
        )
    return math.ceil(Z_FACTOR / effect**2)


def detectable_effect(size: int) -> float:
    """The effect size d a paired t-test on `size` questions detects.

    Raises:
        ValueError: `size` is below 1.
    """
    if size < 1:
        raise ValueError(
            f"a paired t-test needs at least one question, got {formatting.count(size)}"
        )
    return math.sqrt(Z_FACTOR / size)
