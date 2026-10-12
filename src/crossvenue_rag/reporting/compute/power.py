"""The power analysis of the p-value run's recall family: power.json of `run_ids.POWER_RUN`.

`Sources.power` loads it; the analytic McNemar model the methodology states is
`evaluation.mcnemar_power`.
"""

from __future__ import annotations

from ... import formatting
from ...evaluation.power import POWER_FILE, TARGET_POWER_PERCENT, Adjustment
from ..sources import Sources

# How close a saved delta must be to the one asked for: the grid is in hundredths.
DELTA_TOLERANCE = 1e-9


def detectable_gap(src: Sources, adjustment: Adjustment) -> float:
    """The smallest gap detected with `TARGET_POWER_PERCENT` power.

    Under `adjustment`: `holm_first_step` (Holm's first step) or `unadjusted`.

    Raises:
        ValueError: no simulated delta reaches the target power under `adjustment`.
    """
    delta = src.power.delta_at_target(adjustment)
    if delta is None:
        raise ValueError(
            f"no simulated delta reaches {formatting.count(TARGET_POWER_PERCENT)}% power "
            f"({adjustment})"
        )
    return delta


def family_size(src: Sources) -> int:
    """Comparisons of the recall family, which Holm adjusts within."""
    return len(src.power.family)


def family_methods(src: Sources) -> int:
    """The methods the recall family's comparisons compare."""
    return len({method for pair in src.power.family for method in pair})


def power_at_gap(src: Sources, delta: float, adjustment: Adjustment) -> float:
    """The simulated power at a gap of `delta`, under `adjustment` (as `detectable_gap`).

    Raises:
        ValueError: power.json simulates no `delta`, or `delta` is not reachable.
    """
    point = next((p for p in src.power.points if abs(p.delta - delta) < DELTA_TOLERANCE), None)
    if point is None:
        raise ValueError(f"{POWER_FILE} simulates no delta {formatting.design_parameter(delta)}")
    power = point.power_under(adjustment)
    if power is None:
        raise ValueError(
            f"the simulated delta {formatting.design_parameter(delta)} is not reachable"
        )
    return power


def gap_ceiling(src: Sources) -> float:
    """The largest gap the model of the power run allows."""
    return src.power.ceiling


def below_detectable_gap(src: Sources, gap: float, adjustment: Adjustment) -> bool:
    """Whether `gap` is below the smallest gap detected with the target power.

    Raises:
        ValueError: no simulated delta reaches the target power under `adjustment`.
    """
    return gap < detectable_gap(src, adjustment)
