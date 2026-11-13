"""An estimate with its interval, as a compute function returns a mean or a difference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...evaluation.paired import BootstrapDiff, PairedEstimate
from ...evaluation.power import SIGNIFICANCE_LEVEL
from ...generation.prompt_compare import PromptComparison

# A difference the power model is evaluated at; an effect below it is small (methodology D16).
DELTA = 0.20


@dataclass(frozen=True)
class Estimate:
    """A mean or a difference with the bounds of its interval."""

    value: float
    low: float
    high: float

    @classmethod
    def of(cls, row: PairedEstimate | PromptComparison) -> Estimate:
        """The difference of a paired row with its interval: its `diff`, `low` and `high`."""
        return cls(row.diff, row.low, row.high)

    @classmethod
    def of_bootstrap(cls, result: BootstrapDiff) -> Estimate:
        """The mean difference of a paired bootstrap with its interval."""
        return cls(result.mean_diff, result.low, result.high)

    def includes_zero(self) -> bool:
        """Whether 0 lies in the interval, an end at 0 included: the interval does not exclude 0."""
        return self.low <= 0 <= self.high


def is_significant(p: float) -> bool:
    """Whether a p is below the significance level (`power.SIGNIFICANCE_LEVEL`)."""
    return p < SIGNIFICANCE_LEVEL


class HolmRow(Protocol):
    """A comparison row whose family may have left it without a Holm-adjusted p."""

    @property
    def holm_p(self) -> float | None:
        """The Holm-adjusted p, None when the row was not tested."""
        ...


def holm_or_one(row: HolmRow) -> float:
    """A row's Holm-adjusted p, 1.0 for a row with none: a row that was not tested is no effect."""
    return row.holm_p if row.holm_p is not None else 1.0


def same_reading(old: PairedEstimate, new: PairedEstimate, old_p: float, new_p: float) -> bool:
    """Whether two rows agree in sign, in significance and in whether the interval includes 0."""
    same_sign = (old.diff > 0) == (new.diff > 0) and (old.diff < 0) == (new.diff < 0)
    return (
        same_sign
        and is_significant(old_p) == is_significant(new_p)
        and Estimate.of(old).includes_zero() == Estimate.of(new).includes_zero()
    )


def is_small_effect(difference: Estimate, p: float) -> bool:
    """Whether `difference` is an effect whose size is below `DELTA` (methodology D16).

    An effect is a difference whose interval excludes 0, or whose p `is_significant`.
    """
    effect = not difference.includes_zero() or is_significant(p)
    return effect and abs(difference.value) < DELTA
