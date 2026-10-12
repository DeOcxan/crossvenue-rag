"""An estimate with its interval, as a compute function returns a mean or a difference."""

from __future__ import annotations

from dataclasses import dataclass

from ...evaluation.paired import BootstrapDiff, PairedEstimate
from ...evaluation.power import SIGNIFICANCE_LEVEL
from ...generation.prompt_compare import PromptComparison


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
    """Whether an adjusted p is below the significance level (`power.SIGNIFICANCE_LEVEL`)."""
    return p < SIGNIFICANCE_LEVEL


def is_small_effect(difference: Estimate, p: float, delta: float) -> bool:
    """Whether `difference` is an effect whose size is below `delta` (methodology D16).

    An effect is a difference whose interval excludes 0, or whose p `is_significant`.
    """
    effect = not difference.includes_zero() or is_significant(p)
    return effect and abs(difference.value) < delta
