"""The retrieval evaluation on `v0.2.2-gold` beside its baseline on `v0.2.1-gold`.

Both runs were made with the same config on the CPU (`run_ids.SENSITIVITY_RUN`,
`run_ids.SENSITIVITY_BASE_RUN`), so a difference between them comes from the gold labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...evaluation.comparisons import PairedRow
from ...evaluation.results import breakdown_file
from ...run_files import PAIRED_FILE
from ..run_ids import SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from ..sources import Sources
from .distinct import shared_value
from .estimates import Estimate, is_significant
from .lookup import entry


@dataclass(frozen=True)
class Comparison:
    """A paired comparison as the run's paired.csv keys it: a - b on a metric over a scope."""

    a: str
    b: str
    metric: str
    scope: str


def run_metric(src: Sources, run: str, category: str, method: str, name: str) -> float:
    """`name` of `method` on the questions of `category` in `run`'s by_category.csv.

    Raises:
        ValueError: The run has no row for (`category`, `method`) or no column `name`.
    """
    row = entry(
        src.run(run).by_category, (category, method), f"{breakdown_file('category')} of {run}"
    )
    return row.value(name)


def paired_row(src: Sources, run: str, comparison: Comparison) -> PairedRow:
    """The row of `comparison` in `run`'s paired.csv.

    Raises:
        ValueError: The run has no such row.
    """
    key = (comparison.a, comparison.b, comparison.metric, comparison.scope)
    return entry(src.run(run).paired, key, f"{PAIRED_FILE} of {run}")


def tested_p(row: PairedRow) -> float:
    """A comparison's p-value: Holm-adjusted for a secondary one, unadjusted for the primary.

    Raises:
        ValueError: The row has neither p-value.
    """
    p = row.holm_p if row.holm_p is not None else row.randomisation_p
    if p is None:
        raise ValueError(f"{row.a} - {row.b} on {row.metric} carries no p-value")
    return p


def tested_comparisons(src: Sources) -> list[Comparison]:
    """The primary and secondary comparisons of the baseline run, in file order."""
    rows = src.run(SENSITIVITY_BASE_RUN).paired.values()
    return [
        Comparison(r.a, r.b, r.metric, r.scope) for r in rows if r.role in ("primary", "secondary")
    ]


def tested_questions(src: Sources, run: str, scope: str) -> int:
    """Questions the tested comparisons of `run` over `scope` pair, their paired.csv `n`.

    Raises:
        ValueError: The comparisons over `scope` pair different numbers of questions, or none
            is over `scope`.
    """
    sizes = (paired_row(src, run, c).n for c in tested_comparisons(src) if c.scope == scope)
    n = shared_value(sizes)
    if n is None:
        raise ValueError(f"the {scope} comparisons of {run} pair no single number of questions")
    return n


def same_conclusion(src: Sources, comparison: Comparison) -> bool:
    """Whether a comparison keeps its sign, its significance and whether its interval excludes 0.

    Raises:
        ValueError: A run lacks the comparison or its p-value.
    """
    old, new = (paired_row(src, run, comparison) for run in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN))
    old_ci, new_ci = Estimate.of(old), Estimate.of(new)
    same_sign = (old.diff > 0) == (new.diff > 0) and (old.diff < 0) == (new.diff < 0)
    return (
        same_sign
        and is_significant(tested_p(old)) == is_significant(tested_p(new))
        and old_ci.includes_zero() == new_ci.includes_zero()
    )


def conclusions_unchanged(src: Sources) -> bool:
    """Whether every primary and secondary comparison keeps its conclusion on `v0.2.2-gold`."""
    return all(same_conclusion(src, c) for c in tested_comparisons(src))


def identical(src: Sources, comparison: Comparison) -> bool:
    """Whether a comparison's difference and interval are the same in both runs.

    Raises:
        ValueError: A run lacks the comparison.
    """
    old, new = (paired_row(src, run, comparison) for run in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN))
    return (old.diff, old.low, old.high) == (new.diff, new.low, new.high)


def inconclusive(src: Sources) -> list[Comparison]:
    """Comparisons whose interval excludes 0 in both runs with a p-value at or above the level.

    Raises:
        ValueError: A run lacks a comparison or its p-value.
    """
    out = []
    for c in tested_comparisons(src):
        rows = [paired_row(src, run, c) for run in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN)]
        if all(
            not Estimate.of(r).includes_zero() and not is_significant(tested_p(r)) for r in rows
        ):
            out.append(c)
    return out
