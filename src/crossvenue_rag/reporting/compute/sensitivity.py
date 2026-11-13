"""The retrieval evaluation on a later gold version beside a run on an earlier one.

A pair defaults to `v0.2.2-gold` (`run_ids.SENSITIVITY_RUN`) against `v0.2.1-gold`
(`run_ids.SENSITIVITY_BASE_RUN`), two runs made with the same config on the CPU, so a
difference between them comes from the gold labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...evaluation.comparisons import PairedRow
from ...evaluation.results import breakdown_file
from ...run_files import PAIRED_FILE
from ..run_ids import ROUND3_RUN, SENSITIVITY_BASE_RUN, SENSITIVITY_RUN
from ..sources import Sources
from .distinct import shared_value
from .estimates import Estimate, is_significant, same_reading
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


def primary_comparison(src: Sources) -> Comparison:
    """The one comparison the baseline run marks primary.

    Raises:
        StopIteration: the baseline run marks none.
    """
    return next(
        c
        for c in tested_comparisons(src)
        if paired_row(src, SENSITIVITY_BASE_RUN, c).role == "primary"
    )


def tested_comparisons(src: Sources) -> list[Comparison]:
    """The primary and secondary comparisons of the baseline run, in file order."""
    rows = src.run(SENSITIVITY_BASE_RUN).paired.values()
    return [
        Comparison(r.a, r.b, r.metric, r.scope) for r in rows if r.role in ("primary", "secondary")
    ]


def tested_questions(src: Sources, run: str, scope: str) -> int:
    """Questions the baseline's tested comparisons over `scope` pair in `run`, their paired.csv `n`.

    Raises:
        ValueError: The comparisons over `scope` pair different numbers of questions, or none
            is over `scope`.
    """
    sizes = (paired_row(src, run, c).n for c in tested_comparisons(src) if c.scope == scope)
    n = shared_value(sizes)
    if n is None:
        raise ValueError(f"the {scope} comparisons of {run} pair no single number of questions")
    return n


def same_conclusion(
    src: Sources,
    comparison: Comparison,
    base: str,
    run: str,
) -> bool:
    """Whether a comparison keeps its sign, its significance and whether its interval excludes 0.

    Raises:
        ValueError: A run lacks the comparison or its p-value.
    """
    old, new = paired_row(src, base, comparison), paired_row(src, run, comparison)
    return same_reading(old, new, tested_p(old), tested_p(new))


def conclusions_unchanged(src: Sources) -> bool:
    """Whether every primary and secondary comparison keeps its conclusion on `v0.2.2-gold`."""
    return not changed_conclusions(src)


def changed_conclusions(
    src: Sources, base: str = SENSITIVITY_BASE_RUN, run: str = SENSITIVITY_RUN
) -> list[Comparison]:
    """The primary and secondary comparisons whose conclusion differs between two runs."""
    return [c for c in tested_comparisons(src) if not same_conclusion(src, c, base, run)]


def retrieval_changes(src: Sources) -> tuple[list[Comparison], int]:
    """The tested comparisons that change on `v0.3-gold` against `v0.2.1-gold`, and their number.

    Returns the comparisons that change, then the count of primary and secondary comparisons.
    """
    changed = changed_conclusions(src, SENSITIVITY_BASE_RUN, ROUND3_RUN)
    return changed, len(tested_comparisons(src))


def identical(
    src: Sources,
    comparison: Comparison,
    base: str = SENSITIVITY_BASE_RUN,
    run: str = SENSITIVITY_RUN,
) -> bool:
    """Whether a comparison's difference and interval are the same in `base` and `run`.

    Raises:
        ValueError: A run lacks the comparison.
    """
    old = paired_row(src, base, comparison)
    new = paired_row(src, run, comparison)
    return (old.diff, old.low, old.high) == (new.diff, new.low, new.high)


def inconclusive(src: Sources) -> list[Comparison]:
    """Comparisons whose interval excludes 0 in both runs with a p-value at or above the level.

    Raises:
        ValueError: A run lacks a comparison or its p-value.
    """
    out = []
    for c in tested_comparisons(src):
        rows = [paired_row(src, r, c) for r in (SENSITIVITY_BASE_RUN, SENSITIVITY_RUN)]
        if all(
            not Estimate.of(r).includes_zero() and not is_significant(tested_p(r)) for r in rows
        ):
            out.append(c)
    return out
