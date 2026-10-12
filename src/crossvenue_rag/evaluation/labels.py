"""The written form, in output for a person, of a method, a scope, a value or a table."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence

from .. import formatting
from ..retrieval.methods import ORACLE_METHODS, POOL_METHODS


def fmt(value: float | None) -> str:
    """A metric or agreement value at `formatting.METRIC_DECIMALS`, or `formatting.MISSING`.

    A value is missing where its rows define none: a slice with no labels or no judgment
    history, answers without claims, or a statistic undefined on its data.
    """
    return formatting.optional(value, formatting.metric)


def markdown_table(
    header: Sequence[str], rows: Iterable[Sequence[str]], *, right: Collection[int] = ()
) -> str:
    """A Markdown table: the header, its rule, then one line per row of cells.

    The rule is `---` under each column, and `---:`, which right-aligns the column, under
    the columns whose positions `right` holds (0 is the first).
    """
    rule = ["---:" if i in right else "---" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(rule) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return "\n".join(lines)


def scope_label(scope: str) -> str:
    """A comparison scope as a report shows it; `paired.csv` keeps the saved name."""
    return scope.replace("labelled", "labeled")


def method_label(method: str) -> str:
    """A method as a report shows it: a dagger outside the labeling pool, `(oracle)` on oracles."""
    label = method
    if method in ORACLE_METHODS:
        label += " (oracle)"
    if method not in POOL_METHODS:
        label += " \u2020"
    return label


POOL_NOTE = (
    "\u2020 Not in the labeling pool (`retrieval.methods.POOL_METHODS`). `unjudged@k` is "
    "the share of a method's top k that no judgment covered (pool plus augmentation, from "
    "the journal), each such chunk counted as a miss; it does not measure the bias itself. "
    "`ndcg_prime@k` is nDCG on the condensed list (Sakai 2007, full entry under "
    "References in `evals/methodology.md`): unjudged chunks are dropped from the ranking "
    "instead of scored as "
    "irrelevant, and its gap to nDCG@k is the change that dropping them makes. Both are "
    f"`{formatting.MISSING}` when the judgment history is not available. "
    "venue_coverage@k needs no gold and is unaffected. (oracle) runs venue_split with the "
    "question's labeled `venues` in "
    "place of the detected ones, which a deployed system does not have; it bounds what "
    "better venue detection can add to venue coverage, and it is not an upper bound on "
    "recall, precision or nDCG."
)
