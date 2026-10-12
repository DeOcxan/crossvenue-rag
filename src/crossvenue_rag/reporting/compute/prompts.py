"""Prompt v2 against v1: comparison.csv of the prompt comparison (`run_ids.PROMPTS_RUN`).

Each row is one configuration on one metric: the value under v1 and v2, the
paired difference v2 - v1 with its interval, the questions that changed and the
p-values, as `crossvenue compare-prompts` saved them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ... import formatting
from ...generation.prompt_compare import COMPARISON_FILE
from ..sources import Sources
from .answers import configurations
from .estimates import Estimate
from .lookup import entry

# How far a p-value may be from the smallest the randomization test can reach.
FLOOR_TOLERANCE = 0.01


@dataclass(frozen=True)
class PromptChange:
    """One configuration on one metric: its mean under each prompt, and v2 - v1."""

    v1: float
    v2: float
    difference: Estimate
    holm_p: float


@dataclass(frozen=True)
class ChangeRow:
    """One configuration's `change` on each of two metrics."""

    config: str
    first: PromptChange
    second: PromptChange


@dataclass(frozen=True)
class ChangeRange:
    """The fewest and the most `changed_questions` of a configuration, and their total."""

    fewest: int
    most: int
    total: int


def change(src: Sources, config: str, metric: str) -> PromptChange:
    """`config` on `metric` under both prompts."""
    row = entry(src.prompts.comparison, (config, metric), COMPARISON_FILE)
    return PromptChange(row.value("v1"), row.value("v2"), Estimate.of(row), row.value("holm_p"))


def change_rows(src: Sources, metrics: tuple[str, str]) -> list[ChangeRow]:
    """Per configuration, its `change` on each of `metrics`."""
    return [
        ChangeRow(c, change(src, c, metrics[0]), change(src, c, metrics[1]))
        for c in configurations(src)
    ]


def questions(src: Sources, metric: str, config: str | None = None) -> int:
    """Questions `config` is compared on for `metric` (by default the first configuration)."""
    return entry(
        src.prompts.comparison, (config or configurations(src)[0], metric), COMPARISON_FILE
    ).n


def v2_never_worse(src: Sources, metric: str) -> bool:
    """Whether no configuration is worse under v2 on any question on `metric`."""
    return all(
        entry(src.prompts.comparison, (c, metric), COMPARISON_FILE).worse == 0
        for c in configurations(src)
    )


def changed_questions(src: Sources, metric: str) -> list[int]:
    """Per configuration, the questions on which v2 differs from v1 on `metric`."""
    rows = [
        entry(src.prompts.comparison, (c, metric), COMPARISON_FILE) for c in configurations(src)
    ]
    return [r.worse + r.better for r in rows]


def change_range(src: Sources, metric: str) -> ChangeRange:
    """`changed_questions` over the configurations."""
    changed = changed_questions(src, metric)
    return ChangeRange(min(changed), max(changed), sum(changed))


def smallest_p(changed: int) -> float:
    """The smallest two-sided p of the randomization test when `changed` questions differ.

    Raises:
        ValueError: `changed` is below 1: with no changed question every p is 1.
    """
    if changed < 1:
        raise ValueError(
            f"the smallest p needs at least one changed question, got {formatting.count(changed)}"
        )
    return 2 / float(2**changed)


def changes_against_v2_at_floor(src: Sources, metric: str) -> bool:
    """Whether every change on `metric` goes against v2 and every p is at its floor.

    A configuration with no changed question has no p below 1 to be at a floor, so it
    holds as it stands.
    """
    rows = {
        c: entry(src.prompts.comparison, (c, metric), COMPARISON_FILE) for c in configurations(src)
    }
    changed = dict(zip(configurations(src), changed_questions(src, metric), strict=True))
    return all(r.better == 0 for r in rows.values()) and all(
        abs(rows[c].p - smallest_p(changed[c])) <= FLOOR_TOLERANCE
        for c in configurations(src)
        if changed[c] >= 1
    )
