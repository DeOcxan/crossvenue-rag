"""Outcomes of the answer runs per category, per configuration, prompt v1 against v2.

The outcome of an answer is the rule of methodology section 9a
(`answers.outcomes`); a category's questions are those of the gold set.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...generation.answer_rows import in_scope
from ...generation.schemas import ANSWERED, Outcome, is_absence_claim
from ...models import QuestionCategory
from ..sources import Sources
from . import datasets
from .answers import (
    CITATION,
    PROMPT_V1,
    PROMPT_V2,
    answer,
    configurations,
    outcome_count,
    outcomes,
    refusal_groups,
    refusal_ids,
    refusing_config,
    wrong_refusals,
)
from .distinct import shared_value
from .lookup import entry

# The categories of the gold set, in the order the answer tables list them.
CATEGORIES: tuple[QuestionCategory, ...] = (
    "comparison",
    "unanswerable",
    "adversarial",
    "fact",
    "ambiguous",
    "noisy",
    "version_sensitive",
    "multihop",
)


def expected_behaviour(src: Sources, category: str) -> str | None:
    """The one behavior every question of `category` expects; None if they differ."""
    ids = datasets.category_ids(src, category)
    return shared_value(src.questions[q].expected_behavior for q in ids)


def outcome_counts(src: Sources, category: str | None, kind: Outcome, prompt: int) -> list[int]:
    """`outcome_count` of each configuration of `configurations` in turn.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return [outcome_count(src, c, category, kind, prompt) for c in configurations(src)]


def outcome_changes(src: Sources, category: str | None, kind: Outcome) -> list[tuple[int, int]]:
    """Per configuration, (answers with outcome `kind` under v1, under v2)."""
    return list(
        zip(
            outcome_counts(src, category, kind, PROMPT_V1),
            outcome_counts(src, category, kind, PROMPT_V2),
            strict=True,
        )
    )


@dataclass(frozen=True)
class OutcomeRow:
    """One category's answers with one outcome, per configuration, under each prompt."""

    category: str
    questions: int
    expected: str | None  # the category's expected behavior, None when it has no questions
    changes: list[tuple[int, int]]  # `outcome_changes`


def outcome_table(src: Sources, kind: Outcome) -> list[OutcomeRow]:
    """Per category of `CATEGORIES`, its answers with outcome `kind`."""
    return [
        OutcomeRow(
            c,
            datasets.category_count(src, c),
            expected_behaviour(src, c),
            outcome_changes(src, c, kind),
        )
        for c in CATEGORIES
    ]


def no_outcome(src: Sources, kind: Outcome, prompt: int) -> bool:
    """Whether no configuration has an answer with outcome `kind` in the run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return not any(outcome_counts(src, None, kind, prompt))


def clarifications(src: Sources, config: str) -> tuple[int, int]:
    """(clarifying questions of `config` under v1, under v2)."""
    return outcome_count(src, config, None, "asked", PROMPT_V1), outcome_count(
        src, config, None, "asked", PROMPT_V2
    )


def clarifications_only_on(src: Sources, configs: tuple[str, ...], category: str) -> bool:
    """Whether every clarifying question of `configs` under v2 is on a `category` question."""
    return all(
        entry(src.questions, q, "the gold set").effective_category == category
        for (c, q), o in outcomes(src, PROMPT_V2).items()
        if o == "asked" and c in configs
    )


def never_clarify(src: Sources, configs: tuple[str, ...]) -> bool:
    """Whether none of `configs` asks a clarifying question under v1 or v2."""
    return all(sum(clarifications(src, c)) == 0 for c in configs)


def wrong_refusal_changes(src: Sources) -> list[tuple[int, int]]:
    """Per configuration, (wrong refusals under v1, under v2)."""
    return [
        (wrong_refusals(src, c, PROMPT_V1), wrong_refusals(src, c, PROMPT_V2))
        for c in configurations(src)
    ]


def wrong_refusals_fall(src: Sources) -> bool:
    """Whether every configuration has fewer wrong refusals under v2 than under v1."""
    return all(after < before for before, after in wrong_refusal_changes(src))


def citing_refusals(src: Sources) -> int:
    """Refusals of `refusal_ids` whose text cites a chunk as [n]."""
    return sum(
        bool(CITATION.search(answer(src, refusing_config(src), q).answer)) for q in refusal_ids(src)
    )


def refusal_claims(src: Sources) -> int:
    """Claims the refusals of `refusal_ids` carry between them."""
    return sum(len(answer(src, refusing_config(src), q).claims) for q in refusal_ids(src))


def answered_refusals(src: Sources, config: str) -> int:
    """Questions of `refusal_ids` that `config` answered, fully or in part, under v1."""
    v1 = outcomes(src, PROMPT_V1)
    return sum(entry(v1, (config, q), "the v1 outcomes") in ANSWERED for q in refusal_ids(src))


def group_outcomes(src: Sources, group: str) -> dict[Outcome, int]:
    """The v2 outcomes of `refusing_config` on one of `refusal_groups`: full, partial, refused.

    Raises:
        ValueError: `group` is not `a`, `b` or `c`.
    """
    v2 = outcomes(src, PROMPT_V2)
    ids = entry(refusal_groups(src), group, "the refusal groups")
    return {
        k: sum(entry(v2, (refusing_config(src), q), "the v2 outcomes") == k for q in ids)
        for k in (*ANSWERED, "refused")
    }


def non_refusals(src: Sources, config: str, category: str, prompt: int) -> int:
    """Questions of `category` that `config` does not refuse in the run of `prompt`.

    Raises:
        ValueError: `prompt` is not 1 or 2.
    """
    return datasets.category_count(src, category) - outcome_count(
        src, config, category, "refused", prompt
    )


def judged_behaviour(src: Sources, config: str, category: str) -> tuple[int, int]:
    """(answers graded `behaviour_correct` under v1, under v2) of `config` on `category`.

    Over the questions both judged runs judged.

    Raises:
        ValueError: A grade is neither 0 nor 1.
    """
    judged = [
        {
            r.question_id: r.judge.metrics.behaviour_correct
            for r in run.records
            if r.config == config and in_scope(r, category) and r.judge is not None
        }
        for run in (src.judged_v1, src.judged_v2)
    ]
    both = sorted(set(judged[0]) & set(judged[1]))
    grades = [[run[q] for q in both] for run in judged]
    if any(g not in (0, 1) for run in grades for g in run):
        raise ValueError(f"{config} on {category}: a behaviour_correct grade is neither 0 nor 1")
    return sum(g == 1 for g in grades[0]), sum(g == 1 for g in grades[1])


def absence_claims(src: Sources) -> tuple[int, int]:
    """(claims with the absence form, claims) of the prompt v2 answers."""
    claims = [c for r in src.v2.records for c in r.generated().answer.claims]
    return sum(is_absence_claim(c.text) for c in claims), len(claims)


def v2_unjudged_claims(src: Sources) -> int:
    """Claims of the prompt v2 answers whose judgment failed; the attribution table omits them."""
    return sum(
        len(r.generated().answer.claims)
        for r in src.judged_v2.records
        if r.generation is not None and r.judge is None
    )
