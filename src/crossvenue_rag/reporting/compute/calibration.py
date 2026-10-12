"""The judge calibration, on its sample (`run_ids.CALIBRATION_SAMPLE`).

Each candidate's agreement with the human and with the other candidates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from ... import formatting
from ...config_generation import CalibrationDimension
from ...generation.calibration_file import (
    CALIBRATION_FILE,
    DimensionStatistics,
    human_comparison,
    judge_comparison,
    replicate_comparison,
)
from ...generation.calibration_sample import read_sample
from ...generation.grader_agreement import dimension_verdicts, paired_verdicts
from ...generation.records import AnswerRecord
from ...generation.schemas import GeneratedAnswer, is_absence_claim
from ...models import QuestionCategory
from ...stats import cohens_kappa
from ..run_ids import CALIBRATION_SAMPLE
from ..sources import Sources
from .distinct import shared_value
from .estimates import Estimate
from .lookup import entry

# The dimensions a judge is calibrated on, in the order the tables list them.
DIMENSIONS: tuple[CalibrationDimension, ...] = get_args(CalibrationDimension)
# The categories the calibration sample sizes apart; every other category has one size.
OVERSAMPLED: tuple[QuestionCategory, ...] = ("comparison", "unanswerable")
# How the documents and figures write each agreement statistic.
STATISTIC_LABELS = {"kappa": "\u03ba", "spearman": "Spearman"}
# The candidates the calibration table compares, first in calibration.json's order; a row's
# first intervals are each of them against the human.
TABLE_CANDIDATES = 2


def _defined(value: float | None, name: str) -> float:
    """A statistic the documents state; a calibration without it fails."""
    if value is None:
        raise ValueError(f"{CALIBRATION_FILE} records no {name} where the documents state one")
    return value


def _dimension_row(src: Sources, comparison: str, dimension: str) -> DimensionStatistics:
    """calibration.json's statistics of `comparison` on `dimension`; ValueError if it has none."""
    rows = entry(src.calibration.comparisons, comparison, f"{CALIBRATION_FILE}'s comparisons")
    return entry(rows, dimension, f"{CALIBRATION_FILE}'s {comparison}")


def passed_dimensions(src: Sources, judge: str) -> frozenset[str]:
    """The dimensions whose threshold `judge` meets on the point, as calibration.json records.

    Raises:
        ValueError: `judge` is no candidate of calibration.json.
    """
    return frozenset(
        entry(src.calibration.candidates, judge, f"{CALIBRATION_FILE}'s candidates").passes
    )


def passes(src: Sources, judge: str) -> int:
    """Calibration thresholds `judge` meets (`passed_dimensions`).

    Raises:
        ValueError: `judge` is no candidate of calibration.json.
    """
    return len(passed_dimensions(src, judge))


def interval_informative(src: Sources, comparison: str, dimension: str) -> bool:
    """Whether the statistic of `comparison` on `dimension` is defined in every resample.

    An interval over the resamples where it is defined leaves the others out, so it
    cannot establish a threshold.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`.
    """
    row = _dimension_row(src, comparison, dimension)
    return row.undefined(statistic(src, dimension)) == 0


def passes_on_interval(src: Sources, judge: str) -> tuple[str, ...]:
    """The dimensions `judge` passes whose informative interval meets the threshold too.

    Raises:
        ValueError: `judge` is no candidate of calibration.json.
    """
    comparison = human_comparison(judge)
    passed = passed_dimensions(src, judge)
    return tuple(
        d
        for d in DIMENSIONS
        if d in passed
        and interval_informative(src, comparison, d)
        and agreement_interval(src, comparison, d).low >= threshold(src, d)
    )


def passes_by_reach(src: Sources, judge: str) -> tuple[frozenset[str], frozenset[str]]:
    """The dimensions `judge` passes on the interval too, and those it passes on the point only.

    Raises:
        ValueError: `judge` is no candidate of calibration.json.
    """
    point = passed_dimensions(src, judge)
    interval = frozenset(passes_on_interval(src, judge))
    return interval, point - interval


def answers_rated(src: Sources) -> int:
    """Answers of the calibration sample, as calibration.json records."""
    return src.calibration.answers_rated


def dimension_count(src: Sources) -> int:
    """Dimensions the calibration sets a threshold on."""
    return len(src.calibration.thresholds)


def comparison_kappa(src: Sources, comparison: str, dimension: str) -> float:
    """Cohen's kappa of `comparison` (`opus vs human`, ...) on `dimension`.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`.
    """
    return _defined(
        _dimension_row(src, comparison, dimension).kappa,
        "kappa",
    )


def verdict_pairs(src: Sources, judge: str, dimension: str) -> Counter[tuple[bool, bool]]:
    """(judge, human) verdicts on a binary `dimension`.

    One per claim, or one per answer on `behaviour_correct`.

    Raises:
        UserError: `judge` is not a plain name.
        ValueError: `dimension` is not a binary dimension.
    """
    pairs = paired_verdicts(src.judge_grades(judge), src.human)
    return Counter(dimension_verdicts(pairs, dimension))


@dataclass(frozen=True)
class HumanClaims:
    """The human's claims: `claims` rated, `unsupported` and `misattributed`."""

    claims: int
    unsupported: int
    misattributed: int


def human_negatives(src: Sources) -> HumanClaims:
    """The claims the human rated, and those rated unsupported and misattributed."""
    claims = [c for r in src.human.values() for c in r.claims]
    return HumanClaims(
        claims=len(claims),
        unsupported=sum(1 for c in claims if not c.supported),
        misattributed=sum(1 for c in claims if not c.venue_correct),
    )


def venue_disagreement(src: Sources, judge: str) -> tuple[int, int]:
    """(claims either grader marked misattributed, claims both did).

    Raises:
        UserError: `judge` is not a plain name.
    """
    pairs = verdict_pairs(src, judge, "venue_correct")
    either = sum(k for (a, b), k in pairs.items() if not a or not b)
    return either, pairs[(False, False)]


def judge_candidate_count(src: Sources) -> int:
    """Judge candidates the calibration compares."""
    return len(src.calibration.candidates)


def resamples(src: Sources) -> int:
    """Bootstrap resamples of the calibration intervals."""
    return src.calibration.resamples


def bootstrap_seed(src: Sources) -> int:
    """The seed of the calibration bootstrap."""
    return src.calibration.seed


def threshold(src: Sources, dimension: str) -> float:
    """The agreement `dimension` must reach; ValueError if calibration.json sets none.

    Raises:
        ValueError: calibration.json sets no threshold for `dimension`.
    """
    return entry(src.calibration.thresholds, dimension, f"{CALIBRATION_FILE}'s thresholds")


def statistic(src: Sources, dimension: str) -> str:
    """The agreement statistic of `dimension`: `kappa` or `spearman`; ValueError if none.

    Raises:
        ValueError: calibration.json names no statistic for `dimension`.
    """
    return entry(src.calibration.statistic, dimension, f"{CALIBRATION_FILE}'s statistics")


def compared_count(src: Sources, comparison: str, dimension: str) -> int:
    """Claims or answers `comparison` (`opus vs human`, ...) compares on `dimension`.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`.
    """
    return _dimension_row(src, comparison, dimension).n


def agreement_interval(src: Sources, comparison: str, dimension: str) -> Estimate:
    """The statistic of `comparison` on `dimension`, with its interval.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`, or records no interval
          of its statistic there.
    """
    row = _dimension_row(src, comparison, dimension)
    name = statistic(src, dimension)
    bounds = row.interval(name)
    if bounds is None:
        raise ValueError(f"{comparison} records no interval for {name} on {dimension}")
    return Estimate(_defined(row.value(name), name), bounds[0], bounds[1])


def same_agreement(src: Sources, comparison: str, dimension: str, other: str) -> bool:
    """Whether `comparison` has the same `agreement_interval` on two dimensions.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension` or on `other`, or records
          no interval of the statistic there.
    """
    return agreement_interval(src, comparison, dimension) == agreement_interval(
        src, comparison, other
    )


def kappa_undefined(src: Sources, comparison: str, dimension: str) -> int:
    """Resamples in which the kappa of `comparison` on `dimension` is undefined.

    Raises:
        ValueError: calibration.json has no `comparison` on `dimension`.
    """
    return _dimension_row(src, comparison, dimension).undefined("kappa")


def behaviour_kappa(src: Sources, judge: str) -> float:
    """Cohen's kappa of `judge` with the human on `behaviour_correct`, over every graded answer.

    Raises:
        UserError: `judge` is not a plain name.
    """
    pairs = paired_verdicts(src.judge_grades(judge), src.human)
    verdicts = dimension_verdicts(pairs, "behaviour_correct")
    return cohens_kappa([a for a, _ in verdicts], [b for _, b in verdicts]).kappa


def sample_ids(src: Sources) -> list[str]:
    """The answer ids of the calibration sample, in file order."""
    return read_sample(src.path(CALIBRATION_SAMPLE))[1]


def sample_answers(src: Sources) -> list[GeneratedAnswer]:
    """The generated answers of the calibration sample, in answer-run order."""
    ids = set(sample_ids(src))
    return [r.generated().answer for r in src.v1.records if r.answer_id in ids]


def sample_claims(src: Sources) -> int:
    """Claims of the answers of the calibration sample."""
    return sum(len(a.claims) for a in sample_answers(src))


def sample_by(src: Sources, field: str) -> Counter[str]:
    """Answers of the calibration sample per value of the record's `field` (`config`, ...).

    Raises:
        ValueError: `field` is not a text field of an answer record.
    """
    by_id = {r.answer_id: r for r in src.v1.records}
    return Counter(
        _text_field(entry(by_id, a, "the prompt v1 answer run"), field) for a in sample_ids(src)
    )


def _text_field(record: AnswerRecord, field: str) -> str:
    """The text field `field` of an answer record."""
    # A name that is no field reads as None, so it is refused as a field that is not text.
    value = getattr(record, field, None)
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a text field of an answer record")
    return value


def other_category_size(src: Sources) -> int | None:
    """The one sample size of the categories other than `OVERSAMPLED`; None if they differ."""
    sizes = sample_by(src, "category")
    return shared_value(k for c, k in sizes.items() if c not in OVERSAMPLED)


def excludes_exactly(src: Sources, questions: Sequence[str]) -> bool:
    """Whether the calibration excludes the answers to `questions` and no other."""
    excluded = src.default_config.generation.calibration_exclude
    return sorted(excluded) == sorted(questions)


def sample_has_no_v2_forms(src: Sources) -> bool:
    """Whether no calibration answer is partial and no claim of one has the absence form."""
    sample = sample_answers(src)
    return not any(a.partial for a in sample) and not any(
        is_absence_claim(c.text) for a in sample for c in a.claims
    )


def other_candidate(src: Sources) -> str:
    """The one judge candidate of the calibration that is not the prompt v1 judged run's judge.

    Raises:
        ValueError: the candidates other than that judge are not exactly one.
    """
    others = [c for c in candidates(src) if c != src.judged_v1.config.generation.chosen_judge()]
    if len(others) != 1:
        raise ValueError(
            f"the calibration has {formatting.count(len(others))} candidates besides the judge; "
            "the documents name one"
        )
    return others[0]


def judge_with_human(src: Sources) -> str:
    """The comparison of the prompt v1 judged run's judge with the human (`human_comparison`)."""
    return human_comparison(src.judged_v1.config.generation.chosen_judge())


def judge_with_repeat(src: Sources) -> str:
    """The prompt v1 judged run's judge against its own first repeat (`replicate_comparison`)."""
    return replicate_comparison(src.judged_v1.config.generation.chosen_judge(), 1)


def candidates(src: Sources) -> list[str]:
    """The judge candidates, in calibration.json's order."""
    return list(src.calibration.candidates)


def selected_passes_most(src: Sources, judge: str) -> bool:
    """Whether `judge` is selected and passes more thresholds than every other candidate."""
    others = [passes(src, j) for j in candidates(src) if j != judge]
    return src.calibration.selected == judge and all(passes(src, judge) > k for k in others)


@dataclass(frozen=True)
class TableRow:
    """One dimension's row of the calibration table."""

    dimension: str
    with_human: int  # claims or answers the first candidate and the human compare
    between: int  # those the two candidates compare
    statistic: str
    threshold: float
    # `agreement_interval` of the first candidate vs the human, the second vs the human,
    # and the first vs the second.
    intervals: list[Estimate]


def table_rows(src: Sources) -> list[TableRow]:
    """Per dimension of `DIMENSIONS`, the judges' agreement with the human and each other."""
    first, second = candidates(src)[:TABLE_CANDIDATES]
    comparisons = (
        human_comparison(first),
        human_comparison(second),
        judge_comparison(first, second),
    )
    return [
        TableRow(
            d,
            compared_count(src, comparisons[0], d),
            compared_count(src, comparisons[2], d),
            statistic(src, d),
            threshold(src, d),
            [agreement_interval(src, c, d) for c in comparisons],
        )
        for d in DIMENSIONS
    ]


def agreement_range(
    src: Sources, comparison: str, dimensions: Sequence[str]
) -> tuple[float, float]:
    """(lowest, highest) raw agreement of `comparison` over `dimensions`.

    Raises:
        ValueError: calibration.json has no `comparison`.
    """
    values = [
        _defined(_dimension_row(src, comparison, d).agreement, "agreement") for d in dimensions
    ]
    return min(values), max(values)


def human_accepted(src: Sources) -> int:
    """Claims the human rated supported."""
    human = human_negatives(src)
    return human.claims - human.unsupported


def judgement_cost(src: Sources, judge: str, replicate: int = 0) -> float:
    """Recorded cost of `judge`'s calibration judgments in the prompt v1 answer run.

    Its first grading, or with `replicate` > 0 that repeat (`judge_sample.judgements_path`).

    Raises:
        UserError: `judge` is not a plain name.
    """
    rows = src.v1.judgements(judge, replicate)
    return float(sum(r.judge.call.paid_usd() for r in rows if r.judge is not None))


def rejection_kinds_hold(
    src: Sources, kinds: Mapping[str, int], allowed: Sequence[str], once: str
) -> bool:
    """Whether a table of kinds has one row per claim the judge accepted and the human rejected.

    The judge is the prompt v1 judged run's. `kinds` counts the table's rows per kind:
    every kind is one of `allowed`, `once` occurs exactly once, and the rows number those
    claims, the verdict pair (True, False) of `verdict_pairs`.
    """
    judge = src.judged_v1.config.generation.chosen_judge()
    return (
        kinds.get(once, 0) == 1
        and set(kinds) <= set(allowed)
        and sum(kinds.values()) == verdict_pairs(src, judge, "supported")[(True, False)]
    )


def rejects_none_accepted(src: Sources, judge: str) -> bool:
    """Whether `judge` rejects none of the claims the human accepts: no pair (False, True).

    Raises:
        UserError: `judge` is not a plain name.
    """
    return verdict_pairs(src, judge, "supported")[(False, True)] == 0
