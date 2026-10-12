"""The judged answer runs: coverage, behavior and venue attribution.

`src.judged_v1` and `src.judged_v2` are the answers of prompts v1 and v2 as
Opus judged them (`run_ids.JUDGED_V1`, `run_ids.JUDGED_V2`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import get_args

from ...config_generation import AnswerScope
from ...corpus.artifacts import CHUNKS_FILENAME
from ...generation.answer_pairs import ConfigComparison
from ...generation.answer_rows import in_scope
from ...generation.answering import split_answer_id
from ...generation.attribution import WRAPPERS, AttributionSummary
from ...generation.records import AnswerRecord
from ...generation.schemas import Claim, outcome
from ...models import COMPARISON
from ...run_files import PAIRED_FILE, SUMMARY_FILE
from ...stats import mean
from ..sources import Sources
from .answers import (
    PROMPT_V1,
    PROMPTS,
    attribution,
    configurations,
    judged_run,
    primary_pair,
)
from .corpus import source_entries
from .estimates import Estimate, is_significant
from .lookup import entry
from .memo import per_sources


def exchanges(src: Sources) -> set[str]:
    """The exchanges a claim can be attributed to: the venues of data/sources.yaml but CCXT's."""
    return {e.venue for e in source_entries(src)} - WRAPPERS


def judged_summary(src: Sources, prompt: int, config: str, column: str) -> float:
    """A column of `config`'s row of a judged run's summary.csv (`coverage:comparison`).

    Raises:
        ValueError: `prompt` is not 1 or 2, or the judged run's summary.csv has no row of
            `config` or no column `column`.
    """
    return entry(judged_run(src, prompt).summary, config, SUMMARY_FILE).value(column)


def judged_summary_rows(src: Sources, columns: tuple[str, ...]) -> list[tuple[str, list[float]]]:
    """Per configuration, `columns` of its row of the prompt v1 judged summary.csv."""
    return [
        (c, [judged_summary(src, PROMPT_V1, c, col) for col in columns])
        for c in configurations(src)
    ]


@dataclass(frozen=True)
class PrimaryRow:
    """The primary pair a - b on one metric."""

    metric: str
    questions: int
    mean_a: float
    mean_b: float
    difference: Estimate
    randomisation_p: float
    holm_p: float


def primary_rows(src: Sources, metrics: tuple[str, ...], scope: str) -> list[PrimaryRow]:
    """Per metric, the primary pair (`primary_pair`) in the prompt v1 judged run."""
    rows = []
    for metric in metrics:
        r = entry(src.judged_v1.paired, (*primary_pair(src), metric, scope), PAIRED_FILE)
        rows.append(
            PrimaryRow(
                metric,
                r.n,
                r.value("mean_a"),
                r.value("mean_b"),
                Estimate.of(r),
                r.value("randomisation_p"),
                r.value("holm_p"),
            )
        )
    return rows


def means_zero(src: Sources, pair: tuple[str, str], metric: str, scope: str) -> bool:
    """Whether both configurations of `pair` have a mean of 0 on `metric` over `scope`."""
    r = entry(src.judged_v1.paired, (*pair, metric, scope), PAIRED_FILE)
    return r.mean_a == 0 and r.mean_b == 0


@dataclass(frozen=True)
class Tested:
    """A difference with its interval and its Holm p."""

    difference: Estimate
    holm_p: float


def _tested(row: ConfigComparison) -> Tested:
    return Tested(Estimate.of(row), row.value("holm_p"))


@dataclass(frozen=True)
class PairRow:
    """a - b on coverage and `behaviour_correct`; `coverage` is None for the primary pair."""

    a: str
    b: str
    coverage: Tested | None
    behaviour: Tested


def pair_rows(src: Sources, scope: str) -> list[PairRow]:
    """Every pair of configurations on coverage and `behaviour_correct`, prompt v1."""
    rows = []
    for a, b in combinations(configurations(src), 2):
        coverage = entry(src.judged_v1.paired, (a, b, "coverage", scope), PAIRED_FILE)
        behaviour = entry(src.judged_v1.paired, (a, b, "behaviour_correct", scope), PAIRED_FILE)
        primary = coverage.role == "primary"
        rows.append(PairRow(a, b, None if primary else _tested(coverage), _tested(behaviour)))
    return rows


def pair_of[RowT: (PairRow, AttributionPairRow)](
    rows: Sequence[RowT], pair: tuple[str, str]
) -> RowT:
    """The row of `rows` (`pair_rows`, `attribution_pair_rows`) for `pair`; ValueError if none.

    Raises:
        ValueError: `rows` has no row of `pair`.
    """
    for r in rows:
        if (r.a, r.b) == pair:
            return r
    raise ValueError(f"{pair[0]} - {pair[1]} is no pair of configurations in matrix order")


def metric_answer_range(src: Sources, metric: str, scope: str) -> tuple[int, int]:
    """(fewest, most) judged prompt v1 answers of a configuration that carry `metric` in `scope`.

    Raises:
        ValueError: `metric` is not one of `METRICS`.
    """
    counts = [
        sum(
            1
            for r in src.judged_v1.records
            if r.config == config
            and r.judge is not None
            and in_scope(r, scope)
            and r.judged().metrics.value(metric) is not None
        )
        for config in configurations(src)
    ]
    return min(counts), max(counts)


@dataclass(frozen=True)
class AttributionPairRow:
    """a - b on venue_attribution_error, prompt v1: per scope, the questions paired and the test."""

    a: str
    b: str
    comparison_questions: int
    comparison: Tested
    all_questions: int
    all: Tested


def attribution_pair_rows(src: Sources) -> list[AttributionPairRow]:
    """Every pair of configurations on venue_attribution_error, on comparison and all questions."""
    rows = []
    for a, b in combinations(configurations(src), 2):
        cells = [
            entry(src.judged_v1.paired, (a, b, "venue_attribution_error", scope), PAIRED_FILE)
            for scope in get_args(AnswerScope)
        ]
        tested = [_tested(r) for r in cells]
        rows.append(AttributionPairRow(a, b, cells[0].n, tested[0], cells[1].n, tested[1]))
    return rows


def attribution_pairs_touching_zero(src: Sources, scope: str) -> int:
    """Pairs whose venue_attribution_error interval over `scope` includes 0 (`Estimate`)."""
    rows = attribution_pair_rows(src)
    tests = [r.comparison if scope == COMPARISON else r.all for r in rows]
    return sum(1 for t in tests if t.difference.includes_zero())


def any_pair_significant(src: Sources, metric: str) -> bool:
    """Whether any comparison on `metric`, primary or secondary, reaches p_Holm < the level."""
    return any(
        is_significant(r.value("holm_p"))
        for k, r in src.judged_v1.paired.items()
        if k[2] == metric and r.role in ("primary", "secondary")
    )


def ahead_of_every_pair(src: Sources, config: str, metric: str, scope: str) -> bool:
    """Whether `config` is ahead of every other configuration on `metric` over `scope`.

    In the prompt v1 judged run: each pair's difference is in the metric's direction and
    its p_Holm `is_significant`.

    Raises:
        ValueError: paired.csv has no pair of `config` on `metric` over `scope`.
    """
    rows = [
        (a, r)
        for (a, b, m, s), r in src.judged_v1.paired.items()
        if m == metric and s == scope and config in (a, b)
    ]
    if not rows:
        raise ValueError(f"{PAIRED_FILE} has no pair of {config} on {metric} over {scope}")
    return all(
        ((r.diff > 0) if a == config else (r.diff < 0)) == r.higher_is_better
        and r.holm_p is not None
        and is_significant(r.holm_p)
        for a, r in rows
    )


def behaviour_significant(src: Sources, a: str, b: str, scope: str) -> bool:
    """Whether a - b on `behaviour_correct` in `scope` reaches p_Holm < `SIGNIFICANCE_LEVEL`."""
    row = entry(src.judged_v1.paired, (a, b, "behaviour_correct", scope), PAIRED_FILE)
    return is_significant(row.value("holm_p"))


def _comparison_judged(src: Sources, config: str) -> list[AnswerRecord]:
    """`config`'s judged prompt v1 records of comparison questions."""
    return [
        r
        for r in src.judged_v1.records
        if r.config == config and in_scope(r, COMPARISON) and r.judge is not None
    ]


def _coverage(record: AnswerRecord) -> float:
    """The coverage metric of a judged answer; a judged comparison answer always has one."""
    value = record.judged().metrics.coverage
    if value is None:
        raise ValueError(f"{record.answer_id} has no coverage")
    return value


@dataclass(frozen=True)
class CoverageGroup:
    """The judged comparison answers of one outcome group and their mean coverage.

    `mean` is None when the group is empty: a configuration that never refused,
    or one that refused every question, has no mean for that group.
    """

    count: int
    mean: float | None


@dataclass(frozen=True)
class CoverageByOutcome:
    """A configuration's judged comparison answers, the refusals apart from the rest."""

    refused: CoverageGroup
    answered: CoverageGroup


def coverage_by_outcome(src: Sources, config: str) -> CoverageByOutcome:
    """`config`'s judged comparison answers in the prompt v1 judged run, by outcome.

    Each group's mean is None when the group is empty; a claim renders it as the
    empty marker (`formatting.MISSING`).
    """
    refused: list[float] = []
    answered: list[float] = []
    for r in _comparison_judged(src, config):
        is_refusal = outcome(r.generated().answer) == "refused"
        (refused if is_refusal else answered).append(_coverage(r))

    def group(values: list[float]) -> CoverageGroup:
        return CoverageGroup(len(values), mean(values) if values else None)

    return CoverageByOutcome(refused=group(refused), answered=group(answered))


def refusals_stating_facts(src: Sources, config: str) -> int:
    """`config`'s judged comparison refusals that state some of the gold facts."""
    return sum(
        1
        for r in _comparison_judged(src, config)
        if outcome(r.generated().answer) == "refused" and _coverage(r) > 0
    )


@dataclass(frozen=True)
class AttributionRow:
    """One configuration's misattributed claims under one prompt."""

    config: str
    prompt: str  # `v1` or `v2`
    summary: AttributionSummary


def attribution_rows(src: Sources) -> list[AttributionRow]:
    """Per prompt, then configuration, its misattributed claims."""
    return [
        AttributionRow(config, f"v{prompt}", x)
        for prompt in PROMPTS
        for config, x in attribution(src, prompt).items()
    ]


def _cited(src: Sources, record: AnswerRecord, claim: Claim) -> set[str]:
    """The venues of the chunks one claim of an answer record cites."""
    chunk_ids = record.generated().numbered_chunk_ids()
    return {entry(src.venue_of, chunk_ids[n], CHUNKS_FILENAME) for n in claim.chunks}


@dataclass(frozen=True)
class Misattribution:
    """A claim the judge marks misattributed: its answer id and the venues its chunks are from."""

    answer_id: str
    cited: set[str]


@per_sources
def misattributed_v2(src: Sources) -> list[Misattribution]:
    """Each claim of the prompt v2 judged run that Opus marks misattributed."""
    out = []
    for r in src.judged_v2.records:
        if r.judge is None:
            continue
        for i, verdict in enumerate(r.judge.judgement.claims):
            if verdict.venue_correct:
                continue
            claim = r.generated().answer.claims[i]
            out.append(Misattribution(r.answer_id, _cited(src, r, claim)))
    return out


@dataclass(frozen=True)
class MisattributedQuestions:
    """The misattributed prompt v2 claims and the questions they answer."""

    claims: int
    questions: int
    unanswerable: int  # of the questions, those labeled unanswerable


def misattributed_questions(src: Sources) -> MisattributedQuestions:
    """The misattributed prompt v2 claims and their questions."""
    mis = misattributed_v2(src)
    questions = {split_answer_id(m.answer_id)[1] for m in mis}
    unanswerable = sum(
        1
        for q in questions
        if entry(src.questions, q, "the gold set").effective_category == "unanswerable"
    )
    return MisattributedQuestions(len(mis), len(questions), unanswerable)


@dataclass(frozen=True)
class PartialMisattributed:
    """Prompt v2 answers counted as partial and their misattributed claims."""

    partial: int
    misattributed: int  # of them, those with a misattributed claim
    ccxt_alone: int  # of those, the ones whose misattributed claim cites CCXT alone


def partial_misattributed(src: Sources) -> PartialMisattributed:
    """Prompt v2 answers counted as partial and their misattributed claims."""
    partial = {
        r.answer_id
        for r in src.judged_v2.records
        if r.generation is not None and outcome(r.generation.answer) == "partial"
    }
    mis = misattributed_v2(src)
    with_mis = {m.answer_id for m in mis} & partial
    ccxt = {m.answer_id for m in mis if m.cited == WRAPPERS}
    return PartialMisattributed(len(partial), len(with_mis), len(with_mis & ccxt))


def partial_exchange_citations_ok(src: Sources) -> bool:
    """Whether every claim a prompt v2 partial answer attributes to an exchange cites it.

    It must cite a chunk of that exchange or of CCXT.
    """
    venues = exchanges(src)
    for r in src.v2.records:
        answer = r.generated().answer
        if outcome(answer) != "partial":
            continue
        for claim in answer.claims:
            venue = claim.venue.lower()
            if venue in venues and not _cited(src, r, claim) & ({venue} | WRAPPERS):
                return False
    return True
