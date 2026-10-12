"""Test-retest agreement (Cohen's kappa) between two labeling rounds of one annotator.

The unit of agreement is a (question, candidate chunk) pair judged relevant or not.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import formatting, listing
from ..clock import SECONDS_PER_DAY, record_time
from ..errors import UserError
from ..stats import Interval, chance_agreement, cohens_kappa, kappa_value
from .journal import Journal, load_rounds, rounds_path
from .questions import load_questions
from .sampling import read_sample_file

# Landis & Koch (1977): each label holds kappas below its bound; the rest are
# "almost perfect".
LANDIS_KOCH = (
    (0.0, "worse than chance"),
    (0.21, "slight"),
    (0.41, "fair"),
    (0.61, "moderate"),
    (0.81, "substantial"),
)

# Methodology section 5: the second round starts at least 14 days after the
# first, so that the annotator's memory of the first verdicts cannot stand in for the criterion.
MIN_RETEST_DAYS = 14
# Methodology section 5's target, Landis & Koch's "substantial"; below it, kappa is a
# limitation of the version and the criterion is refined.
KAPPA_TARGET = 0.61


@dataclass(slots=True)
class Judgements:
    """One round's verdicts: (`question_id`, `chunk_id`) -> relevant."""

    round_id: int
    annotator: str
    labels: dict[tuple[str, str], bool]


@dataclass(slots=True)
class AgreementReport:
    """Cohen's kappa between two rounds, with the pair counts it is computed from."""

    pairs: int
    both_relevant: int
    both_irrelevant: int
    disagreements: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    questions: int

    @property
    def interpretation(self) -> str:
        """Landis & Koch's label for the kappa.

        The label is a shared vocabulary with conventional, arbitrary cut points.
        """
        return next((label for below, label in LANDIS_KOCH if self.kappa < below), "almost perfect")


def compare_rounds(first: Judgements, second: Judgements) -> AgreementReport:
    """Kappa over the (question, chunk) pairs judged in both rounds.

    A chunk absent from a round's pool is unjudged in that round, so a pair is
    compared only when both rounds showed the chunk.

    Raises:
        UserError: the rounds share no judged pair.
    """
    shared = sorted(set(first.labels) & set(second.labels))
    if not shared:
        raise UserError(
            f"rounds {formatting.own_form(first.round_id)} and "
            f"{formatting.own_form(second.round_id)} share no judged "
            "(question, chunk) pairs",
            hint="compare two rounds that judged the same questions",
        )

    a = [first.labels[key] for key in shared]
    b = [second.labels[key] for key in shared]
    result = cohens_kappa(a, b)

    return AgreementReport(
        pairs=len(shared),
        both_relevant=sum(1 for x, y in zip(a, b, strict=True) if x and y),
        both_irrelevant=sum(1 for x, y in zip(a, b, strict=True) if not x and not y),
        disagreements=sum(1 for x, y in zip(a, b, strict=True) if x != y),
        observed_agreement=result.observed,
        expected_agreement=result.expected,
        kappa=result.kappa,
        questions=len({question_id for question_id, _ in shared}),
    )


def judgements_from_rounds(
    journal: Journal,
    round_id: int,
    annotator: str = "",
    *,
    first_pass: bool = False,
    questions: Collection[str] | None = None,
) -> Judgements:
    """`Judgements` of one round of a question file's journal.

    The labeling tool stores, per round, the candidate pool it displayed and the
    subset marked relevant. Chunks judged later by augmentation are judged too, and
    count (`Judgment.judged_ids`). `first_pass` reads each entry's verdicts before
    review (`Judgment.first_pass_relevant`) instead of the corrected set; `questions`
    restricts the judgments to those question ids.
    """
    labels: dict[tuple[str, str], bool] = {}
    for question_id, entries in journal.questions.items():
        if questions is not None and question_id not in questions:
            continue
        entry = entries.judgment(round_id)
        if entry is None:
            continue
        relevant = entry.first_pass_relevant() if first_pass else set(entry.relevant)
        for chunk_id in entry.judged_ids():
            labels[(question_id, chunk_id)] = chunk_id in relevant
    return Judgements(round_id=round_id, annotator=annotator, labels=labels)


@dataclass(slots=True)
class AgreementVariant:
    """One registered reading of the round-1 against round-2 agreement."""

    reference: str  # "corrected" (registered) or "first pass" (sensitivity)
    scope: str  # "all" (registered) or "unexposed" (sensitivity)
    report: AgreementReport
    interval: Interval | None = None


@dataclass(frozen=True)
class Bootstrap:
    """How a kappa interval is drawn: resamples and seed (`eval.bootstrap_*`)."""

    resamples: int
    seed: int


def kappa_interval(first: Judgements, second: Judgements, bootstrap: Bootstrap) -> Interval:
    """Kappa's interval over questions resampled with replacement.

    Each resample draws as many questions as were compared and pools all their
    shared pairs, a question drawn twice counting twice, as the point estimate
    pools them. Pairs are not independent within a question, so the question is
    the unit drawn. The level is `stats.INTERVAL_LEVEL_PERCENT`; the
    draws follow `evaluation.paired.paired_bootstrap`.
    """
    shared = set(first.labels) & set(second.labels)
    questions = sorted({question_id for question_id, _ in shared})
    # Per question: pairs, relevant in the first round, in the second, agreements.
    counts = np.zeros((len(questions), 4), dtype=np.int64)
    row = {question_id: i for i, question_id in enumerate(questions)}
    for key in shared:
        a, b = first.labels[key], second.labels[key]
        counts[row[key[0]]] += (1, a, b, a == b)
    rng = np.random.default_rng(bootstrap.seed)
    draws = rng.integers(0, len(questions), size=(bootstrap.resamples, len(questions)))
    n, relevant_first, relevant_second, agree = counts[draws].sum(axis=1).T
    expected = chance_agreement(relevant_first / n, relevant_second / n)
    # `stats.cohens_kappa`'s undefined case: one and the same label throughout.
    defined = expected < 1.0
    kappas = kappa_value(agree[defined] / n[defined], expected[defined])
    # A resample is undefined when every pooled pair carried one and the same label.
    return Interval.from_draws(kappas, int((~defined).sum()), bootstrap.resamples)


@dataclass(slots=True)
class VerdictTable:
    """The two rounds' verdicts on a set of shared pairs, as a 2x2 table."""

    both_relevant: int
    first_only: int  # relevant in the first round only
    second_only: int  # relevant in the second round only
    both_irrelevant: int

    @property
    def disagreements(self) -> int:
        """Pairs one round judged relevant and the other not."""
        return self.first_only + self.second_only


def verdict_table(
    first: Judgements, second: Judgements, keep: Mapping[str, Collection[str]] | None = None
) -> VerdictTable:
    """The shared pairs counted by both verdicts.

    `keep`, when given, maps a question id to the chunk ids counted for it;
    other pairs are left out.
    """
    cells = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    for key in set(first.labels) & set(second.labels):
        question_id, chunk_id = key
        if keep is None or chunk_id in keep.get(question_id, ()):
            cells[(first.labels[key], second.labels[key])] += 1
    return VerdictTable(
        cells[(True, True)], cells[(True, False)], cells[(False, True)], cells[(False, False)]
    )


def direction_by_category(
    first: Judgements, second: Judgements, category_of: Mapping[str, str]
) -> list[tuple[str, VerdictTable]]:
    """The 2x2 verdict table per question category, most disagreement first."""
    out = []
    for category in sorted(set(category_of.values())):
        chunks: dict[str, set[str]] = {}
        for question_id, chunk_id in first.labels:
            if category_of.get(question_id) == category:
                chunks.setdefault(question_id, set()).add(chunk_id)
        table = verdict_table(first, second, chunks)
        if table.disagreements + table.both_relevant + table.both_irrelevant:
            out.append((category, table))
    return sorted(out, key=lambda item: -item[1].disagreements)


@dataclass(slots=True)
class SourceChunks:
    """How the two rounds judged the questions' source chunks.

    Round 1 marked a question's source chunk on screen and round 2 hid the
    marker (methodology section 5), so the table shows the marker's effect.
    """

    verdicts: VerdictTable
    unjudged: int  # source chunks of compared questions that a round did not show


def source_chunk_verdicts(
    first: Judgements, second: Judgements, sources: Mapping[str, Collection[str]]
) -> SourceChunks:
    """The verdicts on each compared question's source chunks (`source_chunk_ids`)."""
    shared = set(first.labels) & set(second.labels)
    compared = {question_id for question_id, _ in shared}
    listed = {(q, c) for q in compared for c in sources.get(q, ())}
    return SourceChunks(
        verdicts=verdict_table(first, second, {q: sources.get(q, ()) for q in compared}),
        unjudged=len(listed - shared),
    )


def agreement_variants(
    journal: Journal,
    first: int,
    second: int,
    exposed: Collection[str] = (),
    bootstrap: Bootstrap | None = None,
) -> list[AgreementVariant]:
    """The four agreement readings methodology section 5 registers.

    Round 1 is read as corrected (the gold set) or as first judged, over all
    sampled questions or only those whose gold the answer-rating screen never
    showed (`exposed` lists the rest). With `bootstrap`, each reading carries
    its kappa interval.

    Raises:
        UserError: `exposed` names every question the second round judged.
    """
    later = judgements_from_rounds(journal, second)
    asked = {q for q, _ in later.labels}
    scopes: list[tuple[str, set[str] | None]] = [("all", None)]
    if exposed:
        unexposed = asked - set(exposed)
        if not unexposed:
            raise UserError(
                f"every question judged in round {formatting.own_form(second)} is listed as "
                "exposed, so the unexposed reading has no question to compare",
                hint="the exposed list names the questions whose gold was shown; "
                "check the file given with --exposed",
            )
        scopes.append(("unexposed", unexposed))
    out = []
    for reference, first_pass in (("corrected", False), ("first pass", True)):
        for scope, questions in scopes:
            earlier = judgements_from_rounds(
                journal, first, first_pass=first_pass, questions=questions
            )
            interval = kappa_interval(earlier, later, bootstrap) if bootstrap else None
            out.append(AgreementVariant(reference, scope, compare_rounds(earlier, later), interval))
    return out


def kappa_by_category(
    first: Judgements, second: Judgements, category_of: Mapping[str, str]
) -> list[tuple[str, AgreementReport]]:
    """The agreement restricted to each question category, most disagreement first.

    Methodology section 5 and ADR 0006 define it. A category with no shared pair
    is left out.
    """
    out = []
    for category in sorted(set(category_of.values())):
        keep = {q for q, c in category_of.items() if c == category}
        a = Judgements(first.round_id, first.annotator, _only(first.labels, keep))
        b = Judgements(second.round_id, second.annotator, _only(second.labels, keep))
        if set(a.labels) & set(b.labels):
            out.append((category, compare_rounds(a, b)))
    return sorted(out, key=lambda item: -item[1].disagreements)


def _only(labels: dict[tuple[str, str], bool], questions: set[str]) -> dict[tuple[str, str], bool]:
    return {key: value for key, value in labels.items() if key[0] in questions}


def retest_gaps(journal: Journal, first: int, second: int) -> dict[str, float]:
    """Days between each question's two judgments.

    The earlier judgment ends at its last record: its `annotated_at`, or the last of its
    augmentations, which methodology section 5 counts as round 1. The later one is dated by
    its `annotated_at`.
    """
    gaps = {}
    for question_id, entries in journal.questions.items():
        a, b = entries.judgment(first), entries.judgment(second)
        if a and b and a.annotated_at and b.annotated_at:
            stamps = [a.annotated_at, *(aug.at for aug in a.augmentations or [])]
            end = max(record_time(stamp) for stamp in stamps)
            delta = record_time(b.annotated_at) - end
            gaps[question_id] = delta.total_seconds() / SECONDS_PER_DAY
    return gaps


@dataclass(frozen=True)
class KappaRequest:
    """The `kappa` options.

    They name the rounds, the exposed-question list, and whether a short retest
    gap is reported anyway.
    """

    questions_path: str
    first: int = 1
    second: int = 2
    exposed: Path | None = None
    short_gap: bool = False
    bootstrap: Bootstrap | None = None


@dataclass(frozen=True)
class KappaStudy:
    """The registered kappa reading with its retest gaps and sensitivity readings.

    It also lists the gaps under the minimum and the agreement per category.
    """

    report: AgreementReport
    gaps: dict[str, float]
    short: list[str]
    by_category: list[tuple[str, AgreementReport]]
    variants: list[AgreementVariant]
    direction: VerdictTable
    direction_by_category: list[tuple[str, VerdictTable]]
    sources: SourceChunks


def kappa_study(request: KappaRequest) -> KappaStudy:
    """Cohen's kappa between two rounds and its readings (methodology section 5).

    Raises:
        UserError: The question file has no round history, or a question's two judgments
            are less than `MIN_RETEST_DAYS` apart and `short_gap` is not set.
    """
    path, first, second = request.questions_path, request.first, request.second
    rounds = load_rounds(path)
    if not rounds.questions:
        raise UserError(
            f"no annotation history at {rounds_path(path)}",
            hint="run `crossvenue label` first; round history is written as you judge",
        )
    gaps = retest_gaps(rounds, first, second)
    short = sorted(q for q, days in gaps.items() if days < MIN_RETEST_DAYS)
    if short and not request.short_gap:
        raise UserError(
            f"{formatting.plural(len(short), 'question')} re-judged less than "
            f"{formatting.count(MIN_RETEST_DAYS)} days after "
            f"round {formatting.own_form(first)}: {listing.first_items(short)}",
            hint="methodology section 5 requires the gap; --short-gap reports the kappa "
            "anyway, marked",
        )
    earlier, later = judgements_from_rounds(rounds, first), judgements_from_rounds(rounds, second)
    questions = load_questions(path)
    category_of = {q.id: q.effective_category for q in questions}
    exposed = request.exposed
    listed = read_sample_file(exposed) if exposed is not None and exposed.exists() else []
    return KappaStudy(
        report=compare_rounds(earlier, later),
        gaps=gaps,
        short=short,
        by_category=kappa_by_category(earlier, later, category_of),
        variants=agreement_variants(rounds, first, second, listed, request.bootstrap),
        direction=verdict_table(earlier, later),
        direction_by_category=direction_by_category(earlier, later, category_of),
        sources=source_chunk_verdicts(
            earlier, later, {q.id: q.source_chunk_ids for q in questions}
        ),
    )
