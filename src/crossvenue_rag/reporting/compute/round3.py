"""Annotation round 3, as its plan, `v0.3-gold`, its journal or a run on that version records it.

The agreement of rounds 1 and 3 is descriptive and holds no kappa (methodology section 5b).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

from ...clock import record_time
from ...dataset.added_chunks import read_added_chunks
from ...dataset.agreement import RoundAgreement, round_agreement
from ...dataset.journal import FULL_RELABEL_ROUND, Journal, candidate_ids, load_rounds
from ...dataset.questions import load_questions
from ...evaluation.results import PerQueryRow
from ...generation.answer_pairs import ConfigComparison
from ...models import Question
from ...stats import P50, percentile
from ..run_ids import D23_ANSWER_CHUNKS, GOLD_ROUND3, ROUND3_RUN, SENSITIVITY_RUN
from ..sources import AnswerRun, Sources
from . import agreement, answers
from .estimates import holm_or_one, same_reading
from .memo import per_sources

# The metrics of a retrieval run that read no gold label.
GOLD_FREE_METRICS = ("venue_coverage", "off_venue", "via_ccxt")
UNJUDGED = "unjudged"


@per_sources
def gold(src: Sources) -> dict[str, Question]:
    """`v0.3-gold` by question id."""
    return {q.id: q for q in load_questions(src.path(GOLD_ROUND3))}


@per_sources
def journal(src: Sources) -> Journal:
    """The journal frozen with `v0.3-gold`."""
    return load_rounds(src.path(GOLD_ROUND3))


@per_sources
def rounds_agreement(src: Sources) -> RoundAgreement:
    """Round 3 against round 1 as corrected, over the pairs both judged."""
    categories = {q.id: q.effective_category for q in gold(src).values()}
    return round_agreement(str(src.path(GOLD_ROUND3)), categories)


@dataclass(frozen=True)
class GoldChange:
    """The gold of `v0.3-gold` against that of `v0.2.2-gold`, in chunks and questions."""

    before: int
    after: int
    added: int
    removed: int
    questions_changed: int
    questions: int


def gold_change(src: Sources) -> GoldChange:
    """The chunks added and removed, and the questions whose gold differs."""
    old, new = agreement.adjudicated_gold(src), gold(src)
    pairs = [(set(old[q].gold_chunk_ids), set(new[q].gold_chunk_ids)) for q in new]
    return GoldChange(
        before=sum(len(a) for a, _ in pairs),
        after=sum(len(b) for _, b in pairs),
        added=sum(len(b - a) for a, b in pairs),
        removed=sum(len(a - b) for a, b in pairs),
        questions_changed=sum(a != b for a, b in pairs),
        questions=len(pairs),
    )


def left_gold(src: Sources) -> list[str]:
    """Answerable questions with gold in `v0.2.2-gold` and none in `v0.3-gold`."""
    old, new = agreement.adjudicated_gold(src), gold(src)
    return [q for q in new if old[q].is_labelled and not new[q].is_labelled]


def joined_gold(src: Sources) -> list[str]:
    """Questions with no gold in `v0.2.2-gold` and gold in `v0.3-gold`."""
    old, new = agreement.adjudicated_gold(src), gold(src)
    return [q for q in new if new[q].is_labelled and not old[q].is_labelled]


def stayed_empty(src: Sources) -> list[str]:
    """Answerable questions with no gold in either version."""
    old, new = agreement.adjudicated_gold(src), gold(src)
    return [
        q
        for q in new
        if new[q].effective_category != "unanswerable"
        and not new[q].is_labelled
        and not old[q].is_labelled
    ]


def flagged(src: Sources) -> list[str]:
    """Questions round 3 flagged as misclassified."""
    frozen = journal(src)
    return [q for q in gold(src) if frozen.get(q).rounds[FULL_RELABEL_ROUND].flagged]


@dataclass(frozen=True)
class Round3Size:
    """The questions and candidate decisions round 3 holds for one category."""

    questions: int
    decisions: int


def round3_added(src: Sources) -> dict[str, list[str]]:
    """Per question, the chunks round 3 shows beside the ones round 1 judged."""
    added = read_added_chunks(src.path(D23_ANSWER_CHUNKS))
    return {q: chunks for q, chunks in added.items() if chunks}


def round3_sizes(src: Sources) -> dict[str, Round3Size]:
    """Questions and decisions by category of `v0.2.2-gold`, most decisions first."""
    journal, added = agreement.adjudicated_journal(src), round3_added(src)
    questions: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for question in agreement.adjudicated_gold(src).values():
        first = journal.require_first(question.id)
        questions[question.effective_category] += 1
        decisions[question.effective_category] += len(
            candidate_ids(first, added.get(question.id, ()))
        )
    order = sorted(decisions, key=lambda category: (-decisions[category], category))
    return {c: Round3Size(questions[c], decisions[c]) for c in order}


@dataclass(frozen=True)
class Pace:
    """One labeling round's pace, from the `annotated_at` of its journal entries.

    A gap is the time between two consecutive entries. It belongs to the later entry, whose
    pool as shown (`Judgment.as_shown`) gives its candidates, so `questions` entries give one
    gap fewer.
    """

    questions: int
    decisions: int
    session_seconds: float
    longest_gap_seconds: float
    per_question: float
    per_decision: float


def pace(src: Sources, round_number: int) -> Pace:
    """The session length and the median seconds of a round in the frozen journal."""
    entries = []
    for question in gold(src):
        judgment = journal(src).get(question).judgment(round_number)
        if judgment is not None and judgment.annotated_at is not None:
            entries.append((record_time(judgment.annotated_at), len(judgment.as_shown().pool)))
    entries.sort()
    gaps = [(b[0] - a[0]).total_seconds() for a, b in pairwise(entries)]
    per_decision = [gap / entry[1] for gap, entry in zip(gaps, entries[1:], strict=True)]
    return Pace(
        questions=len(entries),
        decisions=sum(size for _, size in entries),
        session_seconds=(entries[-1][0] - entries[0][0]).total_seconds(),
        longest_gap_seconds=max(gaps),
        per_question=percentile(gaps, P50),
        per_decision=percentile(per_decision, P50),
    )


def _rows(src: Sources, run: str) -> dict[tuple[str, str], PerQueryRow]:
    return {(r.question_id, r.method): r for r in src.run(run).per_query}


def _cutoffs(src: Sources) -> tuple[int, ...]:
    return tuple(src.run(ROUND3_RUN).config.eval.k_values)


def _gold_free_cells(row: PerQueryRow, ks: tuple[int, ...]) -> list[float | None]:
    return [row.number(f"{metric}@{k}") for metric in GOLD_FREE_METRICS for k in ks]


def gold_free_equal(src: Sources) -> bool:
    """Whether every list and gold-free metric of the run equals the `v0.2.2-gold` run's."""
    old, new, ks = _rows(src, SENSITIVITY_RUN), _rows(src, ROUND3_RUN), _cutoffs(src)
    return old.keys() == new.keys() and all(
        old[key].retrieved_ids == row.retrieved_ids
        and _gold_free_cells(old[key], ks) == _gold_free_cells(row, ks)
        for key, row in new.items()
    )


def shared_gold_questions(src: Sources) -> list[str]:
    """Questions with gold in both `v0.2.2-gold` and `v0.3-gold`."""
    old, new = agreement.adjudicated_gold(src), gold(src)
    return [q for q in new if old[q].is_labelled and new[q].is_labelled]


def unjudged_equal(src: Sources) -> bool:
    """Whether `unjudged@k` agrees per question and method on the shared gold questions."""
    old, new, ks = _rows(src, SENSITIVITY_RUN), _rows(src, ROUND3_RUN), _cutoffs(src)
    shared = set(shared_gold_questions(src))
    return all(
        old[key].number(f"{UNJUDGED}@{k}") == row.number(f"{UNJUDGED}@{k}")
        for key, row in new.items()
        if key[0] in shared
        for k in ks
    )


def unjudged_means_differ(src: Sources) -> int:
    """Cells of `unjudged@k` in summary.csv that differ between the two runs."""
    old, new, ks = src.run(SENSITIVITY_RUN).summary, src.run(ROUND3_RUN).summary, _cutoffs(src)
    return sum(
        old[method].value(f"{UNJUDGED}@{k}") != row.value(f"{UNJUDGED}@{k}")
        for method, row in new.items()
        for k in ks
    )


def unjudged_mean_cells(src: Sources) -> int:
    """Cells of `unjudged@k` in summary.csv: one per method and cutoff."""
    return len(src.run(ROUND3_RUN).summary) * len(_cutoffs(src))


def answer_primary_row(src: Sources, run: AnswerRun) -> ConfigComparison:
    """The primary answer comparison's `coverage` row of `run`.

    Raises:
        KeyError: `run` lacks the row.
    """
    return run.paired[(*answers.primary_pair(src), "coverage", "comparison")]


def answer_changes(
    old: AnswerRun, new: AnswerRun
) -> list[tuple[ConfigComparison, ConfigComparison]]:
    """Each primary or secondary row whose conclusion differs, with its row in `new`.

    Raises:
        KeyError: `new` lacks a comparison of `old`.
    """
    pairs = [(row, new.paired[key]) for key, row in old.paired.items() if row.role != "descriptive"]
    return [(a, b) for a, b in pairs if not same_reading(a, b, holm_or_one(a), holm_or_one(b))]


@dataclass(frozen=True)
class Rejudged:
    """The answers a run judged anew, and those on which faithfulness or venue attribution changed.

    `answers` counts the answers judged anew, each with a verdict in both runs; `first_verdicts`
    counts the answers whose judge call failed in the earlier run and that have a verdict in
    this one.
    """

    answers: int
    faithfulness: int
    attribution: int
    first_verdicts: int


def rejudged(old: AnswerRun, new: AnswerRun) -> Rejudged:
    """What moved on the answers `new` judged anew, and the answers it gave a first verdict.

    An answer is judged anew when its judge call differs from `old`'s.

    Raises:
        KeyError: `old` lacks an answer of `new`.
    """
    earlier = {(r.question_id, r.config): r.judge for r in old.records}
    pairs = [(earlier[(r.question_id, r.config)], r.judge) for r in new.records]
    first_verdicts = sum(a is None and b is not None for a, b in pairs)
    anew = [
        (a, b)
        for a, b in pairs
        if a is not None and b is not None and a.call.cache_key != b.call.cache_key
    ]
    return Rejudged(
        answers=len(anew),
        faithfulness=sum(a.metrics.faithfulness != b.metrics.faithfulness for a, b in anew),
        attribution=sum(
            a.metrics.venue_attribution_error != b.metrics.venue_attribution_error for a, b in anew
        ),
        first_verdicts=first_verdicts,
    )


def tested_answer_rows(run: AnswerRun) -> int:
    """The primary and secondary rows of a judged run's paired.csv."""
    return sum(row.role != "descriptive" for row in run.paired.values())


def new_judge_cost(run: AnswerRun) -> float:
    """USD of the judge calls a judged run made, those its cache did not serve."""
    calls = [r.judge.call for r in run.records if r.judge is not None]
    return sum(c.paid_usd() for c in calls if not c.cached)
