"""`venue_coverage` values that follow from what round 1 recorded rather than from judgment.

`covered` follows from a gold chunk of that venue and `missing_from_pool` from a recorded pool
with no candidate from it. `absent_in_docs`, a claim about the whole documentation, and a venue
whose pooled candidates were all left unmarked stay the annotator's call.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..clock import utc_now
from ..dataset.journal import Correction, Journal, Judgment, load_rounds, save_rounds
from ..dataset.questions import load_questions, save_questions
from ..models import Question, VenueCoverage

RULE = (
    "covered: a gold chunk is from the venue; missing_from_pool: the recorded "
    "round-1 pool holds no candidate from the venue"
)


def derivable(
    question: Question, entry: Judgment, chunk_venues: dict[str, str]
) -> dict[str, VenueCoverage]:
    """The `venue_coverage` values the rule settles for this question.

    The result can cover fewer venues than the question names.
    """
    gold = {chunk_venues.get(c) for c in question.gold_chunk_ids}
    pool = {chunk_venues.get(c) for c in entry.pool}
    coverage: dict[str, VenueCoverage] = {}
    for venue in question.venues:
        if venue in gold:
            coverage[venue] = "covered"
        elif venue not in pool:
            coverage[venue] = "missing_from_pool"
    return coverage


@dataclass(slots=True)
class Derived:
    """One question's `venue_coverage` after derivation, and the venues left open."""

    question_id: str
    coverage: dict[str, VenueCoverage]
    undecided: list[str]  # venues with no value after derivation


def derive_coverage(
    questions_path: str | Path,
    chunk_venues: dict[str, str],
    index_version: str,
    exclude: Collection[str],
) -> list[Derived]:
    """Fills derivable `venue_coverage` in a question file and its journal (`fill_derivable`).

    Returns what changed; the files are written only when something did.

    Raises:
        UserError: Nothing is at `questions_path`, or a directory is.
    """
    questions = load_questions(questions_path)
    rounds = load_rounds(questions_path)
    changed = fill_derivable(questions, rounds, chunk_venues, index_version, exclude, now=utc_now())
    if changed:
        save_questions(questions_path, questions)
        save_rounds(questions_path, rounds)
    return changed


def fill_derivable(
    questions: Sequence[Question],
    rounds: Journal,
    chunk_venues: dict[str, str],
    index_version: str,
    exclude: Collection[str],
    *,
    now: str,
) -> list[Derived]:
    """Fills derivable `venue_coverage` on judged answerable comparisons, in place.

    Values already set are kept (an annotator's word is not overwritten by a
    rule); only missing venues are filled. Questions in `exclude` are skipped.
    Every derived value is also journaled in round 1's `corrections` as
    `derived`, with the rule, the previous value and the time `now`, so a reader
    can tell derived coverage from coverage an annotator set. Returns what changed.
    """
    changed: list[Derived] = []
    for question in questions:
        if not (question.needs_venue_coverage and question.is_judged) or question.id in exclude:
            continue
        entry = rounds.first(question.id)
        if entry is None:
            continue
        previous = dict(question.venue_coverage)
        coverage = {**derivable(question, entry, chunk_venues), **previous}
        coverage = {v: coverage[v] for v in question.venues if v in coverage}
        if coverage == previous:
            continue
        correction = Correction(
            at=now,
            derived=True,
            rule=RULE,
            index_version=index_version,
            previous_relevant=list(entry.relevant),
            previous_venue_coverage=previous,
            relevant=list(entry.relevant),
            venue_coverage=coverage,
        )
        entry.corrections = [*(entry.corrections or []), correction]
        entry.venue_coverage = coverage
        question.venue_coverage = coverage
        changed.append(
            Derived(
                question.id,
                coverage,
                [v for v in question.venues if v not in coverage],
            )
        )
    return changed
