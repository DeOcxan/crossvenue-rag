"""`venue_coverage` values that follow from what round 1 recorded rather than from judgment.

`covered` follows from a gold chunk of that venue and `missing_from_pool` from a recorded pool
with no candidate from it. `absent_in_docs`, a claim about the whole documentation, and a venue
whose pooled candidates were all left unmarked stay the annotator's call.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from ..clock import utc_now
from ..dataset.journal import Correction, Judgment, load_rounds, save_rounds
from ..dataset.questions import load_questions, save_questions
from ..models import Question, VenueCoverage

RULE = (
    "covered: a gold chunk is from the venue; missing_from_pool: the recorded "
    "round-1 pool holds no candidate from the venue"
)


def derivable(
    question: Question, entry: Judgment, chunk_venues: dict[str, str]
) -> dict[str, VenueCoverage]:
    """The venue_coverage values the rule settles for this question.

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
    """One question's venue_coverage before and after derivation, and the venues left open."""

    question_id: str
    previous: dict[str, VenueCoverage]
    coverage: dict[str, VenueCoverage]
    undecided: list[str]  # venues the rule does not settle


def derive_coverage(
    questions_path: str | Path,
    chunk_venues: dict[str, str],
    index_version: str,
    exclude: Collection[str] = (),
) -> list[Derived]:
    """Fills derivable venue_coverage on labeled answerable comparisons.

    Values already set are kept (an annotator's word is not overwritten by a
    rule); only missing venues are filled. Questions in `exclude` are skipped.
    Every derived value is also journaled in round 1's `corrections` as
    `derived`, with the rule and the previous value, so a reader can tell
    derived coverage from coverage an annotator set. Returns what changed.
    """
    questions = load_questions(questions_path)
    rounds = load_rounds(questions_path)
    now = utc_now()
    changed: list[Derived] = []
    for question in questions:
        labelled = bool(question.gold_chunk_ids or question.annotated_at)
        if not (question.needs_venue_coverage and labelled) or question.id in exclude:
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
                previous,
                coverage,
                [v for v in question.venues if v not in coverage],
            )
        )
    if changed:
        save_questions(questions_path, questions)
        save_rounds(questions_path, rounds)
    return changed
