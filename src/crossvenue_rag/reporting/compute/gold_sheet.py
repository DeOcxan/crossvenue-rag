"""The gold set as its datasheet (evals/datasheet.md) describes it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ...config import SCHEMA_DEFAULTS
from ...dataset.questions import load_questions
from ...dataset.selection import load_decisions, selection_path
from ...dataset.validate import longest_shared_run
from ...models import MULTI_VENUE, Question
from ..run_ids import CANDIDATES_FINAL, POOL_RUN
from ..sources import Sources
from .datasets import category_ids, comparison_ids
from .lookup import entry

# The venues a question names that the datasheet counts apart: one, two, three or more.
MANY_VENUES = 3


def answerable_count(src: Sources) -> int:
    """Questions whose answer the corpus holds (`is_answerable`)."""
    return sum(q.is_answerable for q in src.questions.values())


@dataclass(frozen=True)
class CategoryRow:
    """The questions of one category and their gold."""

    questions: int
    answerable: int
    labelled: int  # questions with a gold chunk
    gold_chunks: int


def category_row(src: Sources, category: str) -> CategoryRow:
    """The questions of `category` and their gold."""
    of = [src.questions[q] for q in category_ids(src, category)]
    return CategoryRow(
        len(of),
        sum(q.is_answerable for q in of),
        sum(q.is_labelled for q in of),
        sum(len(q.gold_chunk_ids) for q in of),
    )


def categories(src: Sources) -> list[str]:
    """The categories of the gold set, in order of first appearance."""
    return list(dict.fromkeys(q.effective_category for q in src.questions.values()))


def field_counts(src: Sources, name: str) -> Counter[str]:
    """Questions per value of the field `name` (`expected_behavior`, `difficulty`, `tier`).

    Raises:
        AttributeError: `name` is not a field of `Question`.
    """
    return Counter(str(getattr(q, name)) for q in src.questions.values())


@dataclass(frozen=True)
class NamedVenueCounts:
    """Questions by the number of venues they name."""

    one: int
    two: int
    many: int  # `MANY_VENUES` or more


def named_venue_counts(src: Sources) -> NamedVenueCounts:
    """Questions naming one venue, two, and `MANY_VENUES` or more."""
    sizes = Counter(len(set(q.venues)) for q in src.questions.values())
    return NamedVenueCounts(
        sizes[1], sizes[MULTI_VENUE], sum(n for k, n in sizes.items() if k >= MANY_VENUES)
    )


def shared_run_counts(src: Sources) -> dict[int, int]:
    """Questions with gold per longest run of words shared with one of their gold chunks, by run.

    Spans in backticks or double quotes are cut first, as `validate-evals` does; a run of
    no word counts as one word.
    """
    text = {c.chunk_id: c.text for c in src.chunks}
    runs = Counter(
        max(1, *(len(longest_shared_run(q.question, text.get(g, ""))) for g in q.gold_chunk_ids))
        for q in src.questions.values()
        if q.is_labelled
    )
    return dict(sorted(runs.items()))


def longest_gold_run(src: Sources) -> int:
    """The longest run of words a question shares with one of its gold chunks (at least 1)."""
    return max(shared_run_counts(src))


def answerable_without_gold(src: Sources) -> int:
    """Answerable questions with no gold chunk."""
    return sum(q.is_answerable and not q.is_labelled for q in src.questions.values())


def candidate_count(src: Sources) -> int:
    """Questions of the last candidate set, which holds every earlier one."""
    return len(load_questions(src.path(CANDIDATES_FINAL)))


def decisions(src: Sources) -> Counter[str]:
    """The last selection decision of each question shown, per kind (`accept`, ...)."""
    return Counter(
        d.decision
        for d in load_decisions(src.path(selection_path(SCHEMA_DEFAULTS.eval.questions))).values()
    )


def shown_count(src: Sources) -> int:
    """The candidates shown in the selection session: every decision kind together."""
    return sum(decisions(src).values())


def coverage_values(src: Sources, questions: Sequence[Question] | None = None) -> Counter[str]:
    """`venue_coverage` values per kind over `questions` (default: the gold set)."""
    chosen = src.questions.values() if questions is None else questions
    return Counter(v for q in chosen for v in q.venue_coverage.values())


def coverage_value_count(src: Sources) -> int:
    """The `venue_coverage` values the gold set records."""
    return sum(coverage_values(src).values())


@dataclass(frozen=True)
class CoverageQuestions:
    """The questions that record a `venue_coverage` value."""

    recording: int
    comparisons: int  # of them, comparison questions
    comparison_values: int  # the values the comparison questions record


def coverage_questions(src: Sources) -> CoverageQuestions:
    """The questions that record a `venue_coverage` value."""
    recording = [q for q in src.questions.values() if q.venue_coverage]
    comparisons = [q for q in recording if q.is_comparison]
    return CoverageQuestions(
        len(recording), len(comparisons), sum(coverage_values(src, comparisons).values())
    )


def coverage_outside_comparison(src: Sources) -> list[str]:
    """The category of each question outside comparison that records a `venue_coverage` value."""
    return [
        q.effective_category
        for q in src.questions.values()
        if q.venue_coverage and not q.is_comparison
    ]


def comparisons(src: Sources) -> list[Question]:
    """The comparison questions of the gold set, in dataset order (`datasets.comparison_ids`)."""
    return [src.questions[q] for q in comparison_ids(src)]


def added_gold(src: Sources) -> dict[str, set[str]]:
    """Per question, the gold chunks the gold set has and the set before augmentation lacks."""
    before = src.questions_before_augmentation
    added = {
        q.id: set(q.gold_chunk_ids) - set(before[q.id].gold_chunk_ids if q.id in before else ())
        for q in src.questions.values()
    }
    return {q: chunks for q, chunks in added.items() if chunks}


def added_gold_count(src: Sources) -> int:
    """The gold chunks augmentation added, over every question."""
    return sum(len(chunks) for chunks in added_gold(src).values())


def removed_gold(src: Sources) -> int:
    """Gold chunks of the set before augmentation that the gold set does not have."""
    now = src.questions
    return sum(
        len(set(q.gold_chunk_ids) - set(now[q.id].gold_chunk_ids if q.id in now else ()))
        for q in src.questions_before_augmentation.values()
    )


def _pool_ranks(src: Sources) -> dict[tuple[str, str, str], int]:
    """(question, method, chunk) -> rank of the chunk in the method's list of the pool run."""
    return {
        (r.question_id, r.method, cid): rank
        for r in src.run(POOL_RUN).per_query
        for rank, cid in enumerate(r.ids, start=1)
    }


def added_below(src: Sources, rank: int) -> int:
    """Added gold chunks whose best rank in the pool run is below `rank`."""
    best: dict[tuple[str, str], int] = {}
    for (qid, _, cid), r in _pool_ranks(src).items():
        best[(qid, cid)] = min(r, best.get((qid, cid), r))
    return sum(
        best.get((qid, cid), rank + 1) > rank
        for qid, chunks in added_gold(src).items()
        for cid in chunks
    )


def coverage_changes(src: Sources) -> Counter[tuple[str, str]]:
    """(value before augmentation, value after) of each venue whose value changed."""
    before = src.questions_before_augmentation
    return Counter(
        (before[q.id].venue_coverage.get(v, ""), value)
        for q in src.questions.values()
        if q.id in before
        for v, value in q.venue_coverage.items()
        if before[q.id].venue_coverage.get(v, "") != value
    )


def coverage_changes_to(src: Sources, value: str) -> int:
    """The venue values augmentation changed to `value`."""
    return sum(n for (_, after), n in coverage_changes(src).items() if after == value)


def coverage_change_results(src: Sources) -> set[str]:
    """The values the venue values augmentation changed became."""
    return {after for _, after in coverage_changes(src)}


def changed_venue_rank(src: Sources, venue: str, was: str, method: str) -> int | None:
    """The rank in `method`'s pool-run list of the added gold chunk of `venue`.

    Over the questions whose `venue` coverage was `was` before augmentation and is
    `covered` after it; None unless exactly one such chunk is in the list.
    """
    before = src.questions_before_augmentation
    venue_of = src.venue_of
    ranks = _pool_ranks(src)
    found = [
        ranks[(qid, method, cid)]
        for qid, chunks in added_gold(src).items()
        if entry(before, qid, "the set before augmentation").venue_coverage.get(venue) == was
        and src.questions[qid].venue_coverage.get(venue) == "covered"
        for cid in chunks
        if venue_of.get(cid) == venue and (qid, method, cid) in ranks
    ]
    return found[0] if len(found) == 1 else None


def labelled_range(src: Sources, excluding: str) -> tuple[int, int]:
    """(fewest, most) labeled questions of a category with any, other than `excluding`."""
    sizes = [category_row(src, c).labelled for c in categories(src) if c != excluding]
    labelled = [n for n in sizes if n]
    return min(labelled), max(labelled)
