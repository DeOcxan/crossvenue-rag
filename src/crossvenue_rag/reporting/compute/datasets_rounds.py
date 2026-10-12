"""The labeling journals: round 1 of `v0.2-gold` and the round-2 re-judging sample.

The seed round 2 shuffles candidates with and the labeling pool's depth limit are read
from the committed file that sets them, since reporting may not import the labeling layer.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from ...clock import record_time
from ...dataset.journal import Judgment
from ...dataset.sampling import read_sample_file
from ...stats import P50, percentile
from ..run_ids import ROUND2_EXPOSED
from ..sources import Sources
from .datasets import round2_sample

# A pace needs one gap besides the session break, which is left out.
MIN_STAMPS = 3
# Where the re-judging shuffle seed and the pool's depth limit are set.
SHUFFLE_SEED = ("src/crossvenue_rag/labeling/pool.py", r"RETEST_SHUFFLE_SEED = (\d+)")
MAX_DEPTH_MULTIPLIER = ("src/crossvenue_rag/labeling/pool.py", r"_MAX_DEPTH_MULTIPLIER = (\d+)")


def _first_before_augmentation(src: Sources) -> dict[str, Judgment]:
    """Round 1's judgment of each question of `v0.2-gold`."""
    journal = src.rounds_before_augmentation
    return {q: e.first for q, e in journal.questions.items() if e.first is not None}


def _committed_int(src: Sources, path: str, pattern: str) -> int:
    """The integer the first group of `pattern` matches in the committed file `path`."""
    match = re.search(pattern, src.path(path).read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{pattern!r} matches nothing in {path}")
    return int(match.group(1))


def retest_shuffle_seed(src: Sources) -> int:
    """The seed round 2 shuffles each question's candidates with."""
    return _committed_int(src, *SHUFFLE_SEED)


def pool_depth_multiplier(src: Sources) -> int:
    """How many times its configured depth a pool method is asked for at most."""
    return _committed_int(src, *MAX_DEPTH_MULTIPLIER)


def question_count_before_augmentation(src: Sources) -> int:
    """Questions of the gold set before augmentation (`v0.2-gold`)."""
    return len(src.questions_before_augmentation)


def first_pass_marks(src: Sources) -> tuple[int, int]:
    """(candidates marked relevant, candidates shown) in round 1 of `v0.2-gold`."""
    entries = list(_first_before_augmentation(src).values())
    return sum(len(e.relevant) for e in entries), sum(len(e.pool) for e in entries)


@dataclass(frozen=True)
class SourceChunkMarks:
    """Round-1 candidates shown and marked, apart for chunks a question was drafted from."""

    source_shown: int
    source_relevant: int
    other_shown: int
    other_relevant: int


def source_chunk_marks(src: Sources) -> SourceChunkMarks:
    """Round-1 candidates of `v0.2-gold` by whether a question was drafted from them.

    `source_shown` and `source_relevant`: shown chunks a question was drafted
    from, and those marked relevant; `other_shown` and `other_relevant`: the
    rest of the pool.
    """
    before = src.questions_before_augmentation
    out = dict.fromkeys(("source_shown", "source_relevant", "other_shown", "other_relevant"), 0)
    for q, entry in _first_before_augmentation(src).items():
        drafted = set(before[q].source_chunk_ids)
        marked = set(entry.relevant)
        for chunk in entry.pool:
            kind = "source" if chunk in drafted else "other"
            out[f"{kind}_shown"] += 1
            out[f"{kind}_relevant"] += chunk in marked
    return SourceChunkMarks(**out)


def exposed_sample_count(src: Sources) -> int:
    """Questions of the round-2 sample whose gold chunks the answer-rating screen showed."""
    return len(set(read_sample_file(src.path(ROUND2_EXPOSED))) & set(round2_sample(src)))


def unexposed_sample_count(src: Sources) -> int:
    """Questions of the round-2 sample whose gold chunks were not shown."""
    return len(round2_sample(src)) - exposed_sample_count(src)


def sample_categories(src: Sources) -> Counter[str | None]:
    """Questions of the round-2 sample per category (0 for a category it lacks)."""
    counts: Counter[str | None] = Counter(src.questions[q].category for q in round2_sample(src))
    return counts


def single_sampled_categories(src: Sources) -> list[str]:
    """The categories with one question in the round-2 sample, sorted."""
    return sorted(c for c, k in sample_categories(src).items() if k == 1 and c is not None)


def only_single_sampled(src: Sources, categories: Sequence[str]) -> bool:
    """Whether `categories`, and no other, have one question in the round-2 sample."""
    return single_sampled_categories(src) == sorted(categories)


def _sample_entries(src: Sources) -> list[Judgment]:
    """The round-1 judgment of each question of the round-2 sample.

    Raises:
        ValueError: A sampled question has no round-1 judgment in the gold set's journal.
    """
    entries = [src.rounds.first(q) for q in round2_sample(src)]
    missing = [q for q, e in zip(round2_sample(src), entries, strict=True) if e is None]
    if missing:
        raise ValueError(f"no round-1 judgment for the sampled {', '.join(missing)}")
    return [e for e in entries if e is not None]


@dataclass(frozen=True)
class SampleDecisions:
    """The decisions round 1 made on the round-2 sample."""

    total: int
    fewest: int  # per question
    most: int  # per question


def sample_decisions(src: Sources) -> SampleDecisions:
    """The decisions round 1 made on the sample."""
    decisions = [len(e.judged_ids()) for e in _sample_entries(src)]
    return SampleDecisions(sum(decisions), min(decisions), max(decisions))


def sample_augmented(src: Sources) -> int:
    """Questions of the sample whose round-1 entry carries augmentation chunks."""
    return sum(bool(e.augmentations) for e in _sample_entries(src))


@dataclass(frozen=True)
class SampleCorrections:
    """Questions of the sample whose round-1 entry carries a correction, by kind."""

    corrected: int  # any correction
    review: int  # at least one an annotator made in review
    derived: int  # only values derived by rule (`crossvenue derive-coverage`)


def sample_corrected(src: Sources) -> SampleCorrections:
    """Questions of the sample whose round-1 entry carries a correction, split by kind."""
    entries = [e for e in _sample_entries(src) if e.corrections]
    review = sum(any(not c.derived for c in e.corrections or []) for e in entries)
    return SampleCorrections(len(entries), review, len(entries) - review)


def sample_correction_changes(src: Sources) -> int:
    """Questions of the sample whose first-pass verdicts differ from the reviewed ones."""
    return sum(e.first_pass_relevant() != set(e.relevant) for e in _sample_entries(src))


@dataclass(frozen=True)
class SaveTimes:
    """The time between consecutive round-1 saves of `v0.2-gold`, the labeling's pace.

    The longest gap is the break between the two sessions and is left out of every gap;
    the longer session is the one after it.
    """

    median_gap: float  # seconds
    longer_saves: int
    longer_span: float  # seconds from the first save of the longer session to its last
    longer_mean_gap: float  # seconds


def save_times(src: Sources) -> SaveTimes:
    """Round 1's pace from the `annotated_at` stamps of the `v0.2-gold` journal.

    Raises:
        ValueError: Fewer than three judgments carry a stamp.
    """
    stamps = sorted(
        record_time(e.annotated_at)
        for e in _first_before_augmentation(src).values()
        if e.annotated_at
    )
    if len(stamps) < MIN_STAMPS:
        raise ValueError("round 1 needs at least three stamped judgments for a pace")
    gaps = [(b - a).total_seconds() for a, b in pairwise(stamps)]
    pause = max(range(len(gaps)), key=gaps.__getitem__)
    longer = max(stamps[: pause + 1], stamps[pause + 1 :], key=len)
    span = (longer[-1] - longer[0]).total_seconds()
    return SaveTimes(
        median_gap=percentile([g for i, g in enumerate(gaps) if i != pause], P50),
        longer_saves=len(longer),
        longer_span=span,
        longer_mean_gap=span / (len(longer) - 1),
    )


@dataclass(frozen=True)
class ReviewCounts:
    """What the review of `v0.2-gold`'s round 1 changed, from its journaled corrections."""

    reopened: int  # questions with a correction an annotator made (not a derived one)
    changed: int  # of those, questions whose relevant set changed
    added: int  # chunks added to a relevant set
    removed: int  # chunks removed from one


def review_counts(src: Sources) -> ReviewCounts:
    """The review's reach, from the corrections the annotator made (derived ones left out).

    A re-opened question is journaled even when its relevant set stays, since review
    records the venue coverage it confirms.
    """
    reopened = changed = added = removed = 0
    for entry in _first_before_augmentation(src).values():
        made = [c for c in entry.corrections or [] if not c.derived]
        if not made:
            continue
        reopened += 1
        before, after = set(made[0].previous_relevant), set(made[-1].relevant)
        changed += before != after
        added += len(after - before)
        removed += len(before - after)
    return ReviewCounts(reopened, changed, added, removed)
