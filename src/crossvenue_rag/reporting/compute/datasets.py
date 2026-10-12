"""The gold set the documents report (`run_ids.GOLD`), with the candidate sets before it."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import get_args

from ...config_eval import COMPARISON_SCOPE, MULTI_VENUE_SCOPE
from ...dataset.questions import load_questions
from ...dataset.sampling import read_sample_file
from ...dataset.selection import TARGETS, target_size
from ...models import COMPARISON, VenueCoverage
from ...retrieval.retriever import venue_split_splits
from ...retrieval.venue_detect import DETECTION_OUTCOMES, DetectionOutcome, detection_outcome
from ...stats import P50, percentile
from ..run_ids import (
    CANDIDATES_FINAL,
    COMPARISON_SUPPLEMENT,
    POOL_RUN,
    ROUND2_SAMPLE,
)
from ..sources import Sources
from .corpus import dex_venues
from .corpus_changes import dedup_idmap
from .distinct import shared_value
from .memo import per_sources

# The ranks a shown chunk counts as in the top of (the top 10).
TOP_RANKS = 10

# The venue_coverage values that record a venue the question cannot be answered for.
ONE_SIDED = frozenset(get_args(VenueCoverage)) - {"covered"}
# The categories selection took the same excess of.
EVEN_EXCESS = ("ambiguous", "noisy", "version_sensitive")
# The groups of questions the detection table reports: every comparison
# question, every question naming at least `MULTI_VENUE` venues, and all.
DETECTION_GROUPS = (COMPARISON_SCOPE, MULTI_VENUE_SCOPE, "all")
# The prompt of the comparison supplement in `run_ids.COMPARISON_SUPPLEMENT`.
COMPARISON_SUPPLEMENT_HASH = "e1e2519d848abd35"


def comparison_ids(src: Sources) -> list[str]:
    """Ids of the comparison questions, in dataset order."""
    return [q for q, r in src.questions.items() if r.is_comparison]


def one_sided_comparisons(src: Sources) -> int:
    """Comparison questions with a venue recorded missing_from_pool or absent_in_docs."""
    return sum(
        1 for q in comparison_ids(src) if ONE_SIDED & set(src.questions[q].venue_coverage.values())
    )


def comparison_count_before_augmentation(src: Sources) -> int:
    """Comparison questions of the gold set before augmentation (`v0.2-gold`)."""
    return sum(1 for q in src.questions_before_augmentation.values() if q.is_comparison)


def missing_from_pool_before_augmentation(src: Sources) -> int:
    """Venues of `v0.2-gold`'s comparison questions the annotator recorded missing_from_pool."""
    return sum(
        v == "missing_from_pool"
        for q in src.questions_before_augmentation.values()
        if q.is_comparison
        for v in q.venue_coverage.values()
    )


def migrated_pool(src: Sources, question: str) -> tuple[int, int]:
    """(candidates, distinct texts) of `question`'s round-1 pool before deduplication.

    The pool as the `v0.2-gold` journal migrated it; two candidates hold one text
    when the deduplication id map maps them to the same chunk.

    Raises:
        ValueError: The question's round 1 in `v0.2-gold` was not migrated.
    """
    first = src.rounds_before_augmentation.first(question)
    if first is None or first.migrated_from is None:
        raise ValueError(f"{question}'s round 1 in v0.2-gold was not migrated")
    pool = first.migrated_from.pool
    idmap = dedup_idmap(src)
    return len(pool), len({idmap.get(c, c) for c in pool})


def round2_sample(src: Sources) -> list[str]:
    """The questions of the round-2 re-judging sample, in file order."""
    return read_sample_file(src.path(ROUND2_SAMPLE))


def pool_display(src: Sources) -> int:
    """The fused candidates shown to the annotator per question.

    `eval.pool_display` of the retrieval run (`run_ids.RETRIEVAL_RUN`).
    """
    return src.retrieval.config.eval.pool_display


def relabel_decisions(src: Sources) -> int:
    """Decisions of re-labeling the gold set: each question's `pool_display` candidates."""
    return question_count(src) * pool_display(src)


def target_decisions(src: Sources) -> int:
    """Decisions of labeling a set of `target_size` questions, `pool_display` candidates each."""
    return target_size() * pool_display(src)


def question_count(src: Sources) -> int:
    """Questions in the gold set."""
    return len(src.questions)


def labelled_count(src: Sources) -> int:
    """Questions with at least one gold chunk."""
    return sum(1 for q in src.questions.values() if q.is_labelled)


def split_ids(src: Sources) -> list[str]:
    """Questions venue_split splits on the venues detected in their text, sorted."""
    return sorted(
        q for q, r in src.questions.items() if venue_split_splits(src.detector.detect(r.question))
    )


@dataclass(frozen=True)
class AugmentationCounts:
    """The chunks shown in augmentation and those marked relevant (`datasets.augmentation`)."""

    shown: int
    relevant: int
    shown_top10: int
    relevant_top10: int

    @property
    def relevant_lower(self) -> int:
        """Relevant chunks below rank 10 in the pool run."""
        return self.relevant - self.relevant_top10


def augmentation(src: Sources) -> AugmentationCounts:
    """The chunks of venue_split's and the oracle's never-judged top 20 shown in augmentation.

    `shown` and `relevant`: chunks shown by augmentation and those marked
    relevant; `shown_top10` and `relevant_top10`: of them, those at rank 10 or
    better in the pool run (`run_ids.POOL_RUN`); `relevant_lower`: relevant
    chunks below rank 10.
    """
    best_rank: dict[tuple[str, str], int] = {}
    for r in src.run(POOL_RUN).per_query:
        for rank, cid in enumerate(r.ids, start=1):
            key = (r.question_id, cid)
            best_rank[key] = min(rank, best_rank.get(key, rank))
    shown = relevant = shown_top10 = relevant_top10 = 0
    for qid, entries in src.rounds.questions.items():
        first = entries.first
        for aug in (first.augmentations if first is not None else None) or []:
            for cid in aug.shown:
                # A chunk the pool run did not return is not in its top 10.
                top10 = (qid, cid) in best_rank and best_rank[(qid, cid)] <= TOP_RANKS
                shown += 1
                shown_top10 += top10
                relevant += cid in aug.relevant
                relevant_top10 += top10 and cid in aug.relevant
    return AugmentationCounts(shown, relevant, shown_top10, relevant_top10)


def category_ids(src: Sources, category: str) -> list[str]:
    """The ids of the questions of `category`, in set order."""
    return [q for q, r in src.questions.items() if r.effective_category == category]


def category_count(src: Sources, category: str) -> int:
    """Questions of `category`."""
    return len(category_ids(src, category))


def answer_expected_count(src: Sources) -> int:
    """Questions whose expected behavior is not a refusal."""
    return sum(1 for q in src.questions.values() if not q.expects_refusal)


def refuse_expected_count(src: Sources) -> int:
    """Questions whose expected behavior is a refusal."""
    return sum(1 for q in src.questions.values() if q.expects_refusal)


def unchanged_drafts(src: Sources) -> int:
    """Questions whose text is the draft's in the last candidate set, unchanged."""
    drafts = {d.id: d.question for d in load_questions(src.path(CANDIDATES_FINAL))}
    return sum(1 for q, r in src.questions.items() if drafts.get(q) == r.question)


def median_question_words(src: Sources) -> float:
    """The nearest-rank median length of a question in whitespace-separated words (`stats.P50`).

    Raises:
        ValueError: The gold set has no questions.
    """
    return float(percentile([len(r.question.split()) for r in src.questions.values()], P50))


def category_excess(src: Sources) -> dict[str, int]:
    """Per category of `TARGETS`, questions above its target."""
    counts = Counter(r.effective_category for r in src.questions.values())
    return {c: counts[c] - target for c, target in TARGETS.items()}


def excess_total(src: Sources, excluding: Sequence[str] = ()) -> int:
    """Questions above target over every category but those `excluding` names."""
    return sum(k for c, k in category_excess(src).items() if c not in excluding)


def non_comparison_count(src: Sources) -> int:
    """Questions of every category but comparison."""
    return question_count(src) - len(comparison_ids(src))


def excess_is_as_stated(src: Sources) -> bool:
    """Whether every category meets its target, comparison exactly, and `EVEN_EXCESS` evenly.

    The categories of `EVEN_EXCESS` have one count and one excess.
    """
    excess = category_excess(src)
    return (
        shared_value(category_count(src, c) for c in EVEN_EXCESS) is not None
        and shared_value(excess[c] for c in EVEN_EXCESS) is not None
        and excess[COMPARISON] == 0
        and all(v >= 0 for v in excess.values())
    )


def _naming(src: Sources) -> Counter[str]:
    """Questions naming each venue."""
    return Counter(v for r in src.questions.values() for v in set(r.venues))


def questions_naming(src: Sources, venue: str) -> int:
    """Questions whose `venues` name `venue`."""
    return _naming(src)[venue]


def fewest_naming_other(src: Sources, venue: str) -> int:
    """The fewest questions naming any venue but `venue`; ValueError if none names another."""
    return min(k for v, k in _naming(src).items() if v != venue)


def questions_naming_dex(src: Sources) -> int:
    """Questions naming at least one DEX (`corpus.dex_venues`)."""
    dex = set(dex_venues(src))
    return sum(1 for r in src.questions.values() if set(r.venues) & dex)


def supplement_naming(src: Sources, venue: str) -> tuple[int, int]:
    """(questions naming `venue`, questions) of the comparison supplement."""
    supplement = [
        q
        for q in load_questions(src.path(COMPARISON_SUPPLEMENT))
        if q.prompt_hash == COMPARISON_SUPPLEMENT_HASH
    ]
    return sum(venue in q.venues for q in supplement), len(supplement)


def gold_chunk_total(src: Sources) -> int:
    """Gold chunks over every question, a chunk counted once per question."""
    return sum(len(q.gold_chunk_ids) for q in src.questions.values())


def unsplit_count(src: Sources) -> int:
    """Questions venue_split does not split, answered as plain hybrid_rerank."""
    return question_count(src) - len(split_ids(src))


@per_sources
def detection_outcomes(src: Sources) -> dict[str, DetectionOutcome]:
    """Per question, how the venue detector's venues compare with the named ones.

    One of `DETECTION_OUTCOMES` (`retrieval.venue_detect.detection_outcome`).
    """
    return {
        q: detection_outcome(src.detector.detect(r.question), r.venues)
        for q, r in src.questions.items()
    }


def detection_group(src: Sources, group: str) -> list[str]:
    """The questions of a group of `DETECTION_GROUPS`, in dataset order."""
    if group == COMPARISON_SCOPE:
        return comparison_ids(src)
    if group == MULTI_VENUE_SCOPE:
        return [q for q, r in src.questions.items() if r.is_multi_venue]
    return list(src.questions)


def detection_counts(src: Sources, group: str) -> dict[DetectionOutcome, int]:
    """Questions of `group` per detection outcome, each of `DETECTION_OUTCOMES`, `none` too."""
    found = Counter(detection_outcomes(src)[q] for q in detection_group(src, group))
    return {k: found[k] for k in DETECTION_OUTCOMES}


@dataclass(frozen=True)
class DetectionRow:
    """One group of `DETECTION_GROUPS` by detection outcome."""

    group: str
    questions: int
    counts: dict[DetectionOutcome, int]  # `detection_counts`


def detection_rows(src: Sources) -> list[DetectionRow]:
    """Per group of `DETECTION_GROUPS`, its questions by detection outcome."""
    return [
        DetectionRow(g, len(detection_group(src, g)), detection_counts(src, g))
        for g in DETECTION_GROUPS
    ]


@dataclass(frozen=True)
class DetectionMisses:
    """The questions the detector misses."""

    missed: int
    ambiguous: int  # of them, ambiguous questions
    other: int  # of them, questions of the other categories


def detection_misses(src: Sources) -> DetectionMisses:
    """The questions the detector misses, by category."""
    missed = [q for q, o in detection_outcomes(src).items() if o == "missed"]
    ambiguous = sum(src.questions[q].effective_category == "ambiguous" for q in missed)
    return DetectionMisses(len(missed), ambiguous, len(missed) - ambiguous)


def misses_are_unnamed(src: Sources) -> bool:
    """Whether every ambiguous question is missed and no missed question has a detected venue."""
    missed = [q for q, o in detection_outcomes(src).items() if o == "missed"]
    return detection_misses(src).ambiguous == category_count(src, "ambiguous") and all(
        not src.detector.detect(src.questions[q].question) for q in missed
    )
