"""The test-retest agreement of evals/agreement.md, as `crossvenue kappa` computes it.

With it, the disagreements `v0.2.2-gold` resolves. Round 2 is in the working journal and in
the journal frozen with `v0.2.2-gold`; the `v0.2.1-gold` journal (`src.rounds`) ends at
round 1, so it is the reference the working round 1 is held to.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ...dataset.added_chunks import read_added_chunks
from ...dataset.agreement import AgreementVariant, Bootstrap, KappaRequest, KappaStudy, kappa_study
from ...dataset.journal import Journal, Judgment, load_rounds
from ...dataset.questions import load_questions
from ...dataset.sampling import read_sample_file
from ...models import Question
from ..run_ids import (
    D23_ANSWER_CHUNKS,
    GOLD_ADJUDICATED,
    ROUND2_EXPOSED,
    ROUND2_SAMPLE,
    WORKING_QUESTIONS,
)
from ..sources import Sources
from .memo import per_sources


def bootstrap(src: Sources) -> Bootstrap:
    """The resamples and seed of every interval (`eval.bootstrap_*` of the default config)."""
    cfg = src.default_config.eval
    return Bootstrap(cfg.bootstrap_resamples, cfg.bootstrap_seed)


@per_sources
def study(src: Sources) -> KappaStudy:
    """Rounds 1 and 2 of the working journal, with intervals (`dataset.agreement.kappa_study`)."""
    request = KappaRequest(
        str(src.path(WORKING_QUESTIONS)),
        exposed=src.path(ROUND2_EXPOSED),
        bootstrap=bootstrap(src),
    )
    return kappa_study(request)


def registered(src: Sources) -> AgreementVariant:
    """The registered reading: round 1 as corrected, over all sampled questions."""
    return reading(src, "corrected", "all")


def reading(src: Sources, reference: str, scope: str) -> AgreementVariant:
    """The reading of round 1 as `reference` over `scope` (`dataset.agreement.AgreementVariant`).

    Raises:
        ValueError: The study has no reading of round 1 as `reference` over `scope`.
    """
    for v in study(src).variants:
        if (v.reference, v.scope) == (reference, scope):
            return v
    raise ValueError(f"no reading of round 1 as {reference} over {scope}")


def exposed_count(src: Sources) -> int:
    """Questions whose gold chunks the answer-rating screen showed before round 2."""
    return len(read_sample_file(src.path(ROUND2_EXPOSED)))


def round1_is_gold(src: Sources) -> bool:
    """Whether every re-judged question's round-1 entry equals the gold set's journal entry."""
    working = load_rounds(str(src.path(WORKING_QUESTIONS)))
    retested = [q for q, entries in working.questions.items() if entries.judgment(2) is not None]
    return bool(retested) and all(
        _dumped(working.get(q).first) == _dumped(src.rounds.get(q).first) for q in retested
    )


def _dumped(judgment: Judgment | None) -> dict[str, Any] | None:
    """A judgment as its file holds it, so two are compared by content whatever their key order."""
    return None if judgment is None else judgment.model_dump(by_alias=True)


def relevant_counts(src: Sources) -> tuple[int, int]:
    """(pairs relevant in round 1, pairs relevant in round 2) of the registered reading."""
    d = study(src).direction
    return d.both_relevant + d.first_only, d.both_relevant + d.second_only


def sensitivity_spread(src: Sources) -> float:
    """The largest distance of a sensitivity reading's kappa from the registered kappa."""
    kappa = registered(src).report.kappa
    return max(abs(v.report.kappa - kappa) for v in study(src).variants)


def intervals_hold_registered(src: Sources) -> bool:
    """Whether every reading's interval includes the registered kappa."""
    kappa = registered(src).report.kappa
    return all(
        v.interval is not None
        and v.interval.low is not None
        and v.interval.high is not None
        and v.interval.low <= kappa <= v.interval.high
        for v in study(src).variants
    )


def d23_questions(src: Sources) -> list[str]:
    """The questions of deviation D23, as the file listing their answer chunks names them."""
    return list(read_added_chunks(src.path(D23_ANSWER_CHUNKS)))


def outside_sample(src: Sources) -> list[str]:
    """The gold set's questions outside the round-2 sample."""
    sample = set(read_sample_file(src.path(ROUND2_SAMPLE)))
    return [q for q in src.questions if q not in sample]


def round1_criterion_questions(src: Sources) -> list[str]:
    """The questions outside the round-2 sample that are not D23 questions either."""
    d23 = set(d23_questions(src))
    return [q for q in outside_sample(src) if q not in d23]


@dataclass(frozen=True)
class VersionCounts:
    """The counts the datasheet compares between two gold versions."""

    with_gold: int
    gold_chunks: int
    fact: int
    unanswerable: int


def version_counts(questions: dict[str, Question]) -> VersionCounts:
    """Questions with gold, gold chunks, and fact and unanswerable questions."""
    qs = questions.values()
    return VersionCounts(
        with_gold=sum(1 for q in qs if q.gold_chunk_ids),
        gold_chunks=sum(len(q.gold_chunk_ids) for q in qs),
        fact=sum(1 for q in qs if q.category == "fact"),
        unanswerable=sum(1 for q in qs if q.category == "unanswerable"),
    )


@per_sources
def adjudicated_gold(src: Sources) -> dict[str, Question]:
    """`v0.2.2-gold` by question id."""
    return {q.id: q for q in load_questions(src.path(GOLD_ADJUDICATED))}


@per_sources
def adjudicated_journal(src: Sources) -> Journal:
    """The journal frozen with `v0.2.2-gold`."""
    return load_rounds(src.path(GOLD_ADJUDICATED))


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
    journal, added = adjudicated_journal(src), round3_added(src)
    questions: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for question in adjudicated_gold(src).values():
        first = journal.require_first(question.id)
        shown = set(first.judged_ids()) | set(added.get(question.id, ()))
        questions[question.effective_category] += 1
        decisions[question.effective_category] += len(shown)
    order = sorted(decisions, key=lambda category: (-decisions[category], category))
    return {c: Round3Size(questions[c], decisions[c]) for c in order}


def changed_gold(src: Sources) -> list[str]:
    """Questions whose gold differs between `v0.2.1-gold` and `v0.2.2-gold`."""
    new = adjudicated_gold(src)
    return [q for q, old in src.questions.items() if new[q].gold_chunk_ids != old.gold_chunk_ids]


def adjudicated_questions(src: Sources) -> list[str]:
    """Questions with an adjudication entry in the `v0.2.2-gold` journal."""
    journal = adjudicated_journal(src).questions
    return [q for q, entries in journal.items() if entries.adjudicated is not None]


def relabeled_questions(src: Sources) -> list[str]:
    """Questions with a relabel entry in the `v0.2.2-gold` journal."""
    journal = adjudicated_journal(src).questions
    return [q for q, entries in journal.items() if entries.relabeled is not None]


def adjudication_changed(src: Sources) -> list[str]:
    """Adjudicated questions, not relabeled, whose gold changed."""
    relabeled = set(relabeled_questions(src))
    adjudicated = set(adjudicated_questions(src))
    return [q for q in changed_gold(src) if q in adjudicated and q not in relabeled]


def adjudicated_pairs(src: Sources) -> int:
    """Pair decisions recorded in the `v0.2.2-gold` journal."""
    journal = adjudicated_journal(src).questions
    return sum(
        len(adjudication.decisions)
        for entries in journal.values()
        if (adjudication := entries.adjudicated) is not None
    )


def recategorized(src: Sources) -> list[str]:
    """Questions unanswerable in `v0.2.1-gold` and fact in `v0.2.2-gold`."""
    new = adjudicated_gold(src)
    return [
        q
        for q, old in src.questions.items()
        if old.category == "unanswerable" and new[q].category == "fact"
    ]


@dataclass(frozen=True)
class AdjudicationSplit:
    """The adjudicated decisions of `v0.2.2-gold`'s journal against the two rounds."""

    as_first: int  # decisions that agree with round 1 as corrected
    as_second: int  # decisions that agree with round 2


def adjudication_split(src: Sources) -> AdjudicationSplit:
    """How the adjudication decided the pairs rounds 1 and 2 disagree on.

    Raises:
        ValueError: A decided chunk is one both rounds judged alike, so no disagreement.
    """
    as_first = as_second = 0
    for entries in adjudicated_journal(src).questions.values():
        adjudication, first, second = entries.adjudicated, entries.first, entries.judgment(2)
        if adjudication is None or first is None or second is None:
            continue
        for chunk_id, decision in adjudication.decisions.items():
            in_first, in_second = chunk_id in first.relevant, chunk_id in second.relevant
            if in_first == in_second:
                raise ValueError(f"{chunk_id} was judged alike in both rounds")
            as_first += decision.relevant == in_first
            as_second += decision.relevant == in_second
    return AdjudicationSplit(as_first, as_second)
