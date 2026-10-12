"""What evals/datasheet.md claims about `v0.2.2-gold`, the version that supersedes it."""

from __future__ import annotations

from ... import formatting
from ..check import Doc
from ..compute import agreement
from ..sources import Sources
from .comparisons import round2_sample_text


def adjudicated_version(doc: Doc, src: Sources) -> None:
    """The section on `v0.2.2-gold`: its counts beside this version's, and what changed."""
    new = agreement.adjudicated_gold(src)
    doc.check(
        "`v0.2.2-gold` holds the same {} questions in the same order", formatting.count(len(new))
    )
    old_counts, new_counts = agreement.version_counts(src.questions), agreement.version_counts(new)
    doc.table(
        "| Count | `v0.2.1-gold` | `v0.2.2-gold` |",
        {
            label: [
                formatting.count(getattr(old_counts, field)),
                formatting.count(getattr(new_counts, field)),
            ]
            for label, field in (
                ("questions with gold", "with_gold"),
                ("gold chunks", "gold_chunks"),
                ("fact questions", "fact"),
                ("unanswerable questions", "unanswerable"),
            )
        },
    )
    relabeled = agreement.relabeled_questions(src)
    recategorized = agreement.recategorized(src)
    doc.check(
        "The adjudication decided the {} pairs on which round 1 and round 2 disagree, on {} "
        "questions, and changed the gold of {} of them. The D23 relabel made {} questions fact "
        "and kept {} unanswerable. In all, the gold of {} questions differs from this version. "
        "The {} disagreeing pairs of the {} sampled questions and the D23 questions follow the "
        "amended criterion. Every other label follows the round-1 criterion, among them every "
        "label of the other {} questions",
        formatting.count(agreement.adjudicated_pairs(src)),
        formatting.count(len(agreement.adjudicated_questions(src))),
        formatting.count(len(agreement.adjudication_changed(src))),
        formatting.count(len(recategorized)),
        formatting.count(len(relabeled) - len(recategorized)),
        formatting.count(len(agreement.changed_gold(src))),
        formatting.count(agreement.adjudicated_pairs(src)),
        round2_sample_text(src),
        formatting.count(len(agreement.round1_criterion_questions(src))),
    )
