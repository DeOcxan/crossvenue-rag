"""evals/methodology.md: the deviation rows checked against the labeling records (D33, D35, D36)."""

from __future__ import annotations

from ....dataset.journal import FULL_RELABEL_ROUND
from ....models import NOTES_SHOWN
from ...check import Doc
from ...compute import round3
from ...sources import Sources
from ..round3 import require_cited

# The stem of "enumeration" and "enumerate", the behavior section 4 registers (D33).
ENUMERATION = "enumerat"


def record_deviations(doc: Doc, src: Sources) -> None:
    """D33, D35 and D36: the ambiguous questions, the migrated judgment and the unshown notes."""
    ambiguous = [
        q
        for version in (src.questions, round3.gold(src))
        for q in version.values()
        if q.effective_category == "ambiguous"
    ]
    doc.claim(
        "every ambiguous question expects `ask_clarification`, and its note states",
        holds=all(
            q.expected_behavior == "ask_clarification"
            and ENUMERATION not in (q.notes or "").lower()
            for q in ambiguous
        ),
        what="every ambiguous question of `v0.2.1-gold` and `v0.3-gold` expects "
        "`ask_clarification`, and no note names enumeration",
    )
    journal = round3.journal(src)
    migrated = {
        question: judgment
        for question, entry in journal.questions.items()
        for judgment in entry.rounds.values()
        if judgment.migrated_from is not None
    }
    doc.claim(
        "was made on the corpus before exact deduplication and recorded no `index_version`",
        holds=all(j.as_shown().index_version is None for j in migrated.values()),
        what="every migrated judgment of the `v0.3-gold` journal was made with no `index_version`",
    )
    require_cited(doc, src, list(migrated))
    gold = round3.gold(src)
    unmarked = [
        question
        for question, entry in journal.questions.items()
        if (judgment := entry.judgment(FULL_RELABEL_ROUND)) is not None
        and gold[question].effective_category in NOTES_SHOWN
        and not judgment.relevant
    ]
    require_cited(doc, src, unmarked)
