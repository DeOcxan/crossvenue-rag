"""`crossvenue round3-gold`: the gold set built from round 3 alone (methodology section 5b).

Every field of a question comes from the base version except its label fields, which come
from its round-3 entry. The build refuses an unfinished round and a journal in which any
earlier entry differs from the base version's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import formatting, jsonl, listing
from ..errors import UserError
from .journal import (
    FULL_RELABEL_ROUND,
    Journal,
    QuestionJournal,
    load_rounds,
    read_journal,
    rounds_path,
)
from .questions import load_questions
from .records import QuestionRecord, load_records
from .sampling import cited_words


@dataclass(frozen=True)
class Round3Gold:
    """What the build wrote: the question count and the citations of the flagged questions."""

    written: int
    flagged: list[str]


def _earlier_entries(entries: QuestionJournal) -> dict[str, object]:
    """Everything the journal holds on a question besides its round-3 entry."""
    dumped: dict[str, object] = entries.model_dump()
    return {k: v for k, v in dumped.items() if k != str(FULL_RELABEL_ROUND)}


def require_earlier_entries_unchanged(journal: Journal, frozen: Journal) -> None:
    """Refuses a journal that differs from the frozen one in anything but round 3.

    Raises:
        UserError: A question was added, or removed, or an earlier entry differs.
    """
    ids = sorted(set(journal.questions) | set(frozen.questions))
    changed = [
        q for q in ids if _earlier_entries(journal.get(q)) != _earlier_entries(frozen.get(q))
    ]
    if changed:
        raise UserError(
            f"{formatting.plural(len(changed), 'question')} whose earlier journal entries differ "
            f"from the frozen journal: {listing.first_items(changed)}",
            hint="round 3 writes only its own entries; restore the others from the frozen journal",
        )


def labels_from_round3(record: QuestionRecord, entries: QuestionJournal) -> QuestionRecord:
    """`record` with its label fields replaced by its round-3 entry, validated again."""
    round3 = entries.rounds[FULL_RELABEL_ROUND]
    return QuestionRecord.model_validate(
        record.model_dump()
        | {
            "gold_chunk_ids": entries.resolved_relevant(),
            "venue_coverage": round3.venue_coverage or {},
            "annotator": round3.annotator or "",
            "annotated_at": round3.annotated_at or "",
            "annotation_round": FULL_RELABEL_ROUND,
        }
    )


def build_round3_gold(base: Path, questions_path: str, out: Path) -> Round3Gold:
    """Writes the gold set at `out` from `base` and the journal beside `questions_path`.

    Raises:
        UserError: `out` exists, a question has no round-3 entry, or an earlier journal
            entry differs from the journal frozen with `base`.
    """
    if out.exists():
        raise UserError(f"{out} exists", hint="a dataset version is never rewritten")
    records = load_records(base)
    journal = load_rounds(questions_path)
    open_ids = [r.id for r in records if journal.get(r.id).judgment(FULL_RELABEL_ROUND) is None]
    if open_ids:
        raise UserError(
            f"{formatting.plural(len(open_ids), 'question')} without a round-"
            f"{formatting.own_form(FULL_RELABEL_ROUND)} entry: {listing.first_items(open_ids)}",
            hint="finish the round with `make round3`",
        )
    require_earlier_entries_unchanged(journal, read_journal(rounds_path(base)))
    built = [labels_from_round3(r, journal.get(r.id)) for r in records]
    jsonl.write(out, (r.model_dump() for r in built))
    questions = {q.id: q for q in load_questions(out)}
    flagged = [
        f'{r.id} "{cited_words(questions[r.id])}"'
        for r in records
        if journal.get(r.id).rounds[FULL_RELABEL_ROUND].flagged
    ]
    return Round3Gold(len(built), flagged)
