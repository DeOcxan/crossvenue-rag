"""`crossvenue select --finalize`: the question file from the accepted candidates.

The accepted set is checked against the target composition first; only a set
that meets it is written, and the written file is validated like any dataset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import Config
from ..errors import UserError
from .questions import save_questions
from .records import QuestionRecord, as_question
from .selection import (
    CompositionReport,
    accepted_records,
    check_composition,
    load_decisions,
    rejection_summary,
)
from .validate import ValidationReport, validate_dataset


@dataclass(frozen=True)
class Finalized:
    """The outcome of finalizing a selection.

    The composition of the accepted set and, when it met the targets, the
    questions written, the rejection reasons and the validation of the file.
    """

    composition: CompositionReport
    written: int | None = None
    reasons: Counter[str] | None = None
    validation: ValidationReport | None = None

    def require_composition(self) -> None:
        """Refuses a set short of the target composition; nothing was written.

        Raises:
            UserError: The accepted set does not meet the target composition.
        """
        if not self.composition.ok:
            raise UserError(
                "the accepted set does not meet the target composition",
                hint="run `crossvenue select` again to fill the gaps, or revise the targets "
                "in evals/methodology.md decision 2 and selection.TARGETS together",
            )


def finalize_selection(
    cfg: Config, candidates: Sequence[QuestionRecord], questions_path: str, selection_file: str
) -> Finalized:
    """Writes the accepted candidates to `questions_path` if they meet the targets.

    Raises:
        UserError: `selection_file` records no decisions.
    """
    decisions = load_decisions(selection_file)
    if not decisions:
        raise UserError(
            f"no decisions recorded in {selection_file}",
            hint="run `crossvenue select` first",
        )
    composition = check_composition(candidates, decisions)
    if not composition.ok:
        return Finalized(composition)
    kept = accepted_records(candidates, decisions)
    save_questions(questions_path, [as_question(r) for r in kept])
    return Finalized(
        composition,
        written=len(kept),
        reasons=rejection_summary(candidates, decisions),
        validation=validate_dataset(questions_path, cfg, check_near_duplicates=True),
    )
