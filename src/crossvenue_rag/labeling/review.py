"""Corrections to round 1: `crossvenue label --review`.

A change replaces round 1's verdict and keeps the previous set under `corrections` in the
round's journal entry. A second look with the first verdict on screen is not a round 2: it
measures nothing about agreement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from .. import formatting
from ..clock import utc_now
from ..config import Config
from ..dataset.journal import Correction, Journal, Judgment, load_rounds, rounds_path, save_rounds
from ..dataset.questions import load_questions, require_label_versions, save_questions
from ..errors import UserError
from ..models import Question
from ..retrieval.protocols import LabelRetriever
from .judge import RelevanceJudgement, judge
from .pool import review_pool
from .screen import Ask
from .session import LabelTarget


def review_session(
    cfg: Config,
    retriever: LabelRetriever,
    target: LabelTarget,
    ids: Sequence[str],
    *,
    console: Console | None = None,
    prompt: Ask = Prompt.ask,
) -> int:
    """Re-opens labeled questions and records corrections to round 1.

    Each question is shown on the pool round 1 recorded, with the current marks.
    Returns how many questions changed. Unchanged questions write nothing.

    Raises:
        ValueError: `target` carries no `index_version`.
        UserError: An id is not in the question file, or round 1 never judged it.
    """
    console = console or Console()
    questions_path, index_version = target.questions_path, target.index_version
    if not index_version:
        raise ValueError("review_session needs the current index_version")
    require_label_versions(questions_path, index_version, all_rounds=True)
    annotator = target.who()
    questions = load_questions(questions_path)
    rounds = load_rounds(questions_path)
    by_id = {q.id: q for q in questions}
    _check_reviewable(ids, by_id, rounds, questions_path)

    console.print(
        Text.assemble(
            (formatting.plural(len(ids), "question"), "bold"),
            " to review, annotator ",
            (annotator, "bold"),
            ". Changes are recorded as corrections to round 1. Enter skips without change.\n",
        )
    )
    changed = 0
    try:
        for position, question_id in enumerate(ids, start=1):
            question = by_id[question_id]
            entry = rounds.require_first(question_id)
            console.rule(
                Text(
                    f"review {formatting.count(position)}/{formatting.count(len(ids))}  "
                    f"{question_id}"
                )
            )
            pool = review_pool(
                retriever,
                question,
                entry,
                cfg.eval.pool_depth_per_method,
                cfg.eval.pool_display,
            )
            judgement = judge(console, question, pool, set(question.gold_chunk_ids), prompt)
            if judgement is None:
                console.print("[dim]skipped, unchanged[/dim]\n")
            elif record_correction(question, entry, judgement, annotator, index_version):
                changed += 1
                save_questions(questions_path, questions)
                save_rounds(questions_path, rounds)
                console.print(
                    Text(
                        f"corrected: {formatting.plural(len(judgement.relevant), 'chunk')}\n",
                        "green",
                    )
                )
            else:
                console.print("[dim]same as before, nothing written[/dim]\n")
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: corrections so far are saved[/yellow]")
    console.print(
        Text(
            f"{formatting.plural(changed, 'question')} corrected. "
            f"Journal: {rounds_path(questions_path)}"
        )
    )
    return changed


def _check_reviewable(
    ids: Sequence[str],
    by_id: Mapping[str, Question],
    rounds: Journal,
    path: str,
) -> None:
    """Refuses ids the question file lacks or round 1 never judged."""
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise UserError(f"not in {path}: {', '.join(unknown)}")
    unlabelled = [i for i in ids if rounds.first(i) is None]
    if unlabelled:
        raise UserError(
            f"no round-1 judgment to review for: {', '.join(unlabelled)}",
            hint="label them first with `crossvenue label`",
        )


def record_correction(
    question: Question,
    entry: Judgment,
    judgement: RelevanceJudgement,
    annotator: str,
    index_version: str,
) -> bool:
    """Applies a review verdict to round 1, keeping the previous one; False if unchanged."""
    previous_relevant = list(entry.relevant)
    previous_coverage = dict(
        entry.venue_coverage if entry.venue_coverage is not None else question.venue_coverage
    )
    if (
        sorted(previous_relevant) == sorted(judgement.relevant)
        and previous_coverage == judgement.venue_coverage
    ):
        return False
    correction = Correction(
        at=utc_now(),
        annotator=annotator,
        index_version=index_version,
        previous_relevant=previous_relevant,
        previous_venue_coverage=previous_coverage,
        relevant=list(judgement.relevant),
        venue_coverage=dict(judgement.venue_coverage),
    )
    entry.corrections = [*(entry.corrections or []), correction]
    entry.relevant = list(judgement.relevant)
    entry.venue_coverage = dict(judgement.venue_coverage) if judgement.venue_coverage else None
    question.gold_chunk_ids = list(judgement.relevant)
    question.venue_coverage = dict(judgement.venue_coverage)
    return True
