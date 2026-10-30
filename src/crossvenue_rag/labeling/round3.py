"""`crossvenue label --round 3`: every question judged again, blind, under the amended criterion.

Methodology section 5b registers the round. Each question shows the chunks round 1 judged plus
the chunks a file adds, shuffled with a recorded seed, on the round-2 screen with the criterion
above the table. The verdict goes to the journal under round 3; the question file and every
earlier entry stay as they are. A run resumes at the first question with no round-3 entry.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from .. import formatting
from ..clock import utc_now
from ..dataset.journal import (
    FULL_RELABEL_ROUND,
    Journal,
    Judgment,
    load_rounds,
    rounds_path,
    save_rounds,
)
from ..errors import UserError
from ..models import Chunk, Question, Result
from .adjudicate import CRITERION_DATE, CRITERION_SUMMARY
from .judge import RelevanceJudgement, judge
from .pool import ROUND3_SHUFFLE_SEED, candidate_ids, retest_candidates
from .screen import Ask, ScreenView, Show
from .session import progress_table

_KEYS_NOTE = "u re-opens the previous question of this session; a skip there keeps its verdict."
_FLAG_NOTE = (
    "This question is unanswerable: n if no chunk answers it, x if one does, which flags it as "
    "misclassified (methodology section 5a). Numbers are refused."
)


class ReopenPrevious(Exception):
    """The annotator pressed `u`: the previous question is judged again."""


class _Keys:
    """The label prompt with round 3's keys: `u`, and `x` on an unanswerable question."""

    def __init__(self, ask: Ask, console: Console, question: Question, *, can_reopen: bool) -> None:
        self.ask = ask
        self.console = console
        self.question = question
        self.can_reopen = can_reopen
        self.flagged = False

    def __call__(self, message: object, **options: object) -> str:
        """An answer for the label prompt; `u` raises `ReopenPrevious`, `x` answers `n`.

        Raises:
            ReopenPrevious: The annotator pressed `u` where a previous question can be re-opened.
        """
        while True:
            raw = self.ask(message, **options)
            command = raw.strip().lower()
            if command == "u":
                if self.can_reopen:
                    raise ReopenPrevious
                self.console.print("[red]u re-opens the previous question of this session.[/red]")
            elif self.question.is_answerable:
                return raw
            elif command == "x":
                self.flagged = True
                return "n"
            elif command[:1].isdigit():
                self.console.print("[red]Expected n or x on an unanswerable question.[/red]")
            else:
                return raw


@dataclass(frozen=True)
class Round3Target:
    """What the round reads and stamps.

    `questions_path` names the journal it writes; `questions` are the questions as the
    screen shows them, in order; `added` holds, per question, the chunks shown beside
    round 1's.
    """

    questions_path: str
    questions: Sequence[Question]
    added: Mapping[str, Sequence[str]]
    chunks: Mapping[str, Chunk]
    index_version: str
    annotator: str


def round3_candidates(target: Round3Target, rounds: Journal) -> dict[str, list[Result]]:
    """Question id -> its shuffled candidates, every question checked before the first is shown.

    Raises:
        UserError: A question has no round-1 entry, a chunk to show is not in the corpus, or
            `target.added` names a question outside the round.
    """
    ids = {q.id for q in target.questions}
    unknown = sorted(set(target.added) - ids)
    if unknown:
        raise UserError(
            f"the added chunks name questions outside the round: {' '.join(unknown)}",
            hint="the file lists questions of the set, as evals/selection/d23_answer_chunks.jsonl",
        )
    missing = [q.id for q in target.questions if rounds.first(q.id) is None]
    if missing:
        raise UserError(
            f"no round-1 judgment to start from: {' '.join(missing)}",
            hint=f"the journal is {rounds_path(target.questions_path)}",
        )
    return {
        q.id: retest_candidates(
            target.chunks,
            q.id,
            rounds.require_first(q.id),
            ROUND3_SHUFFLE_SEED,
            target.added.get(q.id, ()),
        )
        for q in target.questions
    }


def _record(
    target: Round3Target,
    rounds: Journal,
    question: Question,
    candidates: Sequence[Result],
    judgement: RelevanceJudgement,
    *,
    flagged: bool,
) -> None:
    """Journals one verdict under round 3 and writes the journal.

    `relevant` is kept in candidate order rather than the shuffled order shown: the judge's
    prompt lists gold in that order, so an unchanged gold set stays an identical prompt.
    """
    order = candidate_ids(rounds.require_first(question.id), target.added.get(question.id, ()))
    position = {chunk_id: i for i, chunk_id in enumerate(order)}
    rounds.entry(question.id).rounds[FULL_RELABEL_ROUND] = Judgment(
        annotator=target.annotator,
        annotated_at=utc_now(),
        index_version=target.index_version,
        pool=[r.chunk_id for r in candidates],
        relevant=sorted(judgement.relevant, key=position.__getitem__),
        venue_coverage=judgement.venue_coverage or None,
        shuffle_seed=ROUND3_SHUFFLE_SEED,
        criterion=CRITERION_DATE,
        flagged=True if flagged else None,
    )
    save_rounds(target.questions_path, rounds)


def _notes(console: Console, question: Question) -> Callable[[], None]:
    """What the screen draws between the question and the table: criterion and keys."""

    def draw() -> None:
        console.print(Text(CRITERION_SUMMARY, style="dim"))
        console.print(Text(_KEYS_NOTE, style="dim"))
        if not question.is_answerable:
            console.print(Text(_FLAG_NOTE, style="yellow"))

    return draw


def _recorded(target: Round3Target, rounds: Journal) -> int:
    return sum(1 for q in target.questions if rounds.get(q.id).judgment(FULL_RELABEL_ROUND))


def round3_session(
    target: Round3Target, *, console: Console | None = None, prompt: Ask = Prompt.ask
) -> None:
    """Judges each question without a round-3 entry; Ctrl-C stops, and the next run resumes there.

    Each verdict is written when it is given. `u` at a question re-opens the one before it,
    within this run and without its marks; a skip or Ctrl-C there keeps the verdict it had.
    A skipped question comes back on the next run.

    Raises:
        UserError: The candidates of a question cannot be built (`round3_candidates`).
    """
    console = console or Console()
    rounds = load_rounds(target.questions_path)
    candidates = round3_candidates(target, rounds)
    queue = [q for q in target.questions if not rounds.get(q.id).judgment(FULL_RELABEL_ROUND)]
    total = formatting.count(len(target.questions))
    if not queue:
        console.print(Text(f"Round 3 is complete: {total}/{total} questions recorded."))
        return
    console.print(
        Text(
            f"Round 3: {formatting.plural(len(queue), 'question')} left of {total}, annotator "
            f"{target.annotator}. Records under round 3; nothing else is written.\n"
            "Ctrl-C stops; the next run resumes at the first open question.\n"
        )
    )
    show: Show = console.print
    started = time.monotonic()
    recorded: set[str] = set()
    position = 0
    reopened = False
    try:
        while position < len(queue):
            question = queue[position]
            console.rule(Text(f"{formatting.count(position + 1)}/{formatting.count(len(queue))}"))
            show(
                progress_table(
                    target.questions,
                    queue,
                    position,
                    len(recorded),
                    time.monotonic() - started,
                    labelled=_recorded(target, rounds),
                )
            )
            keys = _Keys(prompt, console, question, can_reopen=position > 0 and not reopened)
            view = ScreenView(blind=True, between=_notes(console, question))
            try:
                judgement = judge(
                    console, question, candidates[question.id], set(), keys, view=view
                )
            except ReopenPrevious:
                position -= 1
                reopened = True
                continue
            if judgement is not None:
                _record(
                    target,
                    rounds,
                    question,
                    candidates[question.id],
                    judgement,
                    flagged=keys.flagged,
                )
                recorded.add(question.id)
                console.print(Text("recorded\n", "green"))
            position += 1
            reopened = False
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: verdicts so far are saved[/yellow]")
    done = formatting.count(_recorded(target, rounds))
    journal = rounds_path(target.questions_path)
    console.print(Text(f"\n{done}/{total} questions recorded under round 3. Journal: {journal}"))
