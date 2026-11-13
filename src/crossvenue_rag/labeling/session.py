"""A labeling session: walking the question file round by round.

Round 1 writes `gold_chunk_ids` into the question file; a later round re-judges judged questions for
intra-rater agreement and writes only to the sidecar, so it cannot overwrite the labels the
metrics were computed from.
"""

from __future__ import annotations

import os
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .. import formatting
from ..clock import utc_now
from ..config import Config
from ..dataset.journal import FIRST_ROUND, Judgment, load_rounds, rounds_path, save_rounds
from ..dataset.questions import load_questions, require_label_versions, save_questions
from ..dataset.sampling import stratified_sample
from ..errors import UserError
from ..models import Question, Result
from ..retrieval.protocols import LabelRetriever
from .judge import RelevanceJudgement, judge
from .pool import RETEST_SHUFFLE_SEED, gather_candidates, retest_candidates
from .screen import Ask, ScreenView, Show

# Below this many questions moved past (skips included), the time-left estimate is marked rough.
PACE_SAMPLES = 5


def progress_table(
    questions: Sequence[Question],
    queue: Sequence[Question],
    handled: int,
    judged: int,
    elapsed_seconds: float,
    judged_overall: int | None,
) -> Table:
    """Where the session stands, and how much longer finishing would take.

    The decision made at the keyboard is whether to stop or push on. That needs the remaining
    count, the shape of what is left, and an estimate from this session's pace:
    a prior average misleads when this batch is all hard comparison questions.

    `handled` counts questions moved past (a skip still costs reading time, so it
    belongs in the pace); `judged` counts the ones recorded. `judged_overall` is the
    count behind "judged overall": None counts the questions with a verdict, and a later
    round passes the ones it has judged.
    """
    if judged_overall is None:
        judged_overall = sum(1 for q in questions if q.is_judged)
    remaining = len(queue) - handled

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row(
        "judged overall",
        Text(f"{formatting.count(judged_overall)}/{formatting.count(len(questions))}"),
    )
    table.add_row(
        "this session",
        Text(
            f"{formatting.count(judged)} recorded, {formatting.count(handled - judged)} skipped, "
            f"{formatting.count(remaining)} left in queue"
        ),
    )

    if handled and elapsed_seconds > 0:
        pace = elapsed_seconds / handled
        table.add_row("pace", Text(f"{formatting.duration(pace)} per question"))
        table.add_row(
            "estimated remaining",
            Text.assemble(
                formatting.duration(pace * remaining),
                ("" if handled >= PACE_SAMPLES else "  (few samples; rough)", "dim"),
            ),
        )
    else:
        table.add_row("estimated remaining", "[dim]needs one question first[/dim]")

    # What is left, not what is done: the remaining shape decides whether to keep
    # going, since ten more `fact` items is a different evening from ten more
    # `multihop`.
    left = Counter(q.effective_category for q in queue[handled:])
    if left:
        table.add_row(
            "remaining by category",
            Text(
                "  ".join(
                    f"{name}={formatting.count(count)}" for name, count in sorted(left.items())
                )
            ),
        )
    return table


def default_annotator() -> str:
    """The annotator name from CROSSVENUE_ANNOTATOR, else USER, else `unknown`."""
    return os.environ.get("CROSSVENUE_ANNOTATOR") or os.environ.get("USER") or "unknown"


@dataclass(frozen=True)
class QueueSpec:
    """Which questions a round covers: `select_queue` reads it."""

    annotation_round: int
    only_unjudged: bool
    sample: int | None
    seed: int
    sample_ids: Sequence[str] | None


@dataclass(frozen=True)
class LabelTarget:
    """What every labeling session writes to and stamps on its judgments.

    The question file, the index the candidates come from, and who judges
    (empty: `default_annotator()`).
    """

    questions_path: str
    index_version: str
    annotator: str

    def who(self) -> str:
        """The annotator recorded on every judgment."""
        return self.annotator or default_annotator()


@dataclass(frozen=True)
class LabelPlan:
    """One labeling session, built once by the command that starts it.

    `limit` defaults to the config's pool geometry when `None`.
    """

    target: LabelTarget
    queue: QueueSpec
    per_method: int
    limit: int | None


def select_queue(questions: Sequence[Question], spec: QueueSpec) -> list[Question]:
    """Which questions this round will cover.

    Round 1 walks the unjudged ones, or every question with `only_unjudged` off;
    when `sample` is smaller than that set, a random `sample` of it. A later round
    ignores `only_unjudged` and covers judged ones only, since re-judging a question
    never judged measures nothing. It covers the ids of a fixed sample file
    when one is given, a subsample stratified by category (`dataset.sampling`) when
    `sample` is smaller than the judged set, and otherwise every judged question.

    Raises:
        UserError: `spec.sample_ids` names a question round 1 never judged, or a later
            round's `spec.sample` is below the number of judged categories.
    """
    if spec.annotation_round > 1:
        return _retest_queue(questions, spec)
    queue = [q for q in questions if not q.is_judged] if spec.only_unjudged else list(questions)
    if spec.sample and spec.sample < len(queue):
        queue = random.Random(spec.seed).sample(queue, spec.sample)
    return queue


def _retest_queue(questions: Sequence[Question], spec: QueueSpec) -> list[Question]:
    """The judged questions a later round re-judges: a sample file's, a stratified sample or all."""
    pool = [q for q in questions if q.is_judged]
    if spec.sample_ids is not None:
        return _sample_file_queue(pool, spec.sample_ids)
    if spec.sample and spec.sample < len(pool):
        pool = stratified_sample(pool, spec.sample, spec.seed)
    return pool


def _sample_file_queue(pool: Sequence[Question], sample_ids: Sequence[str]) -> list[Question]:
    """The judged questions a sample file names; refuses one naming an unjudged question."""
    judged = {q.id for q in pool}
    missing = [i for i in sample_ids if i not in judged]
    if missing:
        raise UserError(
            f"sample file names questions round 1 never judged: {' '.join(missing)}",
            hint="regenerate it with `crossvenue round-sample`",
        )
    wanted = set(sample_ids)
    return [q for q in pool if q.id in wanted]


class _Journal:
    """The question file and its round journal: the session's only writer."""

    def __init__(self, target: LabelTarget, annotation_round: int) -> None:
        self.path = target.questions_path
        self.round = annotation_round
        self.writes_gold = self.round == FIRST_ROUND
        self.annotator = target.who()
        self.index_version = target.index_version
        self.questions = load_questions(self.path)
        self.rounds = load_rounds(self.path)

    def record(
        self, question: Question, pool: Sequence[Result], judgement: RelevanceJudgement
    ) -> None:
        """Journals one judgment; round 1 also writes it into the question as gold.

        The entry records the pool shown as well as the subset marked relevant,
        because a chunk never displayed is unjudged rather than judged-irrelevant,
        which Cohen's kappa depends on.
        """
        now = utc_now()
        rounds = self.rounds.entry(question.id).rounds
        rounds[self.round] = Judgment(
            annotator=self.annotator,
            annotated_at=now,
            index_version=self.index_version,
            pool=[r.chunk_id for r in pool],
            relevant=judgement.relevant,
            venue_coverage=judgement.venue_coverage or None,
            shuffle_seed=None if self.writes_gold else RETEST_SHUFFLE_SEED,
        )
        if self.writes_gold:
            question.gold_chunk_ids = judgement.relevant
            question.venue_coverage = judgement.venue_coverage
            question.annotator = self.annotator
            question.annotated_at = now
            question.annotation_round = self.round
            save_questions(self.path, self.questions)
        save_rounds(self.path, self.rounds)

    def save(self) -> None:
        """Writes what the session holds, as on an interrupt."""
        if self.writes_gold:
            save_questions(self.path, self.questions)
        save_rounds(self.path, self.rounds)

    def judged_overall(self) -> int | None:
        """The questions a later round has judged; None in round 1, which counts its verdicts."""
        if self.writes_gold:
            return None
        return self.rounds.judged_in(self.round, (q.id for q in self.questions))

    def candidates(
        self, retriever: LabelRetriever, question: Question, per_method: int, limit: int
    ) -> tuple[list[Result], set[str]]:
        """The pool to show and the chunks already marked in it.

        A later round judges exactly what round 1 judged, shuffled, on a blind
        screen: test-retest reliability measures independent judgment of the
        same material.

        Raises:
            UserError: A later round reaches a question round 1 did not judge.
        """
        if self.writes_gold:
            candidates = gather_candidates(retriever, question, per_method, limit)
            return candidates, set(question.gold_chunk_ids)
        first = self.rounds.first(question.id)
        if first is None:
            raise UserError(
                f"{question.id}: no round-1 journal entry to re-judge",
                hint=f"label it in round 1 first; the journal is {rounds_path(self.path)}",
            )
        return retest_candidates(retriever.by_id, question.id, first, RETEST_SHUFFLE_SEED), set()


def _announce(console: Console, queue: Sequence[Question], journal: _Journal) -> None:
    mode = (
        Text("Writes gold labels.")
        if journal.writes_gold
        else Text("Re-annotation: records judgments only, round 1 gold stays intact.", "yellow")
    )
    console.print(
        Text.assemble(
            (formatting.plural(len(queue), "question"), "bold"),
            ", round ",
            (formatting.own_form(journal.round), "bold"),
            ", annotator ",
            (journal.annotator, "bold"),
            ". ",
            mode,
            "\nCtrl-C saves and exits.\n",
        )
    )


def label_session(
    cfg: Config,
    retriever: LabelRetriever,
    plan: LabelPlan,
    *,
    console: Console | None = None,
    prompt: Ask = Prompt.ask,
) -> None:
    """Walks the question file, shows pooled candidates and records judgments.

    Every judgment is stamped with `plan.target.index_version`, and the session
    refuses to start if any judgment on record was made against another index:
    chunk ids from two indexes are not comparable.

    Raises:
        ValueError: `plan.target` carries no `index_version`.
        UserError: Nothing is at `plan.target.questions_path`, or a directory is, or
            `select_queue` refuses `plan.queue`.
    """
    console = console or Console()
    target = plan.target
    if not target.index_version:
        raise ValueError("label_session needs the current index_version")
    require_label_versions(target.questions_path, target.index_version, all_rounds=True)
    journal = _Journal(target, plan.queue.annotation_round)
    queue = select_queue(journal.questions, plan.queue)
    if not queue:
        console.print(
            "[green]Nothing to label.[/green] --all revisits judged questions, --review "
            "ID ... corrects specific ones, and --round 2 starts a re-annotation pass."
        )
        return
    _announce(console, queue, journal)
    limit = plan.limit or cfg.eval.pool_display
    _walk(console, retriever, queue, journal, (plan.per_method, limit), prompt)


def _walk(
    console: Console,
    retriever: LabelRetriever,
    queue: Sequence[Question],
    journal: _Journal,
    geometry: tuple[int, int],
    prompt: Ask,
) -> None:
    """Judges the queue in order until it ends or Ctrl-C, then saves and summarizes."""
    show: Show = console.print
    started = time.monotonic()
    judged = 0  # questions recorded this session
    handled = 0  # questions moved past, including skips
    try:
        for position, question in enumerate(queue, start=1):
            console.rule(Text(f"{formatting.count(position)}/{formatting.count(len(queue))}"))
            show(
                progress_table(
                    journal.questions,
                    queue,
                    handled,
                    judged,
                    time.monotonic() - started,
                    judged_overall=journal.judged_overall(),
                )
            )
            candidates, marked = journal.candidates(retriever, question, *geometry)
            view = ScreenView(blind=not journal.writes_gold)
            judgement = judge(console, question, candidates, marked, prompt, view=view)
            if judgement is not None:
                journal.record(question, candidates, judgement)
                judged += 1
                if judgement.relevant:
                    saved = formatting.plural(len(judgement.relevant), "chunk")
                    console.print(Text(f"saved {saved}\n", "green"))
                else:
                    console.print(Text("recorded: no relevant chunk in pool\n", "yellow"))
            handled = position
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: progress saved[/yellow]")
    journal.save()

    show(
        progress_table(
            journal.questions,
            queue,
            handled,
            judged,
            time.monotonic() - started,
            judged_overall=journal.judged_overall(),
        )
    )
    labelled = sum(1 for q in journal.questions if q.is_labelled)
    console.print(
        Text.assemble(
            "\n",
            (f"{formatting.count(labelled)}/{formatting.count(len(journal.questions))}", "bold"),
            f" questions have gold labels. Round history: {rounds_path(journal.path)}",
        )
    )
