"""`crossvenue label --adjudicate`: the third pass over the pairs the labeling rounds disagree on.

Methodology section 5 resolves the disagreements of the test-retest pass by a third
pass. It walks every (question, chunk) pair both rounds judged on which round 1 (as
corrected) and round 2 disagree, grouped by question, shows both earlier verdicts, and records each
decision under the journal key `adjudicated`. The round-1 and round-2 entries are
never touched, so kappa, which reads only those two, is unchanged by it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .. import formatting, listing
from ..clock import utc_now
from ..dataset.journal import Adjudication, Decision, Journal, load_rounds, rounds_path, save_rounds
from ..dataset.questions import load_questions
from ..errors import UserError
from ..models import Chunk, Question, Result
from .screen import Ask, ScreenView, Show, render_full
from .screen import render as render_list

# The date of the criterion the adjudication applies: methodology section 5a as
# amended after round 2.
CRITERION_DATE = "2026-10-10"
CRITERION_SUMMARY = (
    f"Criterion (methodology section 5a, amended {CRITERION_DATE}): a chunk is relevant if it "
    "contains a fact without which a correct answer cannot be assembled.\n"
    "(a) A rule stated on many pages: for a question about a venue's rule or policy, the chunk "
    "that states the rule is relevant; a page of one endpoint showing only that endpoint's "
    "value is relevant only when the question is about that endpoint, or when no candidate "
    "states the rule.\n"
    "(b) A comparison's premise venue: the chunk confirming the premise the question states "
    "is relevant; that venue's chunks without the premise fact are not.\n"
    "(c) A response example showing an outcome is not relevant to a question about a "
    "duration or rule unless the duration or rule follows from it."
)


@dataclass(frozen=True)
class Pair:
    """A (question, chunk) pair on which round 1 (as corrected) and round 2 disagree."""

    question_id: str
    chunk_id: str
    number: int  # the chunk's number on the question's round-2 screen
    first: bool  # relevant in round 1
    second: bool  # relevant in round 2


def disagreeing_pairs(journal: Journal) -> list[Pair]:
    """Every pair the two rounds disagree on, by question id, in round-2 screen order.

    A pair is a chunk both rounds judged: one round 1 judged after round 2 (a later
    augmentation) has no round-2 verdict, so it is no disagreement, as kappa counts it.
    """
    pairs = []
    for question_id in sorted(journal.questions):
        entries = journal.questions[question_id]
        first, second = entries.judgment(1), entries.judgment(2)
        if first is None or second is None:
            continue
        judged_first = set(first.judged_ids())
        relevant_first, relevant_second = set(first.relevant), set(second.relevant)
        for number, chunk_id in enumerate(second.pool, start=1):
            if chunk_id not in judged_first:
                continue
            a, b = chunk_id in relevant_first, chunk_id in relevant_second
            if a != b:
                pairs.append(Pair(question_id, chunk_id, number, a, b))
    return pairs


def decided(journal: Journal, pair: Pair) -> bool | None:
    """The adjudicated verdict on `pair`, None when it is not decided yet."""
    adjudication = journal.get(pair.question_id).adjudicated
    verdict = adjudication.decisions.get(pair.chunk_id) if adjudication is not None else None
    return None if verdict is None else verdict.relevant


@dataclass(frozen=True)
class AdjudicationTarget:
    """What the pass reads and stamps: the question file, chunks, index and annotator."""

    questions_path: str
    chunks: Mapping[str, Chunk]
    index_version: str
    annotator: str


class _Pass:
    """The pairs to decide, the journal, and the screen state of one adjudication session."""

    def __init__(self, target: AdjudicationTarget, console: Console, prompt: Ask) -> None:
        self.target = target
        self.console = console
        self.prompt = prompt
        self.rounds = load_rounds(target.questions_path)
        self.questions = {q.id: q for q in load_questions(target.questions_path)}
        self.pairs = disagreeing_pairs(self.rounds)
        missing = sorted({p.chunk_id for p in self.pairs if p.chunk_id not in target.chunks})
        if missing:
            raise UserError(
                f"{formatting.plural(len(missing), 'disagreeing chunk')} not in the corpus: "
                f"{listing.first_items(missing)}",
                hint="the journal was judged against another index; carry ids with `remap-ids`",
            )
        self.compact = True
        # Panels go through `show`, as in `rate.render_rating`: a Text inside a Panel.
        self.show: Show = console.print

    def question_pairs(self, question_id: str) -> list[Pair]:
        return [p for p in self.pairs if p.question_id == question_id]

    def result(self, pair: Pair) -> Result:
        return Result.from_chunk(self.target.chunks[pair.chunk_id], 0.0)

    def show_question(self, question: Question) -> None:
        pairs = self.question_pairs(question.id)
        self.show(
            Panel(
                Text.assemble(
                    (question.question, "bold"),
                    f"\n\nid: {question.id}   category: {question.effective_category}   "
                    f"venues: {', '.join(question.venues) or formatting.MISSING}\n"
                    f"{formatting.plural(len(pairs), 'disagreeing pair')}\n\n",
                    (CRITERION_SUMMARY, "dim"),
                ),
                title="question",
                border_style="cyan",
            )
        )

    def show_list(self, question: Question) -> None:
        """The question's disagreeing pairs as a numbered list, compact or full (`f`)."""
        results = [self.result(p) for p in self.question_pairs(question.id)]
        view = ScreenView(compact=self.compact, source_note=False, blind=True)
        render_list(self.console, question, results, set(), view)

    def show_pair(self, question: Question, pair: Pair) -> None:
        pairs = self.question_pairs(question.id)
        index = pairs.index(pair) + 1
        render_full(self.console, question, self.result(pair), index)
        self.console.print(
            Text.assemble(
                f"pair {formatting.count(index)}/{formatting.count(len(pairs))} of {question.id} "
                f"(candidate {formatting.count(pair.number)} on the "
                "round-2 screen)   round 1: ",
                _verdict(relevant=pair.first),
                "   round 2: ",
                _verdict(relevant=pair.second),
            )
        )

    def ask(self, question: Question) -> str:
        """`y`, `n` or `b`, after any number of `v<n>` and `f` views."""
        pairs = self.question_pairs(question.id)
        while True:
            raw = self.prompt(
                "relevant? [cyan]y[/cyan]/[cyan]n[/cyan] ([cyan]b[/cyan]=back one pair, "
                "[cyan]v<n>[/cyan]=view, [cyan]f[/cyan]=full/compact list)",
                default="",
                show_default=False,
            )
            command = raw.strip().lower()
            if command in ("y", "n", "b"):
                return command
            if command == "f":
                self.compact = not self.compact
                self.show_list(question)
            elif command.startswith("v") and command[1:].strip().isdigit():
                number = int(command[1:].strip())
                if 1 <= number <= len(pairs):
                    render_full(self.console, question, self.result(pairs[number - 1]), number)
                else:
                    self.console.print(
                        Text(f"Expected v1 .. v{formatting.count(len(pairs))}.", style="red")
                    )
            else:
                self.console.print("[red]Expected y, n, b, v<n> or f.[/red]")

    def record(self, pair: Pair, *, relevant: bool) -> None:
        """Journals one decision under `adjudicated` and writes the journal."""
        now = utc_now()
        question = self.rounds.entry(pair.question_id)
        if question.adjudicated is None:
            question.adjudicated = Adjudication(
                annotator=self.target.annotator,
                criterion=CRITERION_DATE,
                index_version=self.target.index_version,
                decisions={},
            )
        entry = question.adjudicated
        entry.annotated_at = now
        entry.decisions[pair.chunk_id] = Decision(relevant=relevant, annotated_at=now)
        save_rounds(self.target.questions_path, self.rounds)

    def first_open(self) -> int:
        """The first pair with no decision, or the number of pairs when every one is decided."""
        return next(
            (i for i, p in enumerate(self.pairs) if decided(self.rounds, p) is None),
            len(self.pairs),
        )


def _verdict(*, relevant: bool) -> Text:
    return Text("relevant", style="green") if relevant else Text("not relevant", style="red")


def adjudicate_session(
    target: AdjudicationTarget, *, console: Console | None = None, prompt: Ask = Prompt.ask
) -> None:
    """Decides each disagreeing pair in turn; Ctrl-C stops, and the next run resumes there.

    Each decision is written to the journal when it is made, so an interrupt loses
    nothing; `b` goes back one pair and decides it again.
    """
    console = console or Console()
    session = _Pass(target, console, prompt)
    pairs = session.pairs
    position = session.first_open()
    console.print(
        Text(
            f"{formatting.plural(len(pairs), 'disagreeing pair')} over "
            f"{formatting.plural(len({p.question_id for p in pairs}), 'question')}, "
            f"{formatting.count(position)} decided, annotator {target.annotator}. "
            "Records under 'adjudicated'; rounds 1 and 2 stay intact.\nCtrl-C stops; the next "
            "run resumes at the first open pair.\n"
        )
    )
    shown = ""
    try:
        while position < len(pairs):
            pair = pairs[position]
            question = session.questions[pair.question_id]
            if pair.question_id != shown:
                console.rule(Text(question.id))
                session.show_question(question)
                shown = pair.question_id
            session.show_pair(question, pair)
            answer = session.ask(question)
            if answer == "b":
                position = max(0, position - 1)
                continue
            session.record(pair, relevant=answer == "y")
            position += 1
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: decisions so far are saved[/yellow]")
    done = sum(1 for p in pairs if decided(session.rounds, p) is not None)
    console.print(
        Text(
            f"\n{formatting.count(done)}/{formatting.count(len(pairs))} pairs decided. "
            f"Journal: {rounds_path(target.questions_path)}"
        )
    )
