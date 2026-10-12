"""The interactive selection session: the keyboard loop and screen over `dataset.selection`.

The screen shows the question and the chunks it was written from, because whether a question
is grounded, the first rejection reason, cannot be judged from its text alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .. import formatting
from ..config import Config
from ..corpus.artifacts import chunks_path, load_chunks
from ..dataset.records import QuestionRecord
from ..dataset.selection import (
    CATEGORY_ORDER,
    REJECTION_REASONS,
    TARGETS,
    Decision,
    Progress,
    append_decision,
    load_decisions,
    progress,
    queue,
)
from ..errors import UserError
from ..models import Chunk
from ..rich_text import add_text_row
from .screen import Ask, Show

# At or below this many spare candidates a category's headroom is flagged.
FEW_SPARE = 2

CHUNK_PREVIEW_CHARS = 200
# The answerable-comparison row fills the accepted and target columns; the other four stay empty.
EMPTY_TOTAL_CELLS = 4


def selection_progress_table(state: Progress, current_category: str | None = None) -> Table:
    """Accepted against target per category, plus what is left to draw from."""
    table = Table(title="selection progress", show_lines=False)
    table.add_column("category")
    table.add_column("accepted", justify="right")
    table.add_column("target", justify="right")
    table.add_column("still need", justify="right")
    table.add_column("unseen", justify="right")
    table.add_column("available", justify="right")
    table.add_column("", width=12)

    for category in CATEGORY_ORDER:
        accepted = state.accepted[category]
        target = TARGETS[category]
        remaining = state.remaining[category]
        unseen = state.unseen[category]
        short = state.shortfall(category)
        if accepted >= target:
            note = Text("done", "green")
        elif not state.reachable(category):
            # The pool cannot fill this category. Saying so here rather than at
            # finalize leaves time to act on it.
            note = Text("unreachable", "red")
        elif remaining == short:
            # Every remaining candidate must be accepted to hit the target, so
            # the next rejection makes it unreachable.
            note = Text("no headroom", "yellow")
        elif remaining - short <= FEW_SPARE:
            note = Text(f"{formatting.count(remaining - short)} spare", "yellow")
        else:
            note = Text("")
        label = Text(category, "bold" if category == current_category else "")
        add_text_row(
            table,
            label,
            *_counts(accepted, target),
            formatting.count(short) if short else formatting.MISSING,
            *_counts(unseen, remaining),
            note,
        )

    table.add_section()
    add_text_row(
        table,
        Text("total", "bold"),
        *_counts(
            state.total_accepted,
            state.total_target,
            max(0, state.total_target - state.total_accepted),
            sum(state.unseen.values()),
            sum(state.remaining.values()),
        ),
        "",
    )
    # An unanswerable comparison item does not count toward the subgroup the
    # headline claim is measured on.
    shortfall = TARGETS["comparison"] - state.answerable_comparison
    style = "green" if shortfall <= 0 else "yellow"
    add_text_row(
        table,
        Text("answerable comparison", style),
        Text(formatting.count(state.answerable_comparison), style),
        *_counts(TARGETS["comparison"]),
        *[""] * EMPTY_TOTAL_CELLS,
    )
    return table


def _counts(*values: int) -> list[str]:
    return [formatting.count(v) for v in values]


def question_panel(record: QuestionRecord) -> Panel:
    """The question with its metadata and notes, as the selection screen shows it."""
    return Panel(
        Text.assemble(
            (record.question, "bold"),
            "\n\n"
            f"id: {record.id}   category: {record.category}   difficulty: {record.difficulty}\n"
            f"venues: {', '.join(record.venues)}   qtype: {record.qtype}   tier: {record.tier}\n"
            f"answerable: {record.is_answerable}   expected: {record.expected_behavior}"
            f"   adversarial_type: {record.adversarial_type or formatting.MISSING}\n\n"
            f"notes: {record.notes or formatting.MISSING}",
        ),
        title=Text(record.id),
        border_style="cyan",
    )


def grounding_table(record: QuestionRecord, chunks: dict[str, Chunk]) -> Table:
    """The chunks the question was written from: the evidence for "grounded"."""
    table = Table(title="source chunks (provenance only)", show_lines=True)
    table.add_column("venue", width=12)
    table.add_column("heading path", width=34, overflow="fold")
    table.add_column("text", overflow="fold")

    if not record.source_chunk_ids:
        table.add_row("[yellow]none[/yellow]", "", "[yellow]no grounding recorded[/yellow]")
        return table

    for chunk_id in record.source_chunk_ids:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            # A stale pointer after re-chunking; the question may still be fine,
            # but its grounding cannot be checked from here.
            table.add_row(
                Text("missing", "red"), Text(chunk_id), Text("not in the current corpus", "red")
            )
            continue
        preview = " ".join(chunk.text.split())[:CHUNK_PREVIEW_CHARS]
        table.add_row(
            Text(chunk.venue),
            Text(chunk.heading_display),
            Text(preview + (" ..." if len(preview) == CHUNK_PREVIEW_CHARS else "")),
        )
    return table


def _ask_reason(console: Console, prompt: Ask) -> str:
    console.print(
        Text.assemble(
            "reason:",
            *(
                part
                for key, label in REJECTION_REASONS.items()
                for part in ("  ", (key, "cyan"), f"={label}")
            ),
        )
    )
    while True:
        raw = prompt("reason", default="o", show_default=True).strip().lower()
        if raw in REJECTION_REASONS:
            return REJECTION_REASONS[raw]
        if raw in REJECTION_REASONS.values():
            return raw
        console.print("[red]Pick one of the letters above.[/red]")


@dataclass(frozen=True)
class SelectionPlan:
    """A selection session, built once by `crossvenue select`.

    The question file it is for, the decision log it appends to, the categories
    to review (all when `None`), and whether skipped questions come back.
    """

    questions_path: str
    selection_file: str
    categories: Sequence[str] | None = None
    include_skipped: bool = False


def check_categories(categories: Sequence[str] | None) -> None:
    """Refuses a category the targets do not name.

    Raises:
        UserError: A category is not one of the targets'.
    """
    unknown = set(categories or []) - set(TARGETS)
    if unknown:
        raise UserError(
            f"unknown categor{'y' if len(unknown) == 1 else 'ies'}: {', '.join(sorted(unknown))}",
            hint=f"expected any of {', '.join(sorted(TARGETS))}",
        )


def run(
    cfg: Config,
    candidates: Sequence[QuestionRecord],
    plan: SelectionPlan,
    *,
    console: Console | None = None,
    prompt: Ask = Prompt.ask,
) -> None:
    """Walks the candidate pool and records a decision for each question.

    `prompt` reads each answer; a test passes a scripted one.
    """
    console = console or Console()
    show: Show = console.print
    questions_path, selection_file = plan.questions_path, plan.selection_file
    categories, include_skipped = plan.categories, plan.include_skipped
    decisions = load_decisions(selection_file)

    chunks: dict[str, Chunk] = {}
    if chunks_path(cfg).exists():
        chunks = {c.chunk_id: c for c in load_chunks(cfg)}
    else:
        console.print(
            "[yellow]No chunks.jsonl, so source previews are unavailable. "
            "Run `crossvenue ingest` to see the grounding.[/yellow]\n"
        )

    pending = list(queue(candidates, decisions, categories, include_skipped=include_skipped))
    if not pending:
        console.print(
            "[green]Nothing left to decide.[/green] Use --include-skipped to revisit "
            "skipped questions, or `crossvenue select --finalize` to write the set."
        )
        show(selection_progress_table(progress(candidates, decisions)))
        return

    console.print(
        Text.assemble(
            (formatting.plural(len(pending), "candidate"), "bold"),
            " to review. ",
            ("y", "cyan"),
            "=keep  ",
            ("n", "cyan"),
            "=reject  ",
            ("s", "cyan"),
            "=skip  ",
            ("q", "cyan"),
            "=save and quit\n",
        )
    )

    over_target_warned: set[str] = set()

    for position, record in enumerate(pending, start=1):
        state = progress(candidates, decisions)
        console.rule(
            Text(
                f"{formatting.count(position)}/{formatting.count(len(pending))}  {record.category}"
            )
        )
        show(selection_progress_table(state, record.category))

        if (
            state.accepted[record.category] >= TARGETS[record.category]
            and record.category not in over_target_warned
        ):
            console.print(
                Text.assemble(
                    "\n",
                    (
                        f"{record.category} has already reached its target of "
                        f"{formatting.count(TARGETS[record.category])}.",
                        "yellow",
                    ),
                    " Keeping more is fine (the extra items widen the breakdown row), but the "
                    "target is met, so this is a choice rather than a need.",
                )
            )
            over_target_warned.add(record.category)

        show(question_panel(record))
        show(grounding_table(record, chunks))

        answer = prompt(Text("[y]keep / [n]reject / [s]skip / [q]uit"), default="s").strip().lower()
        if answer.startswith("q"):
            console.print("[yellow]saved; run again to continue[/yellow]")
            break
        if answer.startswith("y"):
            decision = Decision.now(record.id, "accept")
            console.print("[green]kept[/green]\n")
        elif answer.startswith("n"):
            decision = Decision.now(record.id, "reject", _ask_reason(console, prompt))
            console.print(Text.assemble(("rejected", "red"), f": {decision.reason}\n"))
        else:
            decision = Decision.now(record.id, "skip")
            console.print("[dim]skipped[/dim]\n")

        append_decision(selection_file, decision)
        decisions[record.id] = decision

    show(selection_progress_table(progress(candidates, decisions)))
    console.print(
        Text.assemble(
            f"\ndecisions: {selection_file}\nwhen the targets are met: ",
            ("crossvenue select --finalize", "cyan"),
            f" -> {questions_path}",
        )
    )
