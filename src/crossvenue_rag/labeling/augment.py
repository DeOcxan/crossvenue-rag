"""Augmenting round 1 with the unjudged chunks that `labeling.candidates` lists for a run.

A relevant chunk that only a method outside the labeling pool finds is unjudged and counts as
a miss. That method's recall is biased downward (evals/methodology.md, pooling bias) until the
chunks it retrieved are judged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .. import formatting
from ..clock import utc_now
from ..dataset.journal import (
    Augmentation,
    CoverageChange,
    Journal,
    Judgment,
    load_rounds,
    save_rounds,
)
from ..dataset.questions import load_questions, require_label_versions, save_questions
from ..errors import UserError
from ..models import Question, Result
from ..retrieval.protocols import ChunkLookup
from .candidates import CandidateList, load_candidates
from .judge import ask_picks
from .screen import Ask, ScreenView, Show, highlight, question_terms, render
from .session import LabelTarget

NOT_COVERED = ("missing_from_pool", "absent_in_docs")


def _render_gold_context(console: Console, question: Question, retriever: ChunkLookup) -> None:
    terms = question_terms(question.question)
    if not question.gold_chunk_ids:
        console.print("[dim]current gold: none (round 1 found nothing relevant)[/dim]")
        return
    show: Show = console.print
    for number, chunk_id in enumerate(question.gold_chunk_ids, start=1):
        chunk = retriever.by_id[chunk_id]
        title = f"current gold g{formatting.count(number)}  {chunk.venue} | {chunk.heading_display}"
        show(
            Panel(
                Text.assemble(highlight(chunk.text.strip(), terms)),
                title=Text(title),
                border_style="green",
            )
        )


def _confirm_coverage_changes(
    console: Console,
    question: Question,
    picked: Sequence[Result],
    prompt: Ask,
) -> list[CoverageChange] | None:
    """Coverage changes the marks imply, or None to go back to selection.

    Marking a chunk from a venue recorded as `missing_from_pool` or
    `absent_in_docs` asks whether that venue becomes `covered`. Declining
    returns to the selection, because the validator rejects a gold chunk from a
    venue that is not `covered`. A change from `absent_in_docs` records the
    earlier assessment as wrong.
    """
    changes: list[CoverageChange] = []
    picked_venues = {r.venue for r in picked}
    for venue in question.venues:
        current = question.venue_coverage.get(venue)
        if venue not in picked_venues or current not in NOT_COVERED:
            continue
        warning = (
            "; that would mean the earlier 'answer absent from the docs' was wrong"
            if current == "absent_in_docs"
            else ""
        )
        console.print(
            Text(
                f"{venue} is recorded as {current}, and you marked a {venue} chunk{warning}.",
                "yellow",
            )
        )
        while True:
            answer = (
                prompt(
                    Text.assemble(
                        f"{venue}: ",
                        ("y", "cyan"),
                        "=set covered, ",
                        ("b", "cyan"),
                        "=back to selection",
                    ),
                    default="",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if answer == "b":
                return None
            if answer == "y" and current is not None:
                changes.append(
                    CoverageChange(
                        venue=venue,
                        old=current,
                        new="covered",
                        previous_assessment_wrong=current == "absent_in_docs",
                    )
                )
                break
            console.print("[red]Expected y or b.[/red]")
    return changes


def augment_session(
    retriever: ChunkLookup,
    target: LabelTarget,
    candidates_path: str | Path,
    *,
    console: Console | None = None,
    prompt: Ask = Prompt.ask,
) -> int:
    """Judges the unjudged chunks listed in `candidates_path`; returns questions recorded.

    Each question is shown with its current gold chunks in full as context and
    only the listed chunks round 1 has not judged.
    """
    console = console or Console()
    questions_path, index_version = target.questions_path, target.index_version
    require_label_versions(questions_path, index_version, all_rounds=True)
    annotator = target.who()
    entries = load_candidates(candidates_path)
    questions = load_questions(questions_path)
    rounds = load_rounds(questions_path)
    by_id = {q.id: q for q in questions}
    _check_candidates(entries, candidates_path, index_version, by_id, rounds)

    console.print(
        Text.assemble(
            (formatting.plural(len(entries), "question"), "bold"),
            " to augment, annotator ",
            (annotator, "bold"),
            ". Only chunks never judged are shown; current gold is context. "
            "Enter skips (nothing recorded), n records that none of the shown chunks is "
            "relevant.\n",
        )
    )
    recorded = 0
    try:
        for position, item in enumerate(entries, start=1):
            question = by_id[item.id]
            entry = rounds.require_first(question.id)
            console.rule(
                Text(
                    f"augment {formatting.count(position)}/{formatting.count(len(entries))}  "
                    f"{question.id}"
                )
            )
            candidates = _new_candidates(item, entry, retriever)
            if not candidates:
                console.print("[dim]every listed chunk is already judged; nothing to do[/dim]\n")
                continue
            outcome = _choose(
                console,
                question,
                candidates,
                prompt,
                partial(_render_gold_context, console, question, retriever),
            )
            if outcome is None:
                console.print("[dim]skipped, nothing recorded[/dim]\n")
                continue
            relevant, changes = outcome
            augmentation = Augmentation(
                at=utc_now(),
                annotator=annotator,
                index_version=index_version,
                source=str(candidates_path),
                shown=[r.chunk_id for r in candidates],
                relevant=relevant,
                coverage_changes=changes,
            )
            record_augmentation(question, entry, augmentation)
            save_questions(questions_path, questions)
            save_rounds(questions_path, rounds)
            recorded += 1
            console.print(Text(_recorded_line(augmentation), "green"))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted: recorded questions are saved[/yellow]")
    console.print(Text(f"{formatting.plural(recorded, 'question')} augmented."))
    return recorded


def _check_candidates(
    entries: Sequence[CandidateList],
    candidates_path: str | Path,
    index_version: str,
    by_id: Mapping[str, Question],
    rounds: Journal,
) -> None:
    """Refuses a list made on another index or naming questions round 1 never judged."""
    stale = sorted({e.index_version for e in entries} - {index_version})
    if stale:
        raise UserError(
            f"{candidates_path} lists chunks from index {', '.join(stale)}, the current index is "
            f"{index_version}",
            hint="rebuild it with `crossvenue augment-candidates` from a run on this index",
        )
    unknown = [e.id for e in entries if e.id not in by_id or rounds.first(e.id) is None]
    if unknown:
        raise UserError(f"no round-1 judgment for: {', '.join(unknown)}")


def _new_candidates(item: CandidateList, entry: Judgment, retriever: ChunkLookup) -> list[Result]:
    """The listed chunks round 1 has not judged, with the ranks that found them."""
    judged = set(entry.judged_ids())
    ranks = {c.chunk_id: c.found_by for c in item.chunks}
    candidates = []
    for chunk_id in (c for c in ranks if c not in judged):
        result = Result.from_chunk(retriever.by_id[chunk_id], 0.0)
        result.component_ranks = dict(ranks[chunk_id])
        candidates.append(result)
    return candidates


def _recorded_line(augmentation: Augmentation) -> str:
    changes = augmentation.coverage_changes
    coverage = f", coverage: {', '.join(c.venue for c in changes)} -> covered" if changes else ""
    return (
        f"recorded: {formatting.count(len(augmentation.shown))} shown, "
        f"{formatting.count(len(augmentation.relevant))} relevant{coverage}\n"
    )


def _choose(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    prompt: Ask,
    gold_context: Callable[[], None],
) -> tuple[list[str], list[CoverageChange]] | None:
    compact = True

    def show() -> None:
        # Question, then the current gold as context, then only the unjudged chunks.
        view = ScreenView(compact=compact, between=gold_context, source_note=False)
        render(console, question, candidates, set(), view)

    def toggle_view() -> None:
        nonlocal compact
        compact = not compact
        show()

    show()
    while True:
        picks = ask_picks(console, question, candidates, prompt, "relevant unjudged", toggle_view)
        if picks is None:
            return None
        if isinstance(picks, str):  # NONE_RELEVANT: the annotator entered `n`
            return [], []
        picked = [candidates[i - 1] for i in picks]
        changes = _confirm_coverage_changes(console, question, picked, prompt)
        if changes is None:
            continue
        return [r.chunk_id for r in picked], changes


def record_augmentation(question: Question, entry: Judgment, augmentation: Augmentation) -> None:
    """Extends round 1 with an augmentation; no existing label is removed or changed.

    The augmentation is journaled under the entry's `augmentations` with every
    chunk shown, so the unmarked ones are judged-irrelevant rather than
    unjudged. Relevant chunks the entry does not hold are appended to its
    `relevant` and to the question's gold.
    """
    entry.augmentations = [*(entry.augmentations or []), augmentation]
    added = [c for c in augmentation.relevant if c not in entry.relevant]
    entry.relevant = [*entry.relevant, *added]
    question.gold_chunk_ids = [*question.gold_chunk_ids, *added]
    if augmentation.coverage_changes:
        coverage = dict(question.venue_coverage)
        for change in augmentation.coverage_changes:
            coverage[change.venue] = change.new
        question.venue_coverage = coverage
        entry.venue_coverage = dict(coverage)
