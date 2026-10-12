"""The annotator's choice of relevant candidates, checked before it is saved.

In every mode a comparison question accounts for each venue without a marked chunk
(`confirm_coverage`), and a version_sensitive or adversarial question shows the marked
chunks' `environment` and `api_version` beside its notes (`confirm_versions`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .. import formatting
from ..models import Question, Result, VenueCoverage
from .screen import Ask, ScreenView, Show, render, render_full


def parse_selection(raw: str, count: int) -> list[int] | None:
    """Indices from `1 3 7`, `1,3,7` or `1-4`.

    None on bad input: a token that is not a number or a range, a reversed range
    (`5-2`), a number outside 1 to `count`, or input with no number at all (`,`).
    Only the explicit none answer, which `ask_picks` handles, records no
    relevant chunk.
    """
    picks: set[int] = set()
    for token in raw.replace(",", " ").split():
        if "-" in token[1:]:
            low, _, high = token.partition("-")
            if not (low.isdigit() and high.isdigit()) or int(low) > int(high):
                return None
            picks.update(range(int(low), int(high) + 1))
        elif token.isdigit():
            picks.add(int(token))
        else:
            return None
    if not picks or any(index < 1 or index > count for index in picks):
        return None
    return sorted(picks)


NONE_RELEVANT: Literal["none"] = "none"


def ask_picks(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    prompt: Ask,
    label: str,
    toggle_view: Callable[[], None],
) -> list[int] | Literal["none"] | None:
    """Prompts until the annotator enters a selection of candidates.

    `v<n>` shows candidate n in full and `f` calls `toggle_view`; both prompt
    again.

    Args:
        console: Where the full view and input errors are printed.
        question: The question the candidates are shown for.
        candidates: The numbered candidates, 1 to `len(candidates)`.
        prompt: Reads one line of input.
        label: What the numbers mark, the first words of the prompt.
        toggle_view: Switches between the compact and the full list and redraws it.

    Returns:
        The selected numbers (1-based, as `parse_selection` gives them),
        `NONE_RELEVANT` for `n`, or None when the annotator skips the question.
    """
    while True:
        raw = prompt(
            f"{label} #s ([cyan]enter[/cyan]=skip, [cyan]n[/cyan]=none relevant, "
            "[cyan]v<n>[/cyan]=view, [cyan]f[/cyan]=full/compact)",
            default="",
            show_default=False,
        )
        command = raw.strip().lower()
        if command == "":
            return None
        if command == "f":
            toggle_view()
            continue
        if command.startswith("v"):
            number = command[1:].strip()
            if number.isdigit() and 1 <= int(number) <= len(candidates):
                render_full(console, question, candidates[int(number) - 1], int(number))
            else:
                console.print(Text(f"Expected v1 .. v{formatting.count(len(candidates))}.", "red"))
            continue
        if command == "n":
            return NONE_RELEVANT
        picks = parse_selection(raw, len(candidates))
        if picks is None:
            console.print("[red]Expected numbers like `1 4 7` or `2-5`.[/red]")
            continue
        return picks


def uncovered_venues(
    question: Question, candidates: Sequence[Result], relevant: Sequence[str]
) -> list[str]:
    """Venues the question names that no marked chunk comes from."""
    by_id = {r.chunk_id: r for r in candidates}
    covered = {by_id[c].venue for c in relevant if c in by_id}
    return [v for v in question.venues if v not in covered]


_COVERAGE_KEYS: dict[str, VenueCoverage] = {"a": "absent_in_docs", "m": "missing_from_pool"}


def confirm_coverage(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    relevant: Sequence[str],
    prompt: Ask,
) -> dict[str, VenueCoverage] | None:
    """Per-venue coverage for a comparison question, or None to go back.

    Every venue with a marked chunk is `covered`. For each one without, the
    annotator is told how many of its candidates the pool held. The annotator
    then says which of the two reasons applies ("the docs do not answer this" or
    "the pool had nothing suitable"), or goes back and marks something. The answer is
    recorded because an incomplete comparison gold set is otherwise
    indistinguishable from a deliberately one-sided one.
    """
    if not question.needs_venue_coverage:
        return {}
    missing = set(uncovered_venues(question, candidates, relevant))
    coverage: dict[str, VenueCoverage] = {}
    for venue in question.venues:
        if venue not in missing:
            coverage[venue] = "covered"
            continue
        numbers = [
            formatting.count(i) for i, r in enumerate(candidates, start=1) if r.venue == venue
        ]
        console.print(
            Text.assemble(
                (f"comparison: nothing marked from {venue}", "yellow"),
                f": {formatting.plural(len(numbers), venue + ' candidate')} in the pool"
                + (f" (#{', #'.join(numbers)})" if numbers else ""),
            )
        )
        while True:
            answer = (
                prompt(
                    Text.assemble(
                        f"{venue}: ",
                        ("b", "cyan"),
                        "=back to selection, ",
                        ("a", "cyan"),
                        f"=answer absent from {venue}'s docs, ",
                        ("m", "cyan"),
                        f"=no suitable {venue} candidate in pool",
                    ),
                    default="",
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if answer == "b":
                return None
            if answer in _COVERAGE_KEYS:
                coverage[venue] = _COVERAGE_KEYS[answer]
                break
            console.print("[red]Expected b, a or m.[/red]")
    return coverage


def version_table(candidates: Sequence[Result], relevant: Sequence[str]) -> Table:
    """The `environment` and `api_version` of each marked chunk, over all its occurrences."""
    by_id = {r.chunk_id: r for r in candidates}
    table = Table(title="marked chunks: environment / api_version", expand=True)
    table.add_column("#", justify="right", width=4)
    table.add_column("venue", width=11)
    table.add_column("environment", width=18)
    table.add_column("api_version", width=14)
    table.add_column("occurrences", overflow="fold")
    order = {r.chunk_id: i for i, r in enumerate(candidates, start=1)}
    for chunk_id in relevant:
        result = by_id[chunk_id]
        sources = result.sources or []
        envs = sorted({s.environment or formatting.MISSING for s in sources})
        versions = sorted({s.api_version or formatting.MISSING for s in sources})
        table.add_row(
            Text(formatting.count(order[chunk_id])),
            Text(result.venue),
            Text(", ".join(envs) or formatting.MISSING),
            Text(", ".join(versions) or formatting.MISSING),
            Text(", ".join(s.doc_path for s in sources) or result.doc_path),
        )
    return table


def confirm_versions(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    relevant: Sequence[str],
    prompt: Ask,
) -> bool:
    """Shows the marked chunks against the trap notes; False goes back to selection.

    Only version_sensitive and adversarial questions with a marked chunk are
    shown; every other question returns True without showing anything.

    The trap named in the notes is not parsed; the annotator compares.
    """
    if question.effective_category not in ("version_sensitive", "adversarial"):
        return True
    if not relevant:
        return True
    console.print(Text.assemble(("notes:", "bold"), f" {question.notes or formatting.MISSING}"))
    show: Show = console.print
    show(version_table(candidates, relevant))
    while True:
        answer = (
            prompt(
                "is any marked chunk only in the version/environment the notes call "
                "the trap? [cyan]enter[/cyan]=save, [cyan]b[/cyan]=back to selection",
                default="",
                show_default=False,
            )
            .strip()
            .lower()
        )
        if answer == "":
            return True
        if answer == "b":
            return False
        console.print("[red]Expected enter or b.[/red]")


@dataclass(slots=True)
class RelevanceJudgement:
    """The annotator's verdict on one question: the relevant chunks and the venue coverage."""

    relevant: list[str]
    venue_coverage: dict[str, VenueCoverage]


def judge(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    marked: set[str],
    prompt: Ask,
    *,
    blind: bool = False,
) -> RelevanceJudgement | None:
    """Asks for the relevant candidates and runs the pre-save checks; None is a skip.

    `blind=True` renders the round-2 screen (`screen.render`).
    """
    compact = True

    def toggle_view() -> None:
        nonlocal compact
        compact = not compact
        render(console, question, candidates, marked, ScreenView(compact=compact, blind=blind))

    render(console, question, candidates, marked, ScreenView(compact=compact, blind=blind))
    while True:
        picks = ask_picks(console, question, candidates, prompt, "relevant", toggle_view)
        if picks is None:
            return None
        if isinstance(picks, str):  # NONE_RELEVANT: the annotator entered `n`
            # An explicit "nothing here" is information: it marks the question as
            # reviewed-and-unanswerable-from-the-pool rather than untouched,
            # which is a direct signal of pool failure.
            relevant: list[str] = []
        else:
            relevant = [candidates[i - 1].chunk_id for i in picks]
        if not confirm_versions(console, question, candidates, relevant, prompt):
            continue
        coverage = confirm_coverage(console, question, candidates, relevant, prompt)
        if coverage is None:
            continue
        return RelevanceJudgement(relevant, coverage)
