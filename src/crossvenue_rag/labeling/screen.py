"""The labeling screen: what a candidate row shows, and how.

A row is compact by default. It shows the venue, the heading path, the methods that found it,
the first body line and a window around a later question term. A screen does not hold 25 full
chunks, and the annotator would end up judging by scroll position; `v<n>` shows candidate n in
full.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import formatting
from ..models import Question, Result
from ..rich_text import add_text_row

# Shorter words of a question are too common to screen chunks by.
MIN_TERM_LENGTH = 3
# The other venues a line names before it says how many more there are.
OTHERS_SHOWN = 2

# How a screen asks: rich's `Prompt.ask`, or a test's scripted answers.
Ask = Callable[..., str]
# Prints one renderable: `Console.print`, or a stand-in a test passes.
Show = Callable[[RenderableType], None]

FIRST_LINE_CHARS = 160


MATCH_WINDOW_CHARS = 120


HIGHLIGHT_STYLE = "bold yellow"


# Words too common to be worth highlighting. Short on purpose: an identifier
# like `order` or `limit` is exactly what the annotator is scanning for.
_STOPWORD_TEXT = """
the and for are was were with that this what whats which who whom how when where why
does did doing can could should would will shall may might must have has had not
you your yours they them their there here from into onto over under about than then
just only also still even very much many more most some any all each every both
get got use using used make made need want like one two its it's i'm i've dont don't
doesn't isn't aren't cant can't
"""


_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./-]*[A-Za-z0-9]|[A-Za-z0-9]{3,}")


def question_terms(question: str) -> list[str]:
    """Terms of the question worth highlighting in candidates, longest first."""
    terms = {t.lower() for t in _TERM_RE.findall(question)}
    terms = {t for t in terms if len(t) >= MIN_TERM_LENGTH and t not in _STOPWORDS}
    # Longest first, so `recvWindow` is highlighted whole before its prefix `recv` can match.
    return sorted(terms, key=lambda t: (-len(t), t))


def highlight(text: str, terms: Sequence[str]) -> Text:
    """Text with every occurrence of the terms styled, case-insensitively.

    A rich `Text` rather than markup: documentation is full of `[text](url)`,
    which markup would remove from the display, and a candidate missing words gets
    mislabeled.
    """
    rendered = Text(text)
    rendered.highlight_words(terms, style=HIGHLIGHT_STYLE, case_sensitive=False)
    return rendered


def _body_lines(result: Result) -> list[str]:
    """Lines of the body, flattened: no breadcrumb, heading lines or code fences.

    Headings are skipped because they repeat the breadcrumb, which already
    carries the venue and section words a question is likely to share.
    """
    crumb = " > ".join(result.heading_path)
    lines = []
    for line in result.text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == crumb or stripped.startswith(("#", "```", "~~~")):
            continue
        lines.append(" ".join(stripped.split()))
    return lines


def first_line(result: Result) -> str:
    """The first line of the body, past the breadcrumb, heading lines and code fences.

    A chunk that opens with a code sample shows its first line of code.
    """
    lines = _body_lines(result)
    if not lines:
        return "(heading only)"
    flat = lines[0]
    return flat[:FIRST_LINE_CHARS] + (" ..." if len(flat) > FIRST_LINE_CHARS else "")


def match_window(
    result: Result, terms: Sequence[str], width: int = MATCH_WINDOW_CHARS
) -> str | None:
    """About `width` characters around the first question term in the body.

    Searched *after* the first line, which the compact row already shows with
    its terms highlighted; repeating it would add nothing to the row.
    The window is centered on the match, shifted inward at either end of the
    text, and trimmed to whole words. None when no term occurs.
    """
    rest = " ".join(_body_lines(result)[1:])
    if not rest or not terms:
        return None
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    match = pattern.search(rest)
    if match is None:
        return None
    start = max(0, match.start() - (width - (match.end() - match.start())) // 2)
    end = min(len(rest), start + width)
    start = max(0, end - width)
    # Trim a word cut in half at either edge, never past the match itself.
    if start > 0 and rest[start - 1] != " ":
        space = rest.find(" ", start, match.start())
        start = space + 1 if space != -1 else start
    if end < len(rest) and rest[end] != " ":
        space = rest.rfind(" ", match.end(), end)
        end = space if space != -1 else end
    snippet = rest[start:end].strip()
    return ("... " if start > 0 else "") + snippet + (" ..." if end < len(rest) else "")


def _copies_note(result: Result) -> str:
    """Where else this exact text occurs, when deduplication merged copies."""
    others = [src.doc_path for src in result.sources[1:]]
    if not others:
        return ""
    shown = ", ".join(others[:OTHERS_SHOWN]) + (
        f", +{formatting.count(len(others) - OTHERS_SHOWN)}" if len(others) > OTHERS_SHOWN else ""
    )
    return f"\n(+{formatting.count(len(others))} identical: {shown})"


# Short names for the "found by" column; distinct, so two methods sharing a
# prefix (venue_split, venue_split_oracle) are never shown alike.
_METHOD_SHORT = {
    "bm25": "bm25",
    "dense": "dense",
    "hybrid": "hybrid",
    "hybrid_rerank": "rerank",
    "venue_split": "split",
    "venue_split_oracle": "oracle",
}


@dataclass(frozen=True)
class ScreenView:
    """How the judging screen is drawn.

    `compact` shortens chunk previews; `between` draws something between the
    question and the table. `source_note=False` drops the "source chunks in
    pool" line, which only makes sense when the candidates are the whole pool.
    `blind=True` is the round-2 screen: no marks, no "found by" column, no
    source-chunk markers and no question notes, so nothing on screen carries the
    first verdict or a cue outside the criterion.
    """

    compact: bool = True
    between: Callable[[], None] | None = None
    source_note: bool = True
    blind: bool = False


def render(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    marked: set[str],
    view: ScreenView = ScreenView(),
) -> None:
    """Question panel, then `view.between()` if given, then the candidate table."""
    blind = view.blind
    _render_question(
        console,
        question,
        candidates,
        with_source=view.source_note and not blind,
        with_notes=not blind,
    )
    if view.between is not None:
        view.between()
    _render_table(
        console, question, candidates, set() if blind else marked, compact=view.compact, blind=blind
    )


def _render_question(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    *,
    with_source: bool,
    with_notes: bool,
) -> None:
    pool_ids = {r.chunk_id for r in candidates}
    source_note, missed = "", ""
    if with_source and question.source_chunk_ids:
        found = [
            formatting.count(i)
            for i, r in enumerate(candidates, start=1)
            if r.chunk_id in question.source_chunk_ids
        ]
        in_pool = sum(1 for c in question.source_chunk_ids if c in pool_ids)
        source_note = (
            f"\nsource chunks in pool: {formatting.count(in_pool)}/"
            f"{formatting.count(len(question.source_chunk_ids))}"
            + (f" (#{', #'.join(found)})" if found else "")
        )
        missed = "  (pool missed the source: note it)" if not in_pool else ""
    notes = f"\nnotes: {question.notes or formatting.MISSING}" if with_notes else ""
    show: Show = console.print
    show(
        Panel(
            Text.assemble(
                (question.question, "bold"),
                "\n\n"
                f"id: {question.id}   category: {question.effective_category}   "
                f"difficulty: {question.difficulty or formatting.MISSING}\n"
                f"venues: {', '.join(question.venues) or formatting.MISSING}   "
                f"expected: {question.expected_behavior}"
                f"{source_note}",
                (missed, "yellow"),
                notes,
            ),
            title="question",
            border_style="cyan",
        )
    )


def _render_table(
    console: Console,
    question: Question,
    candidates: Sequence[Result],
    marked: set[str],
    *,
    compact: bool,
    blind: bool = False,
) -> None:
    terms = question_terms(question.question)
    table = Table(show_lines=not compact, expand=True)
    table.add_column("#", width=5, justify="right", no_wrap=True)
    table.add_column("venue", width=11)
    table.add_column("heading path", width=34, overflow="fold")
    if not blind:
        table.add_column("found by", width=18, overflow="fold")
    table.add_column("first line" if compact else "text", overflow="fold")

    for index, result in enumerate(candidates, start=1):
        found_by = " ".join(
            f"{_METHOD_SHORT.get(m, m)}#{r}" for m, r in sorted(result.component_ranks.items())
        )
        is_source = not blind and result.chunk_id in question.source_chunk_ids
        number = Text()
        number.append("*" if result.chunk_id in marked else " ", style="green")
        # The number itself carries the source mark, so it survives a glance.
        number.append(formatting.count(index), style="bold cyan" if is_source else "")
        number.append("s" if is_source else " ", style="cyan")
        heading = highlight(result.heading_display, terms)
        heading.append(_copies_note(result), style="dim")
        if compact:
            body = highlight(first_line(result), terms)
            window = match_window(result, terms)
            if window is not None:
                body.append("\n")
                body.append_text(highlight(window, terms))
        else:
            body = highlight(result.text.strip(), terms)
        cells: list[str | Text] = [number, result.venue, heading]
        cells += ([] if blind else [found_by]) + [body]
        add_text_row(table, *cells)
    show: Show = console.print
    show(table)
    if blind:
        console.print("[dim]v<n> shows candidate n in full; f toggles full text for all.[/dim]")
    else:
        console.print(
            "[dim]#s = the question's source chunk; * = marked. "
            "v<n> shows candidate n in full; f toggles full text for all.[/dim]"
        )


def render_full(console: Console, question: Question, result: Result, index: int) -> None:
    """Prints candidate `index` in full, question terms highlighted, with every source path."""
    terms = question_terms(question.question)
    where = "\n".join(
        f"{'primary ' if i == 0 else 'copy    '}{src.doc_path}  "
        f"[{src.environment or formatting.MISSING}/{src.api_version or formatting.MISSING}/"
        f"{src.product or formatting.MISSING}/{src.doc_type or formatting.MISSING}]"
        for i, src in enumerate(result.sources)
    )
    show: Show = console.print
    show(
        Panel(
            Text.assemble(
                highlight(result.text.strip(), terms), ("\n\n" + (where or result.doc_path), "dim")
            ),
            title=Text(f"v{formatting.count(index)}  {result.venue} | {result.heading_display}"),
            border_style="magenta",
        )
    )
