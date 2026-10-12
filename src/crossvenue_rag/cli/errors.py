"""The CLI's error boundary: an expected failure prints its message and hint and exits 1.

The expected failures are a `UserError`, a problem the user can fix (fetch before
ingest, a bad config value), and a failure outside the program: an `ExternalFailure`
(a model call that failed) or Postgres refusing a connection. Anything else is a
defect and keeps its traceback, a ValueError or a pydantic ValidationError included;
`CROSSVENUE_DEBUG=1` re-raises everything.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.text import Text

from ..errors import ExpectedFailure
from .app import say

# The database's connection failures, matched by class name: psycopg is imported
# lazily, and importing it here would load it for every command.
DATABASE_ERRORS = frozenset({"OperationalError", "ConnectionException"})


def _describe(exc: Exception) -> tuple[str, str]:
    """(message, hint) of an expected failure."""
    if isinstance(exc, ExpectedFailure):
        return str(exc), exc.hint
    return (
        f"cannot reach Postgres: {exc}".strip(),
        "start it with `make db`, or set CROSSVENUE_PG_DSN to point elsewhere",
    )


def _expected(exc: Exception) -> bool:
    """A UserError, an ExternalFailure, or the database refusing a connection."""
    return isinstance(exc, ExpectedFailure) or type(exc).__name__ in DATABASE_ERRORS


def debug_requested() -> bool:
    """Whether `CROSSVENUE_DEBUG` turns tracebacks on: any value but empty, 0, false or no."""
    return os.environ.get("CROSSVENUE_DEBUG", "").strip().lower() not in {"", "0", "false", "no"}


@contextmanager
def friendly_errors(console: Console) -> Iterator[None]:
    """Prints an expected failure as its message, hint and the debug switch, and exits 1.

    Raises:
        SystemExit: An expected failure was printed; the exit status is 1.
    """
    try:
        yield
    except Exception as exc:
        if debug_requested():
            raise
        if not _expected(exc):
            raise  # a defect: the traceback is the point
        message, hint = _describe(exc)
        # A message quotes what the user typed (a venue, a term, a path), so it is
        # printed as literal text: as markup, `[/x]` raises and `[bold]` vanishes.
        say(console, Text("Error:", style="red"), message)
        if hint:
            say(console, hint, style="dim")
        console.print("[dim]Set CROSSVENUE_DEBUG=1 for the full traceback.[/dim]")
        raise SystemExit(1) from exc
