"""Rich table rows built from values, so a value's brackets are never read as markup."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text


def add_text_row(table: Table, *cells: str | Text, style: str | None = None) -> None:
    """Adds a row whose str cells are literal text, so a value's brackets are never markup."""
    table.add_row(*(Text(c) if isinstance(c, str) else c for c in cells), style=style)
