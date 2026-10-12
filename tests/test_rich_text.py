"""Tests for `crossvenue_rag.rich_text`."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from crossvenue_rag.rich_text import add_text_row


def test_a_str_cell_is_literal_text_and_a_text_cell_keeps_its_style():
    table = Table()
    table.add_column("a")
    table.add_column("b")
    add_text_row(table, "[bold]x[/bold] [/y]", Text("styled", "red"), style="dim")
    console = Console(record=True, width=80)
    console.print(table)
    out = console.export_text()
    assert "[bold]x[/bold] [/y]" in out and "styled" in out
