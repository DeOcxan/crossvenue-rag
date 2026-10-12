"""The checked documents' own text: the items of a list and the cells of a table.

A document states some counts of its own text ("three limits" over a list of
three); these read them from the text as the `Doc` holds it.
"""

from __future__ import annotations

from collections import Counter

from ..patterns import TABLE_HEADER_LINES, table_cells, table_lines


def list_items(text: str, start: str, end: str) -> list[str]:
    """The lines starting "- " of `text` from the first `start` to the first `end` after it.

    Raises:
        ValueError: `text` has no `start`, or no `end` after it.
    """
    first = text.index(start)
    section = text[first : text.index(end, first)]
    return [line for line in section.splitlines() if line.startswith("- ")]


def table_column(text: str, header: str, name: str) -> list[str]:
    """The cells of column `name` of the Markdown table whose header row starts `header`.

    Raises:
        ValueError: `text` has no `header`, or the table has no column `name`.
    """
    lines = [line for _, line in table_lines(text, text.index(header))]
    names = table_cells(lines[0])
    return [table_cells(line)[names.index(name)] for line in lines[TABLE_HEADER_LINES:]]


def table_column_counts(text: str, header: str, name: str) -> dict[str, int]:
    """Rows per value of column `name` of the table whose header row starts `header`.

    Raises:
        ValueError: `text` has no `header`, or the table has no column `name`.
    """
    return dict(Counter(table_column(text, header, name)))
