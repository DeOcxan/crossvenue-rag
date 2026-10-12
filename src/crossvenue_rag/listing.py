"""Prose lists of items for a person to read: `a, b and c`, `a, b, c, d, e, ... 3 more`."""

from __future__ import annotations

from collections.abc import Sequence

from . import formatting

# The items a message lists before it counts the rest (`first_items`).
LISTED_ITEMS = 5


def prose(items: Sequence[str]) -> str:
    """Items as prose: `a`, `a and b`, `a, b and c`."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def first_items(items: Sequence[str], separator: str = ", ") -> str:
    """The first `LISTED_ITEMS` of `items`, then how many more: `a, b, c, d, e, ... 3 more`."""
    shown = separator.join(items[:LISTED_ITEMS])
    rest = len(items) - LISTED_ITEMS
    return shown + (f"{separator}... {formatting.count(rest)} more" if rest > 0 else "")
