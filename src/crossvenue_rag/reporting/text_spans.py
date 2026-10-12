"""Offsets in a Markdown text whose whitespace runs are collapsed to one space.

A document check matches its template against the collapsed text, so re-wrapping
a paragraph changes no match. These classes move a match's offsets back to the
text as written and test whether an offset lies in a set of spans.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable

from .patterns import WHITESPACE, collapse


class Collapsed:
    """A text with every run of whitespace as one space, and the way back to its offsets."""

    def __init__(self, text: str) -> None:
        """The collapsed form of `text`."""
        self.text = collapse(text)
        # Per whitespace run: its offset in the collapsed text, and its start and end in `text`.
        self._flat: list[int] = []
        self._starts: list[int] = []
        self._ends: list[int] = []
        removed = 0
        for m in WHITESPACE.finditer(text):
            self._flat.append(m.start() - removed)
            self._starts.append(m.start())
            self._ends.append(m.end())
            removed += m.end() - m.start() - 1

    def at(self, pos: int) -> int:
        """The offset in the text as written of offset `pos` of the collapsed text."""
        i = bisect.bisect_right(self._flat, pos) - 1
        if i < 0:
            return pos
        if pos == self._flat[i]:
            return self._starts[i]
        return self._ends[i] + pos - self._flat[i] - 1


class Found:
    """A match in the collapsed text, its offsets moved back to the text as written."""

    def __init__(self, m: re.Match[str], flat: Collapsed) -> None:
        """`m`, a match in `flat.text`."""
        self._m, self._flat = m, flat
        self.group = m.group()
        self.groups: tuple[str, ...] = m.groups()
        self.start = flat.at(m.start())
        end = m.end()
        self.span = (self.start, flat.at(end - 1) + 1 if end > m.start() else self.start)

    def group_start(self, i: int) -> int:
        """The offset in the text as written where group `i` starts."""
        return self._flat.at(self._m.start(i))


class Spans:
    """A set of (start, end) spans, for testing whether an offset lies in any of them."""

    def __init__(self, spans: Iterable[tuple[int, int]]) -> None:
        """The union of `spans`."""
        merged: list[tuple[int, int]] = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self._starts = [a for a, _ in merged]
        self._ends = [b for _, b in merged]

    def __contains__(self, pos: int) -> bool:
        """Whether `pos` lies in a span: start included, end excluded."""
        i = bisect.bisect_right(self._starts, pos) - 1
        return i >= 0 and pos < self._ends[i]
