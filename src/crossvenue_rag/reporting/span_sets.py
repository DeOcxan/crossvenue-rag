"""Sets of (start, end) offset spans of a text, for testing whether an offset lies in one."""

from __future__ import annotations

import bisect
from collections.abc import Iterable


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
