"""Which venues a question names, read from its text, for the `venue_split` retrieval method.

The dictionary is each venue's `aliases` in data/sources.yaml (regexes) plus the
venue's own name, matched case-insensitively on whole words, so `hl` matches
"on hl" but not "hloc". Its errors are part of what `venue_split` is measured
on; `venue_split_oracle` reads the question's `venues` field instead.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

from ..corpus.sources import SourceSpec, load_sources

# How a detection compares with a question's `venues` field (`detection_outcome`).
DetectionOutcome = Literal["exact", "partial", "missed", "wrong", "none"]
DETECTION_OUTCOMES: tuple[DetectionOutcome, ...] = get_args(DetectionOutcome)


class VenueDetector:
    """Finds the venues a text names, by each venue's name and aliases."""

    def __init__(self, specs: Sequence[SourceSpec]) -> None:
        """A detector for the venues in `specs`."""
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for spec in specs:
            alternatives = [re.escape(spec.venue), *spec.aliases]
            pattern = r"(?<![A-Za-z0-9_])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9_])"
            self._patterns.append((spec.venue, re.compile(pattern, re.IGNORECASE)))

    @classmethod
    def from_file(cls, sources_path: str | Path) -> VenueDetector:
        """A detector for every venue in `sources_path`."""
        return cls(load_sources(sources_path))

    def detect(self, text: str) -> list[str]:
        """Venues named in `text`, in order of first mention."""
        found: list[tuple[int, str]] = []
        for venue, pattern in self._patterns:
            match = pattern.search(text)
            if match is not None:
                found.append((match.start(), venue))
        return [venue for _, venue in sorted(found)]


def detection_outcome(detected: Sequence[str], expected: Sequence[str]) -> DetectionOutcome:
    """How a detection compares with the question's `venues` field.

    `exact`: the same set; `partial`: a non-empty strict subset, nothing extra;
    `missed`: nothing detected though venues were expected; `wrong`: at least
    one detected venue the question does not name; `none`: nothing expected and
    nothing detected.
    """
    got, want = set(detected), set(expected)
    if got - want:
        return "wrong"
    if got == want:
        return "exact" if want else "none"
    if not got:
        return "missed"
    return "partial"
