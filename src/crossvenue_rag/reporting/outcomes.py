"""The records a document check deals in: problems, listed numbers, special hole values.

A problem, a number or identifier as `reproduce -v` lists it, a text cell (`SKIP`) and a
registered parameter (`Param`).
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import formatting

SKIP = object()  # a table cell the spec does not recompute (text)


class Param(str):
    """A hole of a check that holds a parameter.

    Matched like the others, and listed as given instead of counted as compared.
    """

    reason: str

    def __new__(cls, value: str, reason: str) -> Param:
        """The hole `value`, given for `reason`."""
        obj = super().__new__(cls, value)
        obj.reason = reason
        return obj


@dataclass(slots=True)
class Problem:
    """A problem in a document: its file, line and message."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        """The problem as `path:line: message`."""
        return f"{formatting.location(self.path, self.line)}: {self.message}"


@dataclass(slots=True)
class Listed:
    """A number or identifier as listed: its file, line, text, and what it was listed for."""

    path: str
    line: int
    text: str
    reason: str
