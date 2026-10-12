"""The forms of a document's text the checks match: typography, numbers, identifiers, references.

`TOKEN` is a number of the text, `REF` a run, `index_version` or hash reference, and
`REFERENCES` the patterns whose digits are not numbers.
"""

from __future__ import annotations

import re

from ..formatting import MINUS

# The documents' typography, spelled out: code stays ASCII (tests/test_repo_hygiene.py).
TYPOGRAPHY = (
    ("<->", "\u2194"),  # left right arrow
    (">=", "\u2265"),  # greater-than or equal to
    ("--", "\u2013"),  # en dash, for ranges
    (" - ", f" {MINUS} "),  # minus sign, for differences
    ("nDCG'", "nDCG\u2032"),  # prime
)
WHITESPACE = re.compile(r"\s+")
NUMBER = r"[\u2212+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
# Numbers written as words; a hole matches them too, compared by value. "one"
# is left out: it is as often a pronoun ("each one's") or a definition ("one
# call per venue") as a count.
WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "half": "0.5", "twice": "2", "tenfold": "10",
}  # fmt: skip
WORD = r"(?i:\b(?:" + "|".join(WORDS) + r")\b)"
HOLE = f"({NUMBER}|{WORD})"
# A number with a unit attached ("12ms", "10x") is a number of the document.
UNIT = r"(?:ms|s|x|GB|MB|KB|min|h)"
# A number of the document is not part of an identifier, a version, a date or a path. A
# numeral or a number word joined to a word by a hyphen ("50-question", "five-question") is
# not one either.
_WORD_TOKEN = r"(?i:(?<![\w\-])(?:" + "|".join(WORDS) + r")(?![\w]|-\w))"
TOKEN = re.compile(
    rf"(?<![\w@.#\u00a7\u2032\-]){NUMBER}{UNIT}?(?![\w@\u2032]|\.\d|-\w)|{_WORD_TOKEN}"
)
# An ASCII hyphen used as a minus sign: the documents write U+2212.
HYPHEN_MINUS = re.compile(r"(?<![\w\-])-\d")
# Identifiers the spec checks with `ref`: run directories, index versions and
# hash prefixes. Found inside code spans too, which is where they are written.
REF = re.compile(r"results/\d{8}T\d{6}Z\w*|\b[0-9a-f]{16}\b|\b[0-9a-f]{8,}\u2026")
REF_HOLE = r"([0-9A-Za-z_]+\u2026?)"
REFERENCES = (
    r"ADR \d{4}",
    r"\u00a7\s?\d+[a-z]?",  # section sign
    r"\b(?:rule|rules|round|decision|Rule|Round|Decision) \d+",
    r"\(Sakai \d{4}[,)]",
)


def typeset(text: str) -> str:
    """`text` with the ASCII spelling the claims write replaced by the documents' characters."""
    for ascii_, typographic in TYPOGRAPHY:
        text = text.replace(ascii_, typographic)
    return text


def collapse(text: str) -> str:
    """`text` with every run of whitespace, line breaks included, as one space.

    Templates and documents are compared collapsed, so re-wrapping a paragraph
    changes nothing a check compares.
    """
    return WHITESPACE.sub(" ", text)


def is_reference(text: str, m: re.Match[str]) -> bool:
    """Whether a number match lies inside a reference (`ADR 0007`, a section sign, `rule 5`)."""
    return any(
        a <= m.start() < b
        for pattern in REFERENCES
        for a, b in (x.span() for x in re.finditer(pattern, text))
    )


# A Markdown table's header row and its rule, the lines before the first data row.
TABLE_HEADER_LINES = 2


def table_cells(line: str) -> list[str]:
    """The cells of a Markdown table row, stripped: `| a | b |` gives `["a", "b"]`."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_lines(text: str, start: int) -> list[tuple[int, str]]:
    """The first run of `|` lines of `text` from offset `start` on, each with its offset."""
    pos = start
    table: list[tuple[int, str]] = []
    for line in text[start:].split("\n"):
        if line.startswith("|"):
            table.append((pos, line))
        elif table:
            break
        pos += len(line) + 1
    return table
