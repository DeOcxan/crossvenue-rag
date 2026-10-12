"""Checking the numbers a Markdown document states against values recomputed from their sources.

Each problem a check finds is recorded with its file and line.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

from .. import formatting
from .outcomes import SKIP, Listed, Param, Problem
from .patterns import (
    HOLE,
    HYPHEN_MINUS,
    NUMBER,
    REF,
    REF_HOLE,
    REFERENCES,
    TABLE_HEADER_LINES,
    TOKEN,
    WORD,
    WORDS,
    collapse,
    is_reference,
    table_cells,
    table_lines,
    typeset,
)
from .text_spans import Collapsed, Found, Spans

# A Markdown table's fewest lines: the header, the rule and one row.
TABLE_MIN_LINES = TABLE_HEADER_LINES + 1


def _pattern(template: str, hole: str) -> re.Pattern[str]:
    """A claim template as a pattern: its text typeset and collapsed, each `{}` as `hole`."""
    return re.compile(hole.join(re.escape(collapse(p)) for p in typeset(template).split("{}")))


@functools.cache
def template_regex(template: str) -> re.Pattern[str]:
    """The compiled pattern of a claim template, `{}` holes matching one number each.

    Cached: every run of a document's claims compiles the same templates.

    Raises:
        ValueError: the template holds a number outside its holes.
    """
    for part in typeset(template).split("{}"):
        stray = [m.group() for m in TOKEN.finditer(part) if not is_reference(part, m)]
        if stray:
            raise ValueError(f"template has a literal number {stray[0]!r}: {template!r}")
    return _pattern(template, HOLE)


@functools.cache
def literal_regex(template: str) -> re.Pattern[str]:
    """The compiled pattern of literal text with no holes, as `definition` matches it.

    Literal text that ends in a digit does not match the start of a longer number:
    `is a 502` does not cover `5023`, nor `Opus 5` cover `5.5`.
    """
    # A literal `{}` is text here, so it is matched as written.
    text = _pattern(template, re.escape("{}")).pattern
    boundary = r"(?![\d]|[.,]\d)" if text[-1:].isdigit() else ""
    return re.compile(text + boundary)


@dataclass(frozen=True)
class _Item:
    """A list item of a document, at its offsets in the text."""

    start: int
    end: int
    text: str


@dataclass
class Doc:
    """One Markdown file: its text, its problems, its listed numbers and the count checked.

    The claims (`reporting.claims`) pass the numbers of a document to the `Doc` with the
    text they are written in, through `check`, `table`, `given`, `unsaved` and the other
    methods; `audit` then reports every number no call covered. Numbers inside code, link
    targets, identifiers (`recall@10`, `cv-0045`, `v0.2.1`, dates), the reference patterns
    in `REFERENCES` and numbers joined to a word by a hyphen (`50-question`,
    `patterns.TOKEN`) are not numbers of the document.

    Templates are matched with every run of whitespace collapsed to one space, in the
    template and in the document, so a line break or an indent is never part of a claim.
    Tables, lists and the regexes of `definitions_matching` are read from the text as
    written, since their line structure is what they match.
    """

    path: str
    text: str
    problems: list[Problem] = field(default_factory=list)
    given_numbers: list[Listed] = field(default_factory=list)
    unsaved_numbers: list[Listed] = field(default_factory=list)
    compared_numbers: list[Listed] = field(default_factory=list)
    identifiers: list[Listed] = field(default_factory=list)  # checked with `ref`
    _covered: list[tuple[int, int]] = field(default_factory=list)
    _refs: list[tuple[int, int]] = field(default_factory=list)
    _masked: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def read(cls, path: str | Path, root: str | Path) -> Doc:
        """The document at `path` under `root`, with its non-number spans masked."""
        return cls.from_text(str(path), (Path(root) / path).read_text(encoding="utf-8"))

    @classmethod
    def from_text(cls, path: str, text: str) -> Doc:
        """The document `text`, named `path`, with its non-number spans masked."""
        doc = cls(path, text)
        doc._mask()
        return doc

    @property
    def checked(self) -> int:
        """How many values the checks compared: holes of a template and table cells."""
        return len(self.compared_numbers)

    @property
    def refs(self) -> int:
        """How many identifiers `ref` compared, which are not numbers."""
        return len(self.identifiers)

    def line_of(self, pos: int) -> int:
        """The 1-based line number of offset `pos` in the text."""
        return self.text.count("\n", 0, pos) + 1

    @functools.cached_property
    def _flat(self) -> Collapsed:
        return Collapsed(self.text)

    @property
    def collapsed(self) -> str:
        """The text with every run of whitespace as one space: what templates are matched on."""
        return self._flat.text

    def _find(self, regex: re.Pattern[str]) -> list[Found]:
        """Every match of `regex` in the collapsed text, with offsets in the text as written."""
        return [Found(m, self._flat) for m in regex.finditer(self._flat.text)]

    def _mask(self) -> None:
        """Spans whose digits are not numbers of the document."""
        patterns = [
            r"```.*?```",  # fenced code
            r"`[^`]*`",  # inline code, a line break inside it included
            r"\]\([^)]*\)",  # link targets
            r"<!--.*?-->",
            *REFERENCES,
        ]
        for pattern in patterns:
            self._masked.extend(f.span for f in self._find(re.compile(pattern, flags=re.S)))

    @functools.cached_property
    def _mask_spans(self) -> Spans:
        # Built once, after `_mask` has collected every span: a mutation test calls
        # `is_masked` once per token.
        return Spans(self._masked)

    def is_masked(self, pos: int) -> bool:
        """Whether offset `pos` lies in a span whose digits are not numbers (code, a link)."""
        return pos in self._mask_spans

    def covered_spans(self) -> tuple[tuple[int, int], ...]:
        """The spans the checks so far covered, in call order."""
        return tuple(self._covered)

    def ref_spans(self) -> tuple[tuple[int, int], ...]:
        """The spans `ref` checked so far, in call order."""
        return tuple(self._refs)

    def _regex(self, template: str) -> re.Pattern[str]:
        return template_regex(template)

    def check(self, template: str, *values: str) -> None:
        """Every place `template` matches holds exactly `values`.

        `template` is literal text with `{}` holes, one number per hole, matched with
        whitespace collapsed (as the class docstring describes). A `Param`
        value is matched too, then listed as given (with its reason and the template)
        instead of counted, when every match agrees.
        """
        before = len(self.problems)
        found = self._match(template, values)
        params = [v for v in values if isinstance(v, Param)]
        if not params:
            return
        if found and len(self.problems) == before:
            line = self.line_of(found[0].start)
            for v in params:
                self.given_numbers.append(
                    Listed(self.path, line, str(v), f"{v.reason}: {template}")
                )

    def caption(self, template: str, runs: Sequence[str], *values: str) -> None:
        """A table caption (style rule 9): its run ids and its numbers.

        Its run ids, written where `template` has `{run}`, are checked with `ref`, and its
        numbers, the `{}` holes, with `check`.

        Raises:
            ValueError: `template` has more `{}` holes than `values`.
        """
        if template.count("{}") > len(values):
            raise ValueError(
                f"{template!r} has more holes than the {formatting.count(len(values))} values"
            )
        it = iter(values)
        self.ref(re.sub(r"\{\}", lambda _: next(it), template).replace("{run}", "{}"), *runs)
        text = template
        for run in runs:
            text = text.replace("{run}", run, 1)
        self.check(text, *values)

    def definitions(self, texts: Iterable[str], reason: str) -> None:
        """`definition` for each of `texts`, all with one reason."""
        for text in texts:
            self.definition(text, reason=reason)

    def definitions_matching(self, pattern: str, reason: str) -> None:
        """`definition` for every match of the regex `pattern` in the text."""
        for match in re.finditer(pattern, self.text):
            self.definition(match.group(), reason=reason)

    def given(self, template: str, *values: str, reason: str) -> None:
        """Parameters or references: matched like `check`, then listed with `reason`."""
        found = self._match(template, values, compared=False)
        for f in found:
            self.given_numbers.append(
                Listed(self.path, self.line_of(f.start), " ".join(values), reason)
            )

    def claim(self, template: str, *values: str, holds: bool, what: str) -> None:
        """Numbers the text states as a bound or a condition: "no chunk above 0.7".

        The values are matched, and `holds` is the recomputed truth.
        """
        for f in self._match(template, values):
            if not holds:
                self.problems.append(
                    Problem(self.path, self.line_of(f.start), f"does not hold: {what}")
                )

    def unsaved(self, template: str, reason: str) -> None:
        """Numbers compared with no value: each match of `template` is listed with `reason`.

        They are listed on every run, never passed silently.
        """
        found = self._find(_pattern(template, f"(?:{NUMBER}|{WORD})"))
        if not found:
            self._not_found(template)
        for f in found:
            self._covered.append(f.span)
            self.unsaved_numbers.append(Listed(self.path, self.line_of(f.start), f.group, reason))

    def _match(self, template: str, values: Sequence[str], *, compared: bool = True) -> list[Found]:
        """Every match of `template`, each hole compared with its value; the matches.

        With `compared`, each hole whose value is not a `Param` is listed as compared.
        """
        found = self._find(self._regex(template))
        if not found:
            self._not_found(template)
            return []
        for f in found:
            for i, (want, have) in enumerate(zip(values, f.groups, strict=True)):
                if want != WORDS.get(have.lower(), have):
                    self._mismatch(f, i, have, want, template)
                if compared and not isinstance(want, Param):
                    line = self.line_of(f.group_start(i + 1))
                    self.compared_numbers.append(Listed(self.path, line, have, template))
            self._covered.append(f.span)
        return found

    def _mismatch(self, found: Found, i: int, have: str, want: str, template: str) -> None:
        """Hole `i` of a match holds `have` where `want` was recomputed: a problem at its line."""
        self.problems.append(
            Problem(
                self.path,
                self.line_of(found.group_start(i + 1)),
                f"{have} in the text, {want} recomputed "
                f"(hole {formatting.own_form(i + 1)} of {template!r})",
            )
        )

    def ref(self, template: str, *values: str) -> None:
        """Identifiers written in the text: a run directory, an `index_version`, a hash prefix.

        The holes of `template` match identifiers: run ids, index versions and hash prefixes.
        They are listed in `identifiers`, apart from the values `compared_numbers` lists.
        """
        found = self._find(_pattern(template, REF_HOLE))
        if not found:
            self._not_found(template)
        for f in found:
            for i, (want, have) in enumerate(zip(values, f.groups, strict=True)):
                if want != have:
                    self._mismatch(f, i, have, want, template)
                line = self.line_of(f.group_start(i + 1))
                self.identifiers.append(Listed(self.path, line, have, template))
            self._refs.append(f.span)

    def definition(self, template: str, reason: str) -> None:
        """Literal text whose numbers define rather than measure ("one call per venue").

        Covered and listed with the reason.
        """
        found = self._find(literal_regex(template))
        if not found:
            self._not_found(template)
        for f in found:
            self._covered.append(f.span)
            self.given_numbers.append(Listed(self.path, self.line_of(f.start), f.group, reason))

    def items(self, heading: str, expected: Sequence[str], reason: str) -> None:
        """The list under the line `heading` is exactly `expected`, item for item.

        An item is a line starting with `- `, and the list ends at the first line that is not
        one. Each written item is covered and listed with `reason`; an item missing from the
        list, an item not expected and an expected item out of order are each a problem.
        """
        # The offset of `heading` in the text, a first line included.
        start = f"\n{self.text}".find(f"\n{heading}\n")
        if start < 0:
            self._not_found(heading)
            return
        written: list[_Item] = []
        pos = start + len(heading) + 1
        for line in self.text[pos:].split("\n"):
            if not line.startswith("- "):
                break
            written.append(_Item(pos, pos + len(line), line))
            pos += len(line) + 1
        heading_line = self.line_of(start)
        texts = [item.text for item in written]
        for item in expected:
            if item not in texts:
                self.problems.append(Problem(self.path, heading_line, f"missing item: {item!r}"))
        for w in written:
            if w.text in expected:
                self._covered.append((w.start, w.end))
                self.given_numbers.append(Listed(self.path, self.line_of(w.start), w.text, reason))
            else:
                self.problems.append(
                    Problem(self.path, self.line_of(w.start), f"unexpected item: {w.text!r}")
                )
        present = [item for item in expected if item in texts]
        if [t for t in texts if t in expected] != present:
            self.problems.append(
                Problem(self.path, heading_line, f"items out of order: {heading!r}")
            )

    def _not_found(self, template: str) -> None:
        anchor = collapse(max(typeset(template).split("{}"), key=len)).strip()
        pos = self._flat.text.find(anchor) if anchor else -1
        line = self.line_of(self._flat.at(pos)) if pos >= 0 else 0
        self.problems.append(Problem(self.path, line, f"text not found: {template!r}"))

    def table(
        self,
        after: str,
        rows: Mapping[str, Sequence[object]],
    ) -> None:
        """Compares the first table after `after` with `rows`.

        Every row is looked up by its first cell and compared cell by cell. `rows` maps a
        first cell to the other cells, formatted; `SKIP` for a text cell.
        """
        after = typeset(after)
        rows = {typeset(k): v for k, v in rows.items()}
        flat_start = self._flat.text.find(collapse(after))
        if flat_start < 0:
            self._not_found(after)
            return
        start = self._flat.at(flat_start)
        table = table_lines(self.text, start)
        if len(table) < TABLE_MIN_LINES:
            self.problems.append(Problem(self.path, self.line_of(start), "no table found"))
            return
        seen = set()
        for row_pos, line in table[TABLE_HEADER_LINES:]:
            key = table_cells(line)[0]
            seen.add(key)
            if key not in rows:
                self.problems.append(
                    Problem(self.path, self.line_of(row_pos), f"row {key!r} is not recomputed")
                )
            else:
                self._table_row(row_pos, line, list(rows[key]))
        for key in rows:
            if key not in seen:
                self.problems.append(
                    Problem(self.path, self.line_of(start), f"row {key!r} missing from the table")
                )

    def _table_row(self, row_pos: int, line: str, want: list[object]) -> None:
        """Compares one table row's cells after the first with the recomputed `want`."""
        cells = table_cells(line)
        key, have = cells[0], cells[1:]
        if len(want) != len(have):
            self.problems.append(
                Problem(
                    self.path,
                    self.line_of(row_pos),
                    f"row {key!r}: {formatting.plural(len(have), 'cell')}, "
                    f"{formatting.count(len(want))} recomputed",
                )
            )
            return
        offsets = _cell_offsets(line)
        self._covered.append((row_pos + offsets[0][0], row_pos + offsets[0][1]))
        for i, (w, h) in enumerate(zip(want, have, strict=True)):
            if w is SKIP:
                continue
            w = typeset(str(w))
            if w != h:
                self.problems.append(
                    Problem(
                        self.path,
                        self.line_of(row_pos),
                        f"row {key!r}, column {formatting.own_form(i + 2)}: "
                        f"{h!r} in the text, {w!r} recomputed",
                    )
                )
            a, b = offsets[i + 1]
            self._covered.append((row_pos + a, row_pos + b))
            column = f"row {key!r}, column {formatting.own_form(i + 2)}"
            self.compared_numbers.append(Listed(self.path, self.line_of(row_pos), h, column))

    def audit(self) -> None:
        """Every number no check covered is a problem, in headings too.

        So is a run, `index_version` or hash prefix no `ref` checked, and a minus sign written
        as an ASCII hyphen.
        """
        covered = Spans(self.covered_spans())
        for m in TOKEN.finditer(self.text):
            pos = m.start()
            if self.is_masked(pos) or pos in covered:
                continue
            self.problems.append(
                Problem(
                    self.path,
                    self.line_of(pos),
                    f"number {m.group()!r} is not recomputed by any check",
                )
            )
        refs = Spans(self.ref_spans())
        for m in REF.finditer(self.text):
            if m.start() not in refs:
                self.problems.append(
                    Problem(
                        self.path,
                        self.line_of(m.start()),
                        f"{m.group()!r} is not checked against its source",
                    )
                )
        for m in HYPHEN_MINUS.finditer(self.text):
            if not self.is_masked(m.start()):
                self.problems.append(
                    Problem(
                        self.path,
                        self.line_of(m.start()),
                        f"{m.group()!r}: a minus sign is written U+2212",
                    )
                )


def _cell_offsets(line: str) -> list[tuple[int, int]]:
    """(start, end) of each cell's content within the line.

    A row may leave out its closing pipe, as GFM allows; its last cell ends the line.
    """
    bars = [i for i, ch in enumerate(line) if ch == "|"]
    if not line.rstrip().endswith("|"):
        bars.append(len(line.rstrip()))
    return [(a + 1, b) for a, b in pairwise(bars)]
