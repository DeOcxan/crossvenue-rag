"""Markdown checks of published files: which files, and style rules 30 to 32 and 34.

`tests/test_repo_hygiene.py` runs the checks over every published Markdown file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What the repository publishes: these top-level files and directories, the
# two committed files of data/, and the committed runs that results/runs.json
# records. Everything else in a working copy (the virtualenv, caches, fetched
# data, uncommitted runs, local notes) is outside it.
PUBLISHED_FILES = (
    "README.md", "LICENSE", "Makefile", "pyproject.toml", "uv.lock",
    "docker-compose.yml", ".gitignore", ".vale.ini", ".env.example",
    "data/sources.yaml", "data/manifest.csv", "results/runs.json",
)  # fmt: skip
PUBLISHED_DIRS = (
    ".github", ".vale", "configs", "docs", "evals", "prompts", "scripts", "src", "tests",
)  # fmt: skip
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def published_files() -> list[Path]:
    files = [ROOT / name for name in PUBLISHED_FILES if (ROOT / name).is_file()]
    for name in PUBLISHED_DIRS:
        files += [
            path
            for path in (ROOT / name).rglob("*")
            if path.is_file()
            and path.name != ".DS_Store"
            and not CACHE_PARTS & set(path.relative_to(ROOT).parts)
        ]
    registry = ROOT / "results" / "runs.json"
    if registry.is_file():
        for run in json.loads(registry.read_text(encoding="utf-8"))["runs"]:
            files += [path for path in (ROOT / "results" / run).rglob("*") if path.is_file()]
    return sorted(files)


# Markdown that is not this project's prose: prompt files (versioned artifacts
# that are never edited, rule 44) and test fixtures (vendor-format input the
# tests parse as it is).
MARKDOWN_SKIP = {"prompts", "fixtures"}
INVISIBLE = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE (BOM)",
    "\u00a0": "NO-BREAK SPACE",
    "\u202f": "NARROW NO-BREAK SPACE",
    "\u00ad": "SOFT HYPHEN",
}
# Invisible characters are checked in every published text file this project
# writes, prompts and fixtures included; the committed runs under the top-level
# results/, which hold model output, are skipped.
INVISIBLE_SUFFIXES = {".md", ".yaml", ".yml", ".py", ".toml"}
INVISIBLE_SKIP = {"results"}

LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
ORDERED = re.compile(r"^\s*(\d+)[.)]\s")
RULE = re.compile(r"^ {0,3}([-*_])(?:\s*\1){2,}\s*$")
OTHER_MARKER = re.compile(r"^\s*[*+]\s")
# An ATX heading: one to six `#` and a space or the line's end, so `#123` is text.
HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]|$)")
# An item that opens with a bold or italic span, with or without a colon after it. The
# span may hold the other delimiter, as an identifier such as venue_split does.
LABEL = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(\*\*|__|\*|_)(?!\s)(?:(?!\1).)+?(?<!\s)\1")


def published(suffixes: set[str], skip: set[str], *, top_level: bool = False) -> list[Path]:
    """Published files with a suffix of `suffixes`, outside every directory `skip` names.

    A name in `skip` matches a directory at any depth, or with `top_level` only a
    directory at the repository root, so `results` then leaves `docs/results/` in.
    """

    def skipped(path: Path) -> bool:
        parts = path.relative_to(ROOT).parts[:-1]
        return bool(skip & set(parts[:1] if top_level else parts))

    return [path for path in published_files() if path.suffix in suffixes and not skipped(path)]


def markdown_files() -> list[Path]:
    return published({".md"}, MARKDOWN_SKIP)


def _body_start(lines: list[str]) -> int:
    """The index of the first line after YAML front matter, 0 without any."""
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                return i + 1
    return 0


def _line_kind(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if HEADING.match(line):
        return "heading"
    if stripped.startswith("|"):
        return "table"
    if RULE.match(line):
        return "rule"
    if LIST_ITEM.match(line):
        return "item"
    return "prose"


def prose_lines(text: str) -> list[tuple[int, str, str]]:
    """(line number, kind, line) for every line outside front matter, fenced
    code and HTML comments. Kinds: blank, heading, table, item, rule, prose."""
    lines = text.split("\n")
    out: list[tuple[int, str, str]] = []
    start = _body_start(lines)
    fence: str | None = None
    comment = False
    for number, line in enumerate(lines[start:], start + 1):
        m = FENCE.match(line)
        if fence:
            # A fence closes on a run of its own character at least as long as the opener.
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
        elif comment:
            comment = "-->" not in line
        elif line.strip().startswith("<!--"):
            comment = "-->" not in line.strip()
        elif m:
            fence = m.group(1)
        else:
            out.append((number, _line_kind(line), line))
    return out


def hard_wraps(text: str) -> list[int]:
    """Lines that continue the paragraph or list item on the line before."""
    lines = prose_lines(text)
    return [
        b[0]
        for a, b in pairwise(lines)
        if b[0] == a[0] + 1
        and a[1] in ("prose", "item")
        and (b[1] == "prose" or (a[1] == "prose" and not a[2][:1].isspace() and _continues(b[2])))
    ]


def _continues(line: str) -> bool:
    """A line such as "30. Clarifying ..." after a top-level paragraph line
    continues the paragraph: an ordered list can interrupt a paragraph only at 1
    (CommonMark). After an indented paragraph inside a list it is the next item."""
    m = ORDERED.match(line)
    return m is not None and m.group(1) != "1"


def horizontal_rules(text: str) -> list[int]:
    return [number for number, kind, _ in prose_lines(text) if kind == "rule"]


def other_list_markers(text: str) -> list[int]:
    """Unordered list items marked with `*` or `+` instead of `-`."""
    return [n for n, kind, line in prose_lines(text) if kind == "item" and OTHER_MARKER.match(line)]


def heading_outline(text: str) -> list[int]:
    """Headings that break rule 31's outline: a first heading that is not an H1,
    a second H1, or a heading more than one level below the heading before it."""
    found = []
    previous = 0
    for number, kind, line in prose_lines(text):
        match = HEADING.match(line) if kind == "heading" else None
        if match is None:
            continue
        level = len(match.group(1))
        if level > previous + 1 or (level == 1 and previous):
            found.append(number)
        previous = level
    return found


def labelled_lists(text: str) -> list[int]:
    """First line of every list of two or more items that all open with a bold
    or italic label."""
    found = []
    run: list[tuple[int, str]] = []
    for number, kind, line in [*prose_lines(text), (0, "end", "")]:
        if kind == "item":
            run.append((number, line))
            continue
        if kind == "blank" and run:
            # A blank line inside a loose list does not end it; the next
            # non-item line does.
            continue
        if len(run) >= 2 and all(LABEL.match(line) for _, line in run):
            found.append(run[0][0])
        run = []
    return found


def report(check: Callable[[str], list[int]]) -> list[str]:
    """Each `path:line` where `check` finds a problem, over every published Markdown file."""
    return [
        f"{path.relative_to(ROOT)}:{number}"
        for path in markdown_files()
        for number in check(path.read_text(encoding="utf-8"))
    ]
