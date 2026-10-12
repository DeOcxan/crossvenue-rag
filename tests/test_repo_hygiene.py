"""Repository-level invariants that belong to no single module.

Code and CLI output stay ASCII apart from a short allowlist.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

from crossvenue_rag.reporting.compute.repository import suite_level_test_files
from crossvenue_rag.reporting.sources import Sources
from tests.published import (
    INVISIBLE,
    INVISIBLE_SKIP,
    INVISIBLE_SUFFIXES,
    ROOT,
    hard_wraps,
    heading_outline,
    horizontal_rules,
    labelled_lists,
    other_list_markers,
    published,
    published_files,
    report,
)
from tests.python_code import modules_without_tests

# Source, configuration and CI. Documentation is deliberately excluded: prose
# gets em dashes, Greek letters for statistics and box-drawing for diagrams, and
# none of that can reach a terminal or a log line.
CODE_ROOTS = ("src", "tests", "configs", ".github", "scripts")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}

# Every non-ASCII character allowed in code, with the reason it is allowed.
# A character not on this list is taken for a paste accident: a CJK
# character once reached a CLI message this way, through a copied sentence.
ALLOWED: dict[str, str] = {
    "█": "FULL BLOCK: the bar of the chunk-length histogram; it is the graphic",
    "ç": "in a French heading inside the llms.txt fixture, which exists to test "
    "that non-English GitBook sections are filtered out",
}


def code_files() -> list[Path]:
    files: list[Path] = []
    for root in CODE_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        files += [
            path
            for path in base.rglob("*")
            if path.is_file() and path.suffix in CODE_SUFFIXES and "__pycache__" not in path.parts
        ]
    files += [ROOT / "Makefile", ROOT / "pyproject.toml"]
    return [f for f in files if f.exists()]


def test_code_and_cli_strings_are_ascii():
    """Typographic punctuation has exact ASCII equivalents, so using it in code
    adds nothing and leaves a stray glyph that is hard to spot. Prose is exempt;
    anything that can reach a terminal, a log or a config value is not."""
    offenders: list[str] = []
    for path in code_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for char in line:
                if ord(char) < 128 or char in ALLOWED:
                    continue
                name = unicodedata.name(char, "unnamed")
                where = f"{path.relative_to(ROOT)}:{number}"
                offenders.append(f"{where} U+{ord(char):04X} {name} -> {line.strip()[:70]}")

    assert not offenders, "non-ASCII in code:\n  " + "\n  ".join(offenders)


def test_every_allowlisted_character_is_used_in_code():
    """An allowlist nobody prunes becomes a permission slip. If an entry stops
    appearing, it should be deleted rather than left as precedent."""
    # This file holds every allowlisted character as a key of ALLOWED, so it is left out.
    here = Path(__file__).resolve()
    others = [p for p in code_files() if p.resolve() != here]
    everything = "\n".join(p.read_text(encoding="utf-8") for p in others)
    unused = [f"U+{ord(c):04X}" for c in ALLOWED if c not in everything]
    assert not unused, f"allowlisted but unused, so remove it: {unused}"


def test_markdown_paragraphs_are_one_line():
    """Rule 30: a paragraph or a list item is one line, so a diff shows the
    sentence that changed and `git diff --word-diff` shows the words."""
    offenders = report(hard_wraps)
    assert not offenders, "hard-wrapped Markdown:\n  " + "\n  ".join(offenders)


def test_markdown_has_no_horizontal_rules():
    """Rule 34: headings separate sections."""
    offenders = report(horizontal_rules)
    assert not offenders, "horizontal rule in Markdown:\n  " + "\n  ".join(offenders)


def test_markdown_lists_have_no_label_on_every_item():
    """Rule 32: items that each carry a label and attributes are a table."""
    offenders = report(labelled_lists)
    assert not offenders, "list with a bold or italic label on every item:\n  " + "\n  ".join(
        offenders
    )


def test_markdown_lists_use_hyphens():
    """Rule 32: lists use `-`."""
    offenders = report(other_list_markers)
    assert not offenders, "list item marked with * or +:\n  " + "\n  ".join(offenders)


def test_markdown_headings_form_one_outline():
    """Rule 31: one H1 per file, first, and no skipped heading levels."""
    offenders = report(heading_outline)
    assert not offenders, "heading out of outline:\n  " + "\n  ".join(offenders)


def test_published_text_files_have_no_invisible_characters():
    """Zero-width, non-breaking and soft-hyphen characters look like ordinary
    spaces or nothing, and break search, diff and the reproduce templates."""
    offenders = []
    for path in published(INVISIBLE_SUFFIXES, INVISIBLE_SKIP, top_level=True):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for char in line:
                if char in INVISIBLE:
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{number} U+{ord(char):04X} {INVISIBLE[char]}"
                    )
    assert not offenders, "invisible characters:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize(
    ("text", "wraps"),
    [
        ("one line\n\nanother\n", []),
        ("a wrapped\nparagraph\n", [2]),
        ("- item\n  continued\n", [2]),
        ("- item\n- item\n", []),
        ("```\ncode\ncode\n```\n", []),
        ("| a |\n| b |\n", []),
        ("---\ntitle: x\nmore: y\n---\n\ntext\n", []),
        ("# Heading\ntext\n", []),
        ("a paragraph that ends with\n30. and goes on\n", [2]),
        ("1. first\n2. second\n", []),
        ("a paragraph that ends with\n#123 and goes on\n", [2]),
        ("<!--\n```\n-->\n\na wrapped\nparagraph\n", [6]),
        ("````\n```\ncode\ncode\n````\n", []),
    ],
)
def test_a_line_that_continues_a_paragraph_or_an_item_is_a_hard_wrap(text: str, wraps: list[int]):
    assert hard_wraps(text) == wraps


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("- **A**: one\n- **B**: two\n", [1]),
        ("- *A:* one\n- *B:* two\n", [1]),
        ("- **A**: one\n- plain two\n", []),
        ("- **A** is bold\n- **B** is bold\n", [1]),
        ("- _A_ one\n- _B_ two\n", [1]),
        ("- one **A**\n- two **B**\n", []),
        ("- **venue_split** one\n- **bm25** two\n", [1]),
        ("- *venue_split* one\n- *bm25* two\n", [1]),
        ("- __venue_split__ one\n- __bm25__ two\n", [1]),
        ("- `venue_split` one\n- `bm25` two\n", []),
    ],
)
def test_a_list_whose_items_all_open_with_bold_or_italic_is_reported(text: str, found: list[int]):
    assert labelled_lists(text) == found


def test_a_thematic_break_is_a_horizontal_rule_and_front_matter_is_not():
    assert horizontal_rules("text\n\n---\n\nmore\n") == [3]
    assert horizontal_rules("---\ntitle: x\n---\n\ntext\n") == []


def test_list_items_marked_with_a_star_or_a_plus_are_reported():
    assert other_list_markers("- one\n* two\n+ three\n1. four\n") == [2, 3]
    assert other_list_markers("*italic* text\n\n```\n* code\n```\n") == []


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("# Title\n\n## Part\n\n### Detail\n\n## Next\n", []),
        ("## Part\n", [1]),
        ("# Title\n\n# Another\n", [3]),
        ("# Title\n\n### Detail\n", [3]),
        ("# Title\n\n```\n# comment\n```\n", []),
        ("# Title\n\n#123 is an issue\n", []),
    ],
)
def test_a_first_heading_below_h1_a_second_h1_or_a_skipped_level_breaks_the_outline(
    text: str, found: list[int]
):
    assert heading_outline(text) == found


# The local instruction file, the local tool-state directory and the local
# notes directory are excluded at publish time; no published file may point at
# them. The patterns are assembled so that this file does not match itself.
LOCAL_ONLY = re.compile(
    "|".join(
        [
            re.escape("CLAUDE" + ".md"),
            r"(?<![\w-])" + re.escape("." + "claude") + r"(?:/|\b)",
            r"(?<![\w-])" + re.escape("audit" + "/"),
        ]
    ),
    re.IGNORECASE,
)


def test_no_published_file_refers_to_local_only_paths():
    offenders = []
    for path in published_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if LOCAL_ONLY.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, "reference to a local-only path:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize(
    ("line", "found"),
    [
        ("see " + "CLAUDE" + ".md for rules", True),
        ("state in ." + "claude/settings.json", True),
        ("notes in " + "audit" + "/REPORT.md", True),
        ("model anthropic/claude-opus-5.5", False),
        ("the number audit of the spec", False),
        ("`doc.audit()` reports", False),
    ],
)
def test_the_local_only_pattern_matches_the_local_paths_and_no_lookalike(line: str, *, found: bool):
    assert bool(LOCAL_ONLY.search(line)) is found


# Rule 27's exceptions (docs/code-style.md), the only ones: suite-level test
# files, which exercise no single module, and modules with no test file of
# their name, a directory entry covering every module under it.
SUITE_LEVEL = {
    "tests/reporting/test_claims_adr.py",
    "tests/test_code_hygiene.py",
    "tests/test_input_domains.py",
    "tests/reporting/test_mutations.py",
    "tests/test_pipeline_smoke.py",
    "tests/test_repo_hygiene.py",
}


UNTESTED_MODULES = (
    # Checked against the documents by `make reproduce` and the mutation test.
    "src/crossvenue_rag/reporting/claims/",
    # No behavior: only constants or Protocol classes.
    "src/crossvenue_rag/reporting/run_ids.py",
    "src/crossvenue_rag/retrieval/protocols.py",
)


def test_every_test_file_is_named_after_the_module_it_tests():
    without = set(suite_level_test_files(Sources(ROOT)))
    orphans = sorted(without - SUITE_LEVEL)
    assert not orphans, f"test files with no module of their name: {orphans}"
    stale = sorted(SUITE_LEVEL - without)
    assert not stale, f"listed as suite-level but gone or named after a module: {stale}"


def test_every_module_has_a_test_file_unless_rule_27_exempts_it():
    untested = modules_without_tests(ROOT)
    missing = [m for m in untested if not m.startswith(UNTESTED_MODULES)]
    assert not missing, f"modules with no test file of their name: {missing}"
    stale = [m for m in UNTESTED_MODULES if not m.endswith("/") and m not in untested]
    assert not stale, f"listed as untested but gone or tested: {stale}"


def test_the_exceptions_are_the_ones_rule_27_names():
    text = (ROOT / "docs/code-style.md").read_text(encoding="utf-8")
    # Rule 27's block runs to the next numbered rule or heading, wherever the rules are.
    rule = re.split(r"\n(?:\d+\. |## )", text.split("\n27. ", 1)[1], maxsplit=1)[0]
    items = [line for line in rule.splitlines() if line.lstrip().startswith("- ")]
    named = {name for line in items for name in re.findall(r"`([^`]+)`", line)}
    named -= {"make reproduce"}
    listed = {Path(f).name for f in SUITE_LEVEL}
    listed |= {m.removeprefix("src/crossvenue_rag/") for m in UNTESTED_MODULES}
    assert named == listed


def test_modules_without_tests_are_those_no_test_file_is_named_after(tmp_path):
    for path in (
        "src/crossvenue_rag/labeling/__init__.py",
        "src/crossvenue_rag/labeling/judge.py",
        "src/crossvenue_rag/labeling/pool.py",
        "src/crossvenue_rag/stats.py",
        "tests/labeling/test_judge.py",
        "tests/test_pool.py",
    ):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("", encoding="utf-8")
    assert modules_without_tests(tmp_path) == [
        "src/crossvenue_rag/labeling/pool.py",
        "src/crossvenue_rag/stats.py",
    ]
