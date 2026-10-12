"""Invariants of the Python code that no single module's tests cover.

The checks read the syntax tree (`tests.python_code`) of every file the code rules govern.
"""

from __future__ import annotations

import ast
import re

import pytest

from tests.published import ROOT
from tests.python_code import (
    PYTHON_TREES,
    broad_handlers,
    defined_functions,
    exempt,
    joined_responsibilities,
    long_sentences,
    python_files,
    record_tuples,
    rich_sinks,
    unreferenced,
)

# Rule 15's one exception (docs/code-style.md): the CLI error boundary.
BROAD_HANDLERS = ("src/crossvenue_rag/cli/errors.py",)


def test_only_the_cli_error_boundary_catches_every_exception():
    broad = [
        f"{path.relative_to(ROOT)}:{line}"
        for tree in PYTHON_TREES
        for path in sorted((ROOT / tree).rglob("*.py"))
        for line in broad_handlers(path.read_text(encoding="utf-8"))
    ]
    assert [b for b in broad if not b.startswith(BROAD_HANDLERS)] == []
    assert all(any(b.startswith(f) for b in broad) for f in BROAD_HANDLERS), "stale exception"


def test_the_broad_handler_check_walks_every_tree_the_code_rules_govern():
    text = (ROOT / "docs/code-style.md").read_text(encoding="utf-8")
    scope = next(line for line in text.splitlines() if line.startswith("Rules for the Python"))
    assert set(re.findall(r"`([^`/]+)/`", scope)) == set(PYTHON_TREES)


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ("try:\n    pass\nexcept:\n    raise\n", [3]),
        ("try:\n    pass\nexcept Exception:\n    raise\n", [3]),
        ("try:\n    pass\nexcept (ValueError, BaseException):\n    raise\n", [3]),
        ("import builtins\ntry:\n    pass\nexcept builtins.Exception:\n    raise\n", [4]),
        ("try:\n    pass\nexcept ValueError:\n    raise\n", []),
    ],
)
def test_a_bare_or_exception_wide_handler_is_reported(source: str, lines: list[int]):
    assert broad_handlers(source) == lines


# The modules whose strings reach a person as command output, help text, error
# messages or service error details (docs/code-style.md rule 25).
PERSON_FACING = ("src/crossvenue_rag/cli", "src/crossvenue_rag/service")


@pytest.mark.parametrize(
    "text",
    [
        "methods ['bm25', 'dense']",
        'methods ["bm25"]',
        "counts {'a': 1}",
        "ranks {1: 2}",
        "Now run the index",
        "Behaviours differ",
        "Labelled pool",
    ],
)
def test_the_output_defect_check_finds_string_lists_number_keyed_dicts_and_capitals(text: str):
    from tests.conftest import output_defects

    assert output_defects(text)


@pytest.mark.parametrize(
    "text",
    [
        "interval [+0.160, +0.400]",
        "the field `behaviour_correct` and [1, 2]",
        "the known set {bm25, dense}",
        "a line with `['bm25']` in code",
    ],
)
def test_the_output_defect_check_passes_intervals_code_spans_and_schema_fields(text: str):
    from tests.conftest import output_defects

    assert output_defects(text) == []


def test_the_cli_and_service_strings_have_no_output_defects():

    from tests.conftest import output_defects

    found = []
    for base in PERSON_FACING:
        for path in sorted((ROOT / base).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                # A logging format string (`%(levelname)-7s`) is not text a person reads.
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and "%(" not in node.value
                ):
                    found += [f"{path.name}:{node.lineno}: {d}" for d in output_defects(node.value)]
    assert not found, found


# Rich reads square brackets in a str as markup, so a value holding `[/x]` raises and one
# holding `[bold]` loses its brackets. A runtime value reaches rich only through
# `cli.app.say` (markup and highlighting off) or as a `Text`; these are the exceptions, by
# (file, function), with a function of None for the whole file.
RICH_SINK_EXEMPT: dict[tuple[str, str | None], str] = {
    ("src/crossvenue_rag/cli/app.py", "say"): "the one helper every value is printed through",
    ("src/crossvenue_rag/rich_text.py", "add_text_row"): "wraps every str cell in a Text",
}


def test_a_runtime_value_reaches_rich_only_through_say_or_as_a_text():
    found = [
        f"{path.relative_to(ROOT)}:{line} (in {function})"
        for tree in ("src", "scripts")
        for path in sorted((ROOT / tree).rglob("*.py"))
        for line, function in rich_sinks(path.read_text(encoding="utf-8"))
        if not exempt(str(path.relative_to(ROOT)), function, tuple(RICH_SINK_EXEMPT))
    ]
    assert found == [], found


def test_every_rich_exemption_names_a_file_and_function_that_exist():
    for where, function in RICH_SINK_EXEMPT:
        assert (ROOT / where).exists(), where
        if function is not None:
            assert f"def {function}(" in (ROOT / where).read_text(encoding="utf-8"), function


@pytest.mark.parametrize(
    ("source", "lines"),
    [
        ('console.print(f"[bold]{path}[/bold]")\n', [1]),
        ("console.print(name)\n", [1]),
        ('console.print("[yellow]x[/yellow] " + name)\n', [1]),
        ("console.log(str(path))\n", [1]),
        ('t = Table(title=f"rounds {n}")\n', [1]),
        ("p = Panel(answer, title=Text(t))\n", [1]),
        ("table.add_row(venue, Text(n))\n", [1]),
        ('table.add_column("n", footer=count(n))\n', [1]),
        ("Text.from_markup(message)\n", [1]),
        ('console.print("[dim]static[/dim]")\n', []),
        ('say(console, Text("results:", style="green"), str(path))\n', []),
        ('t = Table(title=Text(f"rounds {n}"))\ntable.add_row("total", Text(n))\n', []),
        ("x = math.log(p)\n", []),
        ("Prompt.ask(question)\n", [1]),
        ("progress.add_task(name)\n", [1]),
        ("status.update(message)\n", [1]),
        ("counts.update(other)\n", []),
    ],
)
def test_a_value_handed_to_rich_as_markup_is_reported(source: str, lines: list[int]):
    assert [line for line, _ in rich_sinks(source)] == lines


def test_no_docstring_or_comment_sentence_runs_over_35_words():
    found = [
        f"{path}:{line}: {opening}"
        for path, source in python_files()
        for line, function, opening in long_sentences(source)
    ]
    assert found == [], found


@pytest.mark.parametrize(
    ("source", "found"),
    [
        (f'"""{" ".join(["word"] * 36)}."""\n', 1),
        (f"# {' '.join(['word'] * 36)}.\n", 1),
        (f'"""Summary.\n\nArgs:\n    x: {" ".join(["word"] * 36)}.\n"""\n', 1),
        (f'"""{" ".join(["word"] * 20)}. {" ".join(["word"] * 20)}."""\n', 0),
        (f'"""The keys: {", ".join(f"`k{i}`" for i in range(40))}."""\n', 0),
        (f'"""Usage.\n\n    {" ".join(["word"] * 40)}\n\nAfter.\n"""\n', 0),
        (f'"""{" ".join(["word"] * 20)} e.g. {" ".join(["word"] * 20)}."""\n', 1),
        (f'X = 1\n"""{" ".join(["word"] * 36)}."""\n', 1),
    ],
)
def test_a_sentence_over_35_words_is_reported_unless_its_words_are_code(source: str, found: int):
    assert len(long_sentences(source)) == found


# Code rule 6's records that stay tuples, by (file, top-level function or class).
TUPLES_KEPT = {
    ("src/crossvenue_rag/corpus/dedup.py", "primary_key"): "a sort key",
    ("src/crossvenue_rag/dataset/sampling.py", "stratified_quotas"): "a sort key",
    ("src/crossvenue_rag/evaluation/machine.py", "MachineState"): "os.getloadavg's triple",
    ("src/crossvenue_rag/evaluation/machine.py", "RecordedState"): "machine.json's saved list",
}


def test_no_record_is_passed_as_a_tuple_of_three_or_more_fields():
    found = [
        f"{path}:{line}: {record}"
        for path, source in python_files(("src", "scripts"))
        for line, where, record in record_tuples(source)
        if (path, where) not in TUPLES_KEPT
    ]
    assert found == [], found
    kept = {(p, w) for p, s in python_files(("src", "scripts")) for _, w, _ in record_tuples(s)}
    assert set(TUPLES_KEPT) <= kept, "a kept tuple is gone: remove it from TUPLES_KEPT"


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("def f() -> tuple[int, str, float]: ...\n", ["tuple[int, str, float]"]),
        ("def f(rows: list[tuple[str, int, int]]) -> None: ...\n", ["tuple[str, int, int]"]),
        ("Row = tuple[int, int, float, str]\n", ["tuple[int, int, float, str]"]),
        ("x: dict[str, tuple[int, int, int]] = {}\n", ["tuple[int, int, int]"]),
        ("def f() -> tuple[int, str]: ...\n", []),
        ("def f() -> tuple[str, ...]: ...\n", []),
        ("x: dict[tuple[str, str, str], int] = {}\n", []),
    ],
)
def test_a_tuple_of_three_or_more_fields_is_reported_as_a_record(source: str, found: list[str]):
    assert [record for _, _, record in record_tuples(source)] == found


def test_no_module_docstring_joins_two_responsibilities_with_and():
    found = [
        f"{path}: {sentence}"
        for path, source in python_files(("src", "scripts"))
        if not path.endswith("__init__.py")
        if (sentence := joined_responsibilities(source))
    ]
    assert found == [], found


@pytest.mark.parametrize(
    ("docstring", "joined"),
    [
        ("The schema of a dataset, and its file I/O.", True),
        ("Reading and writing JSON Lines files.", True),
        (
            "OpenRouter's protocol: the request body, the POST with its retries, and the models.",
            False,
        ),
        ("The cost of the answer runs: dollars, call time.", False),
        ("docs/x.md: reading and writing files.", True),
        ("docs/x.md, the judged answers: H-v2 and venue attribution.", False),
    ],
)
def test_a_module_docstring_with_and_before_its_colon_is_reported(docstring: str, *, joined: bool):
    assert (joined_responsibilities(f'"""{docstring}"""\n') is not None) is joined


# Code rule 4's exception for scripts: a script's entry point is `main`, as the CLI's is.
SCRIPT_NAMES_SHARED = {"main"}


def test_no_script_function_reuses_a_package_function_name():
    package = set().union(*(defined_functions(s) for _, s in python_files(("src",))))
    scripts = set().union(*(defined_functions(s) for _, s in python_files(("scripts",))))
    assert scripts & package == SCRIPT_NAMES_SHARED


# Code rule 23's dead code that stays: a name only something outside Python calls.
UNREFERENCED_KEPT = {
    "src/crossvenue_rag/service/app.py app_factory": "uvicorn's --factory entry point (make serve)",
}


def test_every_definition_is_used_outside_itself_and_its_own_tests():
    files = dict(python_files())
    found = {entry.split(":")[0] + " " + entry.split(" ")[1] for entry in unreferenced(files)}
    assert found == set(UNREFERENCED_KEPT)


def test_a_definition_only_its_own_test_uses_is_reported():
    files = {
        "src/crossvenue_rag/m.py": "LIMIT = 3\nUSED = 1\n\ndef dead():\n    return 1\n",
        "src/crossvenue_rag/n.py": "from .m import USED\n\nclass C:\n    def unused(self):\n"
        "        return USED\n",
        "tests/test_m.py": "from crossvenue_rag.m import dead\n\ndead()\n",
    }
    assert unreferenced(files) == [
        "src/crossvenue_rag/m.py:1 LIMIT",
        "src/crossvenue_rag/m.py:4 dead",
        "src/crossvenue_rag/n.py:3 C",
        "src/crossvenue_rag/n.py:4 C.unused",
    ]


def test_a_bare_name_in_another_module_is_a_local_and_no_use():
    files = {
        "src/crossvenue_rag/m.py": "def dead():\n    return 1\n",
        "src/crossvenue_rag/n.py": (
            "from . import m\n\ndef g():\n    dead = 2\n    return dead + m.X\n"
        ),
    }
    assert "src/crossvenue_rag/m.py:1 dead" in unreferenced(files)
