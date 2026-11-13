"""The repository's own files, as the documents describe them."""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys
import tokenize
import tomllib
from pathlib import Path, PurePosixPath

from ...errors import UserError
from ...run_files import file_sha256
from ..sources import Sources
from .memo import per_sources

# A stated size of the code is "about" a size: the counted size to the nearest thousand.
SOURCE_LINES_ROUNDING = 1000
# How design.md states the size of the code: "about <count> lines under `src/`".
STATED_SOURCE_LINES = r"about ([\d,]+) lines under"
# Code-style rule 2: above about this many lines of code, a module is split.
MODULE_CODE_LINES_BOUND = 400
# The trees code-style rule 2 governs.
CODE_TREES = ("src", "tests", "scripts")
# The package's and the scripts' directories and the project's settings, relative to the
# repository.
PACKAGE = "src/crossvenue_rag"
SCRIPTS = "scripts"
PYPROJECT = "pyproject.toml"
# The test configuration that holds the fixture corpus.
FIXTURE_CONFTEST = "tests/conftest.py"
# The nodes whose first statement may be a docstring, and the tokens that are not code.
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_NOT_CODE = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
    }
)


def source_lines(src: Sources) -> int:
    """Lines of every `.py` file under `src/`, blank and comment lines included."""
    return sum(len(p.read_bytes().splitlines()) for p in src.path("src").rglob("*.py"))


def code_lines(text: str) -> int:
    """Lines of code of a module's source: non-blank, non-comment lines, docstrings excluded.

    Raises:
        SyntaxError: `text` is not valid Python.
    """
    docstrings: set[int] = set()
    for node in ast.walk(ast.parse(text)):
        body = getattr(node, "body", None)
        if isinstance(node, _DOCUMENTED) and body:
            first = body[0]
            value = first.value if isinstance(first, ast.Expr) else None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.update(range(first.lineno, (first.end_lineno or 0) + 1))
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type not in _NOT_CODE:
            lines.update(range(token.start[0], token.end[0] + 1))
    return len(lines - docstrings)


@per_sources
def largest_module(src: Sources) -> tuple[str, int]:
    """(path, lines of code) of the module with the most `code_lines` in `CODE_TREES`."""
    root = src.path(".")
    sizes = [
        (code_lines(p.read_text(encoding="utf-8")), str(p.relative_to(root)))
        for tree in CODE_TREES
        for p in src.path(tree).rglob("*.py")
    ]
    lines, path = max(sizes)
    return path, lines


def stated_source_lines(text: str) -> int:
    """The size of the code a document states ("about <count> lines under"); 0 if none."""
    written = re.search(STATED_SOURCE_LINES, text)
    return int(written.group(1).replace(",", "")) if written else 0


def source_lines_agree(stated: int, counted: int) -> bool:
    """Whether `stated` is `counted` to the nearest `SOURCE_LINES_ROUNDING`, halves rounded up."""
    nearest = (counted + SOURCE_LINES_ROUNDING // 2) // SOURCE_LINES_ROUNDING
    return stated == nearest * SOURCE_LINES_ROUNDING


def subpackage_count(src: Sources) -> int:
    """Subpackages of `crossvenue_rag`: directories with an `__init__.py`."""
    package = src.path(PACKAGE)
    return sum(1 for p in package.iterdir() if (p / "__init__.py").exists())


def module_tree(src: Sources, path: str) -> ast.Module:
    """The syntax tree of the Python file at `path`, relative to the repository."""
    return ast.parse(src.path(path).read_text(encoding="utf-8"))


def _bound(node: ast.stmt, name: str) -> ast.expr | None:
    """The value `node` binds to `name` when it is `name = ...` or `name: T = ...`; else None."""
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets
    ):
        return node.value
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.value if node.target.id == name else None
    return None


def module_literal(src: Sources, path: str, name: str) -> object:
    """The literal bound to `name` at module level in the file at `path`, or where it imports it.

    `path` is relative to the repository; a relative import
    `from ..package.module import name` is resolved against `path`.

    Raises:
        ValueError: neither `path` nor the module it imports `name` from binds a literal `name`.
    """
    for node in module_tree(src, path).body:
        value = _bound(node, name)
        if value is not None:
            return ast.literal_eval(value)
        if (
            isinstance(node, ast.ImportFrom)
            and node.level
            and any(a.name == name for a in node.names)
        ):
            base = PurePosixPath(path).parents[node.level - 1]
            imported = base.joinpath(*(node.module or "").split(".")).with_suffix(".py")
            return module_literal(src, str(imported), name)
    raise ValueError(f"no literal {name} in {path} or the module it imports it from")


def _pyproject_value(src: Sources, *keys: str) -> object:
    """The value at the table path `keys` of pyproject.toml; None where it sets none.

    Raises:
        UserError: pyproject.toml is not UTF-8 text or not valid TOML.
    """
    path = src.path(PYPROJECT)
    try:
        value: object = tomllib.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        raise UserError(
            f"{path} is not UTF-8 text: {exc.reason}", hint=f"save {path} with UTF-8 encoding"
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise UserError(
            f"{path} is not valid TOML: {exc}",
            hint=f"correct {PYPROJECT} at the line and column the message names",
        ) from exc
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def import_contract_count(src: Sources) -> int:
    """import-linter contracts in `pyproject.toml`.

    Raises:
        UserError: pyproject.toml sets no list `tool.importlinter.contracts`.
    """
    contracts = _pyproject_value(src, "tool", "importlinter", "contracts")
    if not isinstance(contracts, list):
        raise UserError(
            f"{PYPROJECT} sets no list tool.importlinter.contracts",
            hint="write the import-linter contracts as [[tool.importlinter.contracts]] tables",
        )
    return len(contracts)


def suite_level_test_files(src: Sources) -> list[str]:
    """Test files not named after a module: tests/a/test_b.py tests src/crossvenue_rag/a/b.py.

    The suite-level tests, which are named after no module. A test file under
    tests/scripts/ tests the script of its name: tests/scripts/test_b.py tests scripts/b.py.
    """
    tests = src.path("tests")
    return sorted(
        str(path.relative_to(src.root))
        for path in tests.rglob("test_*.py")
        if not _tested_module(src, path.relative_to(tests)).exists()
    )


def _tested_module(src: Sources, test: Path) -> Path:
    folder = test.parent
    base = src.path(".") if folder.parts[:1] == (SCRIPTS,) else src.path(PACKAGE)
    return base / folder / test.name.removeprefix("test_")


def fixture_pages(src: Sources) -> int:
    """Pages of the fixture corpus in tests/conftest.py (`CORPUS`).

    Raises:
        ValueError: `CORPUS` is not a dict literal.
    """
    corpus = module_literal(src, FIXTURE_CONFTEST, "CORPUS")
    if not isinstance(corpus, dict):
        raise ValueError(f"CORPUS in {FIXTURE_CONFTEST} is not a dict")
    return len(corpus)


def same_bytes(src: Sources, first: str | Path, second: str | Path) -> bool:
    """Whether the files at `first` and `second` have one sha256."""
    return file_sha256(src.path(first)) == file_sha256(src.path(second))


def python_requirement(src: Sources) -> str:
    """The lowest Python `requires-python` in pyproject.toml admits (`3.12`).

    `?` when it sets none, or one that is not a lower bound `>=x.y`.
    """
    required = _pyproject_value(src, "project", "requires-python")
    written = re.fullmatch(r">=([\d.]+)", str(required or ""))
    return written.group(1) if written else "?"


def direction_count(src: Sources) -> int:
    """Development directions in docs/explanation/directions.md: its `## ` headings."""
    text = src.path("docs/explanation/directions.md").read_text(encoding="utf-8")
    return len(re.findall(r"^## ", text, re.M))


def pytest_count(root: str, *, every: bool) -> int:
    """Tests `pytest` collects under `root`: those `make test` runs, or with `every` all of them.

    The sum of `pytest --collect-only -q` per file: the configured `-q` makes it
    print one "path: count" line per test file. `every` clears the configured
    marker filter (`-m ""`), so the slow tests are counted too.

    Raises:
        RuntimeError: `pytest` reports no collected test.
    """
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"]
    out = subprocess.run(
        [*command, "-m", ""] if every else command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    counts = [
        int(line.rpartition(":")[2]) for line in out.splitlines() if line.startswith("tests/")
    ]
    if not counts:
        raise RuntimeError("pytest reported no collected tests")
    return sum(counts)
