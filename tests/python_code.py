"""The checks of the code rules that read a Python file's syntax tree.

`tests/test_code_hygiene.py` runs each check over the files it names.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import tokenize
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

from tests.published import ROOT


def modules_without_tests(root: Path) -> list[str]:
    """Modules with no test file of their name: src/crossvenue_rag/a/b.py has tests/a/test_b.py.

    Package `__init__.py` files are not modules of their own and are left out.
    """
    tests, package = root / "tests", root / "src/crossvenue_rag"
    return sorted(
        str(path.relative_to(root))
        for path in package.rglob("*.py")
        if path.name != "__init__.py"
        and not (tests / path.parent.relative_to(package) / f"test_{path.name}").exists()
    )


# The trees whose Python the code rules govern (docs/code-style.md, first line).
PYTHON_TREES = ("src", "tests", "scripts")


def python_files(trees: tuple[str, ...] = PYTHON_TREES) -> Iterator[tuple[str, str]]:
    """(path relative to the root, source) of every `.py` file under `trees`."""
    for tree in trees:
        for path in sorted((ROOT / tree).rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")


def exempt(path: str, function: str | None, table: tuple[tuple[str, str | None], ...]) -> bool:
    """Whether `table` exempts `function` of `path`, or the whole file or directory."""
    return any(
        (path == where or path.startswith(f"{where}/")) and name in (None, function)
        for where, name in table
    )


def _top_level_functions(tree: ast.Module) -> list[tuple[int, int, str]]:
    return [
        (node.lineno, node.end_lineno or node.lineno, node.name)
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]


def _function_at(spans: list[tuple[int, int, str]], line: int) -> str | None:
    return next((name for start, end, name in spans if start <= line <= end), None)


_BROAD = ("Exception", "BaseException")


def broad_handlers(source: str) -> list[int]:
    """Lines of every bare `except` and every `except` naming `Exception` or `BaseException`.

    The name counts alone, in a tuple or as an attribute (`builtins.Exception`).
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
        if caught is None or any(
            (isinstance(n, ast.Name) and n.id in _BROAD)
            or (isinstance(n, ast.Attribute) and n.attr in _BROAD)
            for n in names
        ):
            lines.append(node.lineno)
    return lines


_RICH_PRINTS = {"print", "log", "rule", "status", "input", "add_task"}
# Calls that read markup only on these receivers: a prompt's question, a live display's text.
_RICH_RECEIVERS = {
    "ask": {"Prompt", "Confirm", "IntPrompt", "FloatPrompt"},
    "update": {"status", "progress", "live"},
}
# `math.log(x)` is a logarithm; rich's `Console.log` is another function of that name.
_NOT_A_CONSOLE = {"math", "np", "numpy"}


def _literal(node: ast.expr) -> bool:
    """A str constant, or a `Text` built directly (anything but `Text.from_markup`)."""
    if isinstance(node, ast.Constant):
        return True
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Text") or (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "Text"
        and func.attr != "from_markup"
    )


def _sink_arguments(call: ast.Call) -> list[ast.expr]:
    """The arguments of `call` that rich would read as markup if they were a str."""
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    keywords = {k.arg: k.value for k in call.keywords}
    if isinstance(func, ast.Attribute) and name in _RICH_PRINTS:
        receiver = func.value
        if isinstance(receiver, ast.Name) and receiver.id in _NOT_A_CONSOLE:
            return []
        return list(call.args)
    if isinstance(func, ast.Attribute) and name in _RICH_RECEIVERS:
        receiver = func.value
        if isinstance(receiver, ast.Name) and receiver.id in _RICH_RECEIVERS[name]:
            return list(call.args)
        return []
    if isinstance(func, ast.Attribute) and name == "add_row":
        return list(call.args)
    if isinstance(func, ast.Attribute) and name == "add_column":
        return [v for k, v in keywords.items() if k == "footer"]
    if name == "from_markup":
        return list(call.args)
    if name in ("Table", "Panel"):
        titles = [v for k, v in keywords.items() if k in ("title", "caption", "subtitle")]
        return titles + (list(call.args[:1]) if name == "Panel" else [])
    return []


def rich_sinks(source: str) -> list[tuple[int, str | None]]:
    """(line, enclosing function) of every call that passes rich a value to read as markup.

    A call is a rich print, log, rule, status, input or `add_task`, a `Prompt.ask` (and
    its kin), an `update` of a status, progress or live display, a Table or Panel title,
    an `add_row` cell or a column footer. A value is anything but a str constant or a
    `Text`: an f-string, a variable, a call or a concatenation. A bare builtin `print` is
    ruff's T20 to refuse; rich's `print`, imported by name, is read by neither.
    """
    found: list[tuple[int, str | None]] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name if function is None else function
        if isinstance(node, ast.Call) and not all(map(_literal, _sink_arguments(node))):
            found.append((node.lineno, function))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(ast.parse(source), None)
    return found


# Style rule 29, which code rule 24 applies to comments and docstrings.
MAX_SENTENCE_WORDS = 35
_CODE_SPAN = re.compile(r"`[^`\n]*`")
# A sentence ends at . ! or ? before a space, but not after an abbreviation.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)\s+(?=[A-Za-z\x00\"'(\[])")
_PRAGMA = re.compile(r"^(fmt|type|noqa|pragma)\b|^noqa")
_DOCUMENTED = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


_SECTION = re.compile(r"^(Args|Returns|Raises|Yields|Attributes|Note):$")
# A row that opens a unit of its own: a list item, or an entry of a Google-style section.
_UNIT_START = re.compile(r"^\s*[-*] |^ {4}[\w`.*]+(?: \([^)]*\))?: ")


def _docstring_units(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """(line, text) of each paragraph, list item and section entry of every docstring.

    Attribute docstrings count too: a string statement right after an assignment in a
    module or class body. A paragraph whose every row is indented is a code block,
    which is skipped.
    """
    for node in ast.walk(tree):
        if isinstance(node, _DOCUMENTED):
            doc = ast.get_docstring(node)
            if doc:
                yield from _units(doc, node.body[0].lineno)
        if isinstance(node, ast.Module | ast.ClassDef):
            for before, after in pairwise(node.body):
                if (
                    isinstance(before, ast.Assign | ast.AnnAssign)
                    and isinstance(after, ast.Expr)
                    and isinstance(after.value, ast.Constant)
                    and isinstance(after.value.value, str)
                ):
                    yield from _units(inspect.cleandoc(after.value.value), after.lineno)


def _units(doc: str, line: int) -> Iterator[tuple[int, str]]:
    """(line, text) of each paragraph, list item and section entry of one docstring."""
    for paragraph in doc.split("\n\n"):
        rows = paragraph.splitlines()
        if rows and not all(r.startswith(" ") and not _UNIT_START.match(r) for r in rows):
            unit: list[str] = []
            for row in rows:
                if (_SECTION.match(row) or _UNIT_START.match(row)) and unit:
                    yield line, " ".join(unit)
                    unit = []
                if not _SECTION.match(row):
                    unit.append(row.strip())
            if unit:
                yield line, " ".join(unit)
        line += len(rows) + 1


def _comment_blocks(source: str) -> Iterator[tuple[int, str]]:
    """(line, text) of each run of comment lines, a trailing comment its own run."""
    block: list[str] = []
    start = last = 0
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        text = token.string.lstrip("#").strip()
        own_line = token.line.strip().startswith("#")
        if block and (not own_line or token.start[0] != last + 1):
            yield start, " ".join(block)
            block = []
        if _PRAGMA.match(text) or not text:
            last = token.start[0]
            continue
        if not block:
            start = token.start[0]
        block.append(text)
        last = token.start[0]
        if not own_line:
            yield start, " ".join(block)
            block = []
    if block:
        yield start, " ".join(block)


def sentences(text: str) -> list[str]:
    """The sentences of a paragraph, each code span as one `\\x00`."""
    masked = _CODE_SPAN.sub("\x00", text)
    return [s for s in _SENTENCE_END.split(masked) if s.strip()]


def words(sentence: str) -> int:
    """The words of a sentence outside its code spans: values in code font make a list."""
    return sum(1 for w in sentence.split() if w.strip("\x00.,;:()[]") != "")


def long_sentences(source: str) -> list[tuple[int, str | None, str]]:
    """(line, enclosing top-level function, first words) of every sentence over the bound.

    The sentences are those of every docstring and comment. A sentence is counted
    in words outside its code spans, so a list of values in code font is exempt.
    """
    tree = ast.parse(source)
    spans = _top_level_functions(tree)
    found = []
    for line, text in [*_docstring_units(tree), *_comment_blocks(source)]:
        for sentence in sentences(text):
            if words(sentence) > MAX_SENTENCE_WORDS:
                opening = " ".join(sentence.replace("\x00", "`...`").split()[:6])
                found.append((line, _function_at(spans, line), opening))
    return found


def _is_record(node: ast.expr) -> bool:
    """A `tuple[A, B, C, ...]` of three or more fixed fields; `tuple[X, ...]` is a sequence."""
    if not (isinstance(node, ast.Subscript) and ast.unparse(node.value) == "tuple"):
        return False
    fields = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    variadic = len(fields) == 2 and ast.unparse(fields[1]) == "..."
    return len(fields) >= 3 and not variadic


def _records_in(annotation: ast.expr) -> list[str]:
    """The record tuples in an annotation; a mapping's key tuple is an index and is left out."""
    keys = {
        id(n.slice.elts[0])
        for n in ast.walk(annotation)
        if isinstance(n, ast.Subscript)
        and ast.unparse(n.value) in ("dict", "Mapping", "Counter", "defaultdict")
        and isinstance(n.slice, ast.Tuple)
    }
    return [
        ast.unparse(n)
        for n in ast.walk(annotation)
        if isinstance(n, ast.expr) and id(n) not in keys and _is_record(n)
    ]


def record_tuples(source: str) -> list[tuple[int, str | None, str]]:
    """(line, enclosing top-level function, the tuple) of every record passed as a tuple.

    Code rule 6: a record passed between functions has named fields. The annotations
    read are those of parameters, returns, variables and type aliases.
    """
    tree = ast.parse(source)
    spans = _top_level_functions(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        annotations: list[ast.expr] = []
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            annotations += [a.annotation for a in arguments if a.annotation is not None]
            annotations += [node.returns] if node.returns is not None else []
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, ast.Assign | ast.TypeAlias) and isinstance(node.value, ast.Subscript):
            annotations.append(node.value)
        for annotation in annotations:
            for record in _records_in(annotation):
                found.append((node.lineno, _function_at(spans, node.lineno), record))
    return found


def _first_sentence(docstring: str) -> str:
    return sentences(docstring.split("\n\n", 1)[0])[0].replace("\x00", "`...`")


def joined_responsibilities(source: str) -> str | None:
    """The first sentence of the module docstring when it names two things it does.

    Code rule 2: the responsibility, the sentence before its first colon, needs no
    "and". A list of the parts of one responsibility follows the colon. A sentence that
    opens with a label of one word or path (`docs/x.md: what it claims`) states its
    responsibility after that label.
    """
    doc = ast.get_docstring(ast.parse(source))
    if not doc:
        return None
    sentence = _first_sentence(doc)
    label, _, rest = sentence.partition(": ")
    responsibility = rest if rest and " " not in label else sentence
    return sentence if re.search(r"\band\b", responsibility.split(": ", 1)[0]) else None


def defined_functions(source: str) -> set[str]:
    """The names of every function a module defines, methods included."""
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }


def _module_name(path: str) -> str:
    """The dotted name `path` is imported as; a script imports its sibling by file name."""
    parts = Path(path).with_suffix("").parts
    parts = parts[1:] if parts[0] in ("src", "scripts") else parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _own_test(path: str) -> str:
    """The test file code rule 27 names for a module of the package; '' for any other file."""
    prefix = "src/crossvenue_rag/"
    if not path.startswith(prefix):
        return ""
    module = Path(path.removeprefix(prefix))
    return str(Path("tests") / module.parent / f"test_{module.name}")


def _imports(path: str, tree: ast.Module) -> tuple[set[str], set[tuple[str, str]]]:
    """The modules `path` imports whole, and the (module, name) pairs it imports by name."""
    package = _module_name(path).split(".")
    if not path.endswith("__init__.py"):
        package = package[:-1]
    modules: set[str] = set()
    names: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            for alias in node.names:
                names.add((module, alias.name))
                modules.add(f"{module}.{alias.name}")
    return modules, names


def _uses(tree: ast.Module) -> list[tuple[str, int, bool]]:
    """(identifier, line, bare) of every name, attribute and word of a string but a docstring.

    `bare` is True for a plain name, which in another module is a local of that module.
    """
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, _DOCUMENTED) and node.body and isinstance(node.body[0], ast.Expr)
    }
    found = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Name):
            found.append((node.id, node.lineno, True))
        elif isinstance(node, ast.Attribute):
            found.append((node.attr, node.lineno, False))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found += [(w, node.lineno, False) for w in re.findall(r"\w+", node.value)]
    return found


def _definitions(tree: ast.Module) -> Iterator[tuple[str, ast.stmt]]:
    """(name, node) of every top-level function, class and constant, and every method."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            yield node.name, node
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            yield from ((t.id, node) for t in targets if isinstance(t, ast.Name))
        if isinstance(node, ast.ClassDef) and "Protocol" not in map(ast.unparse, node.bases):
            for method in node.body:
                if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                    yield f"{node.name}.{method.name}", method


def _registered(node: ast.stmt) -> bool:
    """A dunder, or a function a framework calls: a validator, a serializer, a command."""
    name = getattr(node, "name", "") or next(
        (t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)), ""
    )
    decorators = [ast.unparse(d) for d in getattr(node, "decorator_list", [])]
    hooks = ("validator", "serializer", "command", "fixture")
    return name.startswith("__") or any(h in d for d in decorators for h in hooks)


def unreferenced(files: dict[str, str]) -> list[str]:
    """`path:line name` of every definition only its own definition and its own tests use.

    A top-level name counts as used where its module uses it outside the definition
    or another file imports it. It counts too where a file that imports its module reads
    it as an attribute or names it in a string. A method counts as used wherever its
    name appears outside its definition and its module's own test file.
    """
    trees = {path: ast.parse(source) for path, source in files.items()}
    uses = {path: _uses(tree) for path, tree in trees.items()}
    imports = {path: _imports(path, tree) for path, tree in trees.items()}
    found = []
    for path, tree in trees.items():
        if path.startswith("tests/"):
            continue
        module, own_test = _module_name(path), _own_test(path)
        for name, node in _definitions(tree):
            if _registered(node):
                continue
            short, start, end = name.split(".")[-1], node.lineno, node.end_lineno or node.lineno
            used = any(w == short and not start <= line <= end for w, line, _ in uses[path])
            for other in (p for p in trees if p not in (path, own_test)):
                modules, names = imports[other]
                method_reads = any(w == short for w, _, _ in uses[other])
                attribute_reads = any(w == short and not bare for w, _, bare in uses[other])
                used |= (
                    (module, short) in names
                    or ("." in name and method_reads)
                    or (module in modules and attribute_reads)
                )
            if not used:
                found.append(f"{path}:{start} {name}")
    return found
