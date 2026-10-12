"""The command line as docs/reference/cli.md describes it, read from the source of `cli`.

`reporting` may not import `cli`, the layer above it, so each option's default and
least value are read from the syntax tree of the command's module.
"""

from __future__ import annotations

import ast

from ..sources import Sources
from .lookup import entry
from .memo import per_sources
from .repository import PACKAGE

# The package the commands are defined in, relative to the repository.
CLI_PACKAGE = f"{PACKAGE}/cli"


def _parse(src: Sources, module: str, package: str = CLI_PACKAGE) -> ast.Module:
    """The syntax tree of `module` (dotted, relative to `package`)."""
    path = src.path(package) / f"{module.replace('.', '/')}.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def _bound(node: ast.stmt, name: str) -> ast.expr | None:
    """The value `node` binds to `name` when it is a module-level `name = ...`; else None."""
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in node.targets
    ):
        return node.value
    return None


def _constant(src: Sources, module: str, name: str) -> object:
    """The literal a module-level `name = ...` of the cli `module` binds, or one it imports.

    A name imported with `from ..package.module import name` is looked up in that module.
    """
    tree = _parse(src, module)
    for node in tree.body:
        value = _bound(node, name)
        if value is not None:
            return ast.literal_eval(value)
        if isinstance(node, ast.ImportFrom) and any(a.name == name for a in node.names):
            # `from ..dataset.sampling import X` in cli/ names src/crossvenue_rag/dataset/.
            imported = _parse(src, node.module or "", PACKAGE if node.level > 1 else CLI_PACKAGE)
            for item in imported.body:
                value = _bound(item, name)
                if value is not None:
                    return ast.literal_eval(value)
    raise ValueError(f"no literal {name} in {module} or the module it imports it from")


@per_sources
def commands(src: Sources) -> dict[str, tuple[str, ast.FunctionDef]]:
    """Each command's module and function, by command name, as `cli/__init__.py` registers it.

    A registration is `app.command("name")(_module.function)`, or `app.command()` for a
    command named after its function with `_` as `-`.

    Raises:
        ValueError: a registration names a function its module does not define.
    """
    registry = _parse(src, "__init__")
    modules = {
        alias.asname: alias.name
        for node in registry.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname
    }
    found: dict[str, tuple[str, ast.FunctionDef]] = {}
    for node in ast.walk(registry):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Call)):
            continue
        register, target = node.func, node.args[0]
        if not (
            isinstance(register.func, ast.Attribute)
            and register.func.attr == "command"
            and isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
        ):
            continue
        name = (
            ast.literal_eval(register.args[0]) if register.args else target.attr.replace("_", "-")
        )
        module = entry(modules, target.value.id, "the imports of cli/__init__.py")
        function = next(
            (
                f
                for f in _parse(src, module).body
                if isinstance(f, ast.FunctionDef) and f.name == target.attr
            ),
            None,
        )
        if function is None:
            raise ValueError(f"cli/{module}.py defines no {target.attr}")
        found[name] = (module, function)
    return found


def _declares(annotation: ast.expr | None, option: str) -> bool:
    """Whether a parameter's annotation holds `typer.Option(option, ...)` or `typer.Argument`."""
    return annotation is not None and any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("Option", "Argument")
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == option
        for n in ast.walk(annotation)
    )


def option_default(src: Sources, command: str, option: str) -> object:
    """The default of `option` (`--rounds`) of `command`: a literal of its signature.

    A default that is a name is the literal the name binds (`_constant`).

    Raises:
        ValueError: no command `command` is registered, or it declares no `option` with a default.
    """
    module, function = entry(commands(src), command, "the registered commands")
    args = function.args.args
    defaults = dict(zip((a.arg for a in args[len(args) - len(function.args.defaults) :]),
                        function.args.defaults, strict=True))  # fmt: skip
    for arg in args:
        if _declares(arg.annotation, option) and arg.arg in defaults:
            default = defaults[arg.arg]
            if isinstance(default, ast.Name):
                return _constant(src, module, default.id)
            return ast.literal_eval(default)
    raise ValueError(f"{command} declares no {option} with a literal default")


def least_value(src: Sources, command: str, option: str) -> int:
    """The least value `command` admits for `option`: the `low` of its `at_least` call.

    Raises:
        ValueError: no command `command` is registered, or it makes no `at_least` call for
            `option` with an integer bound.
    """
    for node in ast.walk(entry(commands(src), command, "the registered commands")[1]):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "at_least"
            and ast.literal_eval(node.args[0]) == option
        ):
            low = ast.literal_eval(node.args[2])
            if isinstance(low, int):
                return low
    raise ValueError(f"{command} checks no lower bound of {option}")
