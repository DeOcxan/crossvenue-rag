"""Checks on a path or name from config, a command line or a download, made before it is used.

A name joined to a root stays inside the root (`confined`), a name that becomes part of a file
name is plain (`PlainName`), and a directory to be made has no file at or above it
(`file_on_path`, `DirectoryPath`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator

from .errors import UserError, is_blank


def confined(root: Path, part: str, *, what: str, hint: str) -> Path:
    """`root / part`, after checking that it resolves to a path strictly inside `root`.

    Symlinks are resolved on both sides, so a link under `root` that points
    elsewhere is refused too. The returned path is the plain join, not the
    resolved one, so a caller can still take it `relative_to(root)`.

    Args:
        root: The directory the path must stay under; it need not exist yet.
        part: The name or relative path to join: a venue, a URL path, a file name.
        what: What `part` is, for the error: "venue", "llms.txt link".
        hint: Where the user fixes `part`.

    Raises:
        UserError: `part` is empty or blank, holds a control character, is absolute, or
            leads to `root` itself or outside it (a `..` segment, a symlink); the error
            names `part` and `root`.
    """
    if is_blank(part):
        raise UserError(f"{what} {part!r} is empty, so it names no path under {root}", hint=hint)
    if _has_control_character(part):
        raise UserError(
            f"{what} {part!r} holds a control character, so it names no path under {root}",
            hint=hint,
        )
    if Path(part).is_absolute():
        raise UserError(f"{what} {part!r} is absolute, so it names no path under {root}", hint=hint)
    joined = root / part
    base = root.resolve()
    if base not in joined.resolve().parents:
        raise UserError(f"{what} {part!r} names no path inside {root}", hint=hint)
    return joined


def _has_control_character(text: str) -> bool:
    return any(ord(c) < ord(" ") or c == "\x7f" for c in text)


def plain_name_problem(name: str) -> str | None:
    """Why `name` cannot name one directory entry, or None when it can.

    A plain name is not blank, not `.` or `..`, and holds no path separator and no
    control character.
    """
    if is_blank(name) or name in {".", ".."}:
        return "is empty or a relative directory"
    if "/" in name or "\\" in name:
        return "holds a path separator"
    if _has_control_character(name):
        return "holds a control character"
    return None


def file_on_path(path: Path) -> Path | None:
    """The first of `path` and its parents that exists as a file, or None.

    Nothing can be created at or under such a path, so a directory that `path`
    names cannot be made.
    """
    for candidate in (path, *path.parents):
        if candidate.is_file():
            return candidate
    return None


def _directory_path(path: Path) -> Path:
    blocker = file_on_path(path)
    if blocker is not None:
        raise ValueError(f"{path} cannot be a directory: {blocker} is a file")
    return path


def _plain_name(name: str) -> str:
    problem = plain_name_problem(name)
    if problem is not None:
        raise ValueError(f"name {name!r} {problem}")
    return name


# A config path a directory is made at: refused when it loads, before any work or paid call,
# if a file stands at it or above it.
DirectoryPath = Annotated[Path, AfterValidator(_directory_path)]
# A config name that becomes part of a file name (`plain_name_problem`).
PlainName = Annotated[str, AfterValidator(_plain_name)]
