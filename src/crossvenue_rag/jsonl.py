"""JSON Lines I/O: one JSON object per line, UTF-8."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from . import formatting
from .errors import UserError, invalid_file, require_file

# What a line that is not a JSON object is told to do.
_LINE_HINT = "fix that line of the file: each line holds one JSON object"


def read(path: str | Path) -> list[dict[str, Any]]:
    """Every record of a JSON Lines file, in file order; blank lines are skipped.

    Raises:
        UserError: The file is missing or a directory, or a line is not UTF-8, not JSON
            or holds something other than an object (`iter_rows`); the error names the
            file and the line.
    """
    return [row for _, row in iter_rows(path)]


def read_models[M: BaseModel](path: str | Path, model: type[M], hint: str = _LINE_HINT) -> list[M]:
    """Every record of a JSON Lines file validated as `model`, in file order."""
    return [record for _, record in iter_models(path, model, hint)]


def iter_models[M: BaseModel](
    path: str | Path, model: type[M], hint: str = _LINE_HINT
) -> Iterator[tuple[int, M]]:
    """`(line number, record)` for every non-blank line, validated as `model`, read lazily.

    `hint` is what the error for a line that is not a valid `model` tells the reader to do.

    Raises:
        UserError: A line is not a valid `model`; the error names the file, the line and the
            field.
    """
    for number, row in iter_rows(path):
        try:
            record = model.model_validate(row)
        except ValidationError as exc:
            raise invalid_file(formatting.location(path, number), exc, hint=hint) from exc
        yield number, record


def iter_rows(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """`(line number, record)` for every non-blank line, read lazily.

    Line numbers start at 1 and count blank lines, so an error can name the line
    of the file. The file is read as bytes and decoded line by line, so a byte that is
    not UTF-8 is reported on its own line.

    Raises:
        UserError: Nothing is at `path`, or a directory is. A line is not UTF-8.
    """
    source = require_file(path, hint="check the path of the file")
    with source.open("rb") as fh:
        for number, raw in enumerate(fh, start=1):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UserError(
                    f"{formatting.location(path, number)}: not UTF-8 text ({exc.reason})",
                    hint=_LINE_HINT,
                ) from exc
            if not line.strip():
                continue
            yield number, json_object(line, path, number)


def json_object(line: str, path: str | Path, number: int) -> dict[str, Any]:
    """The JSON object that `line`, line `number` of `path`, holds.

    Raises:
        UserError: The line is not JSON, or holds something other than an object; the error
            names the file and the line.
    """
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise UserError(
            f"{formatting.location(path, number)}: not JSON: {exc.msg} "
            f"(column {formatting.own_form(exc.colno)})",
            hint=_LINE_HINT,
        ) from exc
    if not isinstance(row, dict):
        raise UserError(
            f"{formatting.location(path, number)}: a JSON {type(row).__name__} where an object "
            "is expected",
            hint=_LINE_HINT,
        )
    return row


def dumps(rows: Iterable[Mapping[str, Any]], *, ascii_only: bool = False) -> str:
    """Records as JSON Lines text, one line per record, each ending in a newline.

    Args:
        rows: The records.
        ascii_only: Whether non-ASCII characters are escaped (`json.dumps` with
            `ensure_ascii`), for files whose committed bytes were written that way.
    """
    return "".join(json.dumps(row, ensure_ascii=ascii_only) + "\n" for row in rows)


def write(path: str | Path, rows: Iterable[Mapping[str, Any]], *, ascii_only: bool = False) -> None:
    """Writes records to a JSON Lines file, replacing it."""
    Path(path).write_text(dumps(rows, ascii_only=ascii_only), encoding="utf-8")


def append(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Appends records to a JSON Lines file, creating it if it does not exist."""
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(dumps(rows))
