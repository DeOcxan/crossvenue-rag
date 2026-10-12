"""The expected failures every layer raises, each a message with a hint on what to do.

`UserError` is a problem the user can fix; `ExternalFailure` is a failure outside
the program, such as a model call. The CLI prints either as its message and hint
(`cli.errors.friendly_errors`), as it does Postgres refusing a connection; any other
exception keeps its traceback, because that one is a bug.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from . import formatting
from .listing import first_items


class ExpectedFailure(Exception):
    """A failure the program expects: a `UserError` or an `ExternalFailure`."""

    def __init__(self, message: str, hint: str = "") -> None:
        """A failure saying `message`, with `hint` on what to do ("" for none)."""
        super().__init__(message)
        self.hint = hint

    def with_hint(self) -> str:
        """The message and, in parentheses, the hint: the failure as one line of text."""
        return f"{self} ({self.hint})" if self.hint else str(self)


class UserError(ExpectedFailure):
    """A problem the user can fix, with a suggestion for how."""


class ExternalFailure(ExpectedFailure):
    """A failure outside the program, such as a failed model call; its hint says what to try."""


def validation_problems(exc: ValidationError) -> str:
    """The first problems of a failed validation as text: each field's path and what is wrong.

    The text holds no input value, so no raw representation of a list or dict.
    """
    return first_items(
        [
            f"{'.'.join(formatting.own_form(p) for p in error['loc'])}: {error['msg']}"
            if error["loc"]
            else error["msg"]
            for error in exc.errors()
        ],
        "; ",
    )


def invalid_file(path: str | Path, exc: ValidationError, hint: str) -> UserError:
    """A user-edited file that failed validation, naming the file and its first problems."""
    return UserError(f"invalid {path}: {validation_problems(exc)}", hint=hint)


def is_blank(text: str) -> bool:
    """Whether `text` is empty or only whitespace."""
    return not text.strip()


def not_blank(name: str, value: str) -> str:
    """`value`, after checking it holds more than whitespace.

    Raises:
        UserError: `value` is empty or only whitespace; the message names `name`.
    """
    if is_blank(value):
        raise UserError(f"{name} is empty", hint=f"pass {name} with some text")
    return value


def require_file(path: str | Path, hint: str) -> Path:
    """`path`, after checking it names an existing file.

    Raises:
        UserError: Nothing is at `path`, or a directory is; `hint` says where to fix it.
    """
    path = Path(path)
    if not path.exists():
        raise UserError(f"{path} not found", hint=hint)
    if not path.is_file():
        raise UserError(f"{path} is a directory where a file is expected", hint=hint)
    return path
