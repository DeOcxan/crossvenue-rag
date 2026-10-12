"""Reading a YAML file whose top level must be a mapping, such as a config or a list of sources.

`read_model` returns the file validated as a model. `read_mapping` returns the mapping as
YAML parsed it, which a config's `--set` overrides change before it is validated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from . import formatting
from .errors import UserError, invalid_file, require_file


def read_mapping(path: str | Path, *, hint: str) -> dict[str, object]:
    """The mapping at the top of the YAML file `path`; an empty file reads as `{}`.

    The values are whatever YAML parsed, for a model to validate.

    Raises:
        UserError: Nothing is at `path`, or a directory is. The file is not UTF-8 or not valid YAML,
            its top level is a list or a scalar, or a top-level key is not a name. `hint` says where
            to fix it, except for the encoding, whose hint is to save the file as UTF-8.
    """
    try:
        text = require_file(path, hint).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UserError(
            f"{path} is not UTF-8 text: {exc.reason}", hint=f"save {path} with UTF-8 encoding"
        ) from exc
    try:
        raw: object = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise UserError(f"{path} is not valid YAML: {yaml_problem(exc)}", hint=hint) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise UserError(
            f"{path} holds a {type(raw).__name__} at the top level where a mapping of keys "
            "is expected",
            hint=hint,
        )
    mapping: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise UserError(f"{path} has the top-level key {key!r}, which is not a name", hint=hint)
        mapping[key] = value
    return mapping


def yaml_problem(exc: yaml.YAMLError) -> str:
    """What a YAML parser error says, and the line it points at when it names one."""
    problem = getattr(exc, "problem", None) or str(exc).splitlines()[0]
    mark = getattr(exc, "problem_mark", None)
    return (
        f"{problem} (line {formatting.own_form(mark.line + 1)})"
        if mark is not None
        else str(problem)
    )


def read_model[M: BaseModel](path: str | Path, model: type[M], *, hint: str) -> M:
    """The YAML file `path` validated as `model`; an empty file reads as `{}`.

    Raises:
        UserError: `model` rejects the file; `hint` says where to fix it.
    """
    try:
        return model.model_validate(read_mapping(path, hint=hint))
    except ValidationError as exc:
        raise invalid_file(path, exc, hint=hint) from exc
