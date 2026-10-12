"""The base every config section shares, which refuses what a typo or a YAML boolean gives.

The sections themselves are in `config_pipeline`, `config_eval` and `config_generation`;
`config.Config` holds them all.
"""

from __future__ import annotations

from typing import get_args

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from . import formatting
from .errors import is_blank


def distinct[T](values: list[T], what: str) -> list[T]:
    """`values`, after checking none repeats.

    Raises:
        ValueError: a value repeats; the message names `what` and the repeated values.
    """
    repeated = sorted({formatting.own_form(v) for v in values if values.count(v) > 1})
    if repeated:
        raise ValueError(f"{what} repeats {', '.join(repeated)}")
    return values


def named(value: str, key: str) -> str:
    """`value`, after checking it holds more than whitespace.

    Raises:
        ValueError: `value` is empty or blank; the message names `key`.
    """
    # pydantic turns only a ValueError into a validation error, so this raises one and shares
    # the test of `errors.not_blank`.
    if is_blank(value):
        raise ValueError(f"{key} must not be empty or blank")
    return value


def _numeric(annotation: object) -> bool:
    """Whether a field type holds an int or a float anywhere, a Literal of numbers included."""
    if annotation in (int, float):
        return True
    return any(
        _numeric(arg) or (isinstance(arg, int | float) and not isinstance(arg, bool))
        for arg in get_args(annotation)
    )


def _holds_boolean(value: object) -> bool:
    """Whether `value`, or any item or mapping value inside it, is a boolean."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list | tuple):
        return any(_holds_boolean(v) for v in value)
    return isinstance(value, bool)


class Section(BaseModel):
    """A config section that refuses an unknown key, so a typo cannot fall back to a default.

    A number field also refuses a boolean: YAML reads `yes`, `on` and `true` as True,
    which pydantic would otherwise take as 1.
    """

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _no_boolean_number(cls, value: object, info: ValidationInfo) -> object:
        name = info.field_name
        numeric = name is not None and _numeric(cls.model_fields[name].annotation)
        if numeric and _holds_boolean(value):
            raise ValueError(
                "true or false where a number is required (YAML reads yes, no, on and "
                "off as true or false)"
            )
        return value
