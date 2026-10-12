"""The configuration as docs/reference/config.md describes it.

A key's domain and its value keep the form of a config value (docs/style.md rule 11).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, get_args, get_origin

import annotated_types
from pydantic import BaseModel

from ... import formatting
from ...config import Config
from ...config_generation import SHA256_PATTERN, CalibrationDimension
from ...config_loading import DEFAULT_CONFIG
from ...errors import UserError
from ...yaml_files import read_mapping
from ..sources import Sources
from .distinct import shared_value
from .lookup import entry
from .memo import per_sources

# How the reference writes each lower bound a pydantic `Field` sets, after the type; `>=`
# is typeset as U+2265 where it is compared (`patterns.typeset`).
LOWER_BOUNDS = {annotated_types.Gt: ("gt", ">"), annotated_types.Ge: ("ge", ">=")}
# The keys whose number is a threshold, written with `formatting.CONFIG_THRESHOLD_DECIMALS`
# as configs/default.yaml writes it; generation.calibration_thresholds is a map, no number.
THRESHOLD_KEYS = frozenset({("eval", "duplicate_threshold")})


def section_model(section: str) -> type[BaseModel]:
    """The schema class of the config section `section` (`chunking` gives `ChunkingConfig`).

    Raises:
        ValueError: `Config` has no section `section`, or it is not a model.
    """
    model = entry(Config.model_fields, section, "the Config schema").annotation
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise ValueError(f"config section {section!r} is not a model")
    return model


def _bound(value: object) -> str:
    """A bound as the reference writes it: `0` for 0.0, `1000` for 1000."""
    if not isinstance(value, int | float):
        raise ValueError(f"a bound {value!r} that is not a number")
    return formatting.own_form(int(value) if float(value).is_integer() else value)


def field_domain(section: str, key: str) -> str | None:
    """The domain of a number key, as the reference's Type column writes it.

    `int > 0`, `finite float >= 0`, `int in [2, 100]`, `1 or 2`, `str, at most 49
    characters`; None for a key whose domain is not a number or a length (a path, a
    list, a Literal of names).

    Raises:
        ValueError: `Config` has no model section `section`, or the section has no `key`.
    """
    info = entry(section_model(section).model_fields, key, f"config section {section}")
    annotation = info.annotation
    if get_origin(annotation) is Literal:
        args = get_args(annotation)
        numbers = all(isinstance(a, int) and not isinstance(a, bool) for a in args)
        return " or ".join(formatting.own_form(a) for a in args) if numbers else None
    metadata = info.metadata
    if annotation is str:
        lengths = [m.max_length for m in metadata if isinstance(m, annotated_types.MaxLen)]
        return f"str, at most {formatting.own_form(lengths[0])} characters" if lengths else None
    if annotation not in (int, float):
        return None
    finite = any(getattr(m, "allow_inf_nan", True) is False for m in metadata)
    kind = ("finite float" if finite else "float") if annotation is float else "int"
    lows = [
        (sign, getattr(m, attr))
        for m in metadata
        for cls, (attr, sign) in LOWER_BOUNDS.items()
        if isinstance(m, cls)
    ]
    highs = [m.le for m in metadata if isinstance(m, annotated_types.Le)]
    if lows and highs:
        return f"{kind} in [{_bound(lows[0][1])}, {_bound(highs[0])}]"
    if lows:
        return f"{kind} {lows[0][0]} {_bound(lows[0][1])}"
    return kind


def sha256_digits() -> int:
    """The hex digits a prompt sha256 pin has (`config_generation.SHA256_PATTERN`).

    Raises:
        ValueError: the pattern states no `{digits}` count.
    """
    found = re.search(r"\{(\d+)\}", SHA256_PATTERN)
    if found is None:
        raise ValueError(f"no digit count in {SHA256_PATTERN!r}")
    return int(found.group(1))


@per_sources
def default_yaml(src: Sources) -> dict[str, dict[str, object]]:
    """configs/default.yaml as the YAML parser reads it (`read_mapping`): each section a mapping.

    Raises:
        UserError: a section is not a mapping of keys.
    """
    hint = "write the config as sections of keys"
    data = read_mapping(src.path(DEFAULT_CONFIG), hint=hint)
    sections: dict[str, dict[str, object]] = {}
    for name, keys in data.items():
        if not isinstance(keys, dict):
            raise UserError(f"section {name} of {DEFAULT_CONFIG} is not a mapping", hint=hint)
        sections[str(name)] = {str(k): v for k, v in keys.items()}
    return sections


def yaml_value(src: Sources, section: str, key: str) -> object:
    """The value configs/default.yaml sets for `section.key`; None when it sets none."""
    return default_yaml(src).get(section, {}).get(key)


def number_value(src: Sources, section: str, key: str) -> str | None:
    """`yaml_value` in its own form when it is a number, else None (a boolean is none).

    A threshold (`THRESHOLD_KEYS`) has `formatting.CONFIG_THRESHOLD_DECIMALS` decimals.
    """
    value = yaml_value(src, section, key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if (section, key) in THRESHOLD_KEYS:
        return formatting.own_form(value, formatting.CONFIG_THRESHOLD_DECIMALS)
    return formatting.own_form(value)


def shared_threshold(src: Sources, dimensions: Sequence[CalibrationDimension]) -> float | None:
    """The calibration threshold configs/default.yaml sets for all `dimensions`; None if several."""
    thresholds = src.default_config.generation.calibration_thresholds
    return shared_value(entry(thresholds, d, "calibration_thresholds") for d in dimensions)


def schema_default_length(section: str, key: str) -> int:
    """The items of the schema default of a list key: the methods `eval` runs by default.

    Raises:
        ValueError: `Config` has no model section `section`, or the section has no `key`.
    """
    default = entry(section_model(section).model_fields, key, f"config section {section}").default
    return len(default)


def yaml_length(src: Sources, section: str, key: str) -> int:
    """The items configs/default.yaml lists for `section.key`.

    Raises:
        ValueError: configs/default.yaml sets no list or mapping for `section.key`.
    """
    value = yaml_value(src, section, key)
    if not isinstance(value, list | dict):
        raise ValueError(f"{section}.{key} is not a list in {DEFAULT_CONFIG}")
    return len(value)
