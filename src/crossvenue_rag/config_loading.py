"""Reading the pipeline configuration: the YAML file, `--set` overrides and `CROSSVENUE_PG_DSN`.

The result is validated as `config.Config`. An ablation is another config
(`crossvenue eval --config <another>.yaml`) or an override such as
`--set chunking.max_tokens=256`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from . import listing
from .config import Config
from .errors import UserError, invalid_file, require_file
from .yaml_files import read_mapping, yaml_problem

# The config every command and the service read unless told otherwise.
DEFAULT_CONFIG = Path("configs/default.yaml")

# How an override is written, for the hint of every override error.
_OVERRIDE_HINT = "write --set section.field=value, for example --set retrieval.k=20"


def _set_nested(data: dict[str, object], override: str) -> None:
    """Applies one `section.field=value` override in place.

    Raises:
        UserError: The override has no `=`, an empty key, a key under a value that is not
            a section, or a value that is not valid YAML.
    """
    if "=" not in override:
        raise UserError(
            f"override must look like section.field=value, got {override!r}", hint=_OVERRIDE_HINT
        )
    dotted, _, raw_value = override.partition("=")
    keys = dotted.strip().split(".")
    if not all(keys):
        raise UserError(f"override {override!r} has an empty key", hint=_OVERRIDE_HINT)
    try:
        # The value is parsed as YAML, so `--set retrieval.k=20` gives an int and
        # `--set eval.methods=[bm25]` a list.
        value = yaml.safe_load(raw_value.strip())
    except yaml.YAMLError as exc:
        raise UserError(
            f"override {override!r}: the value is not valid YAML: {yaml_problem(exc)}",
            hint="quote a value that holds [, {, @ or :, for example --set 'eval.methods=[bm25]'",
        ) from exc
    cursor: dict[str, object] = data
    for depth, key in enumerate(keys[:-1]):
        child = cursor.get(key)
        if child is None:
            section: dict[str, object] = {}
            cursor[key] = section
            cursor = section
        elif isinstance(child, dict):
            cursor = child
        else:
            raise UserError(
                f"override {override!r}: {'.'.join(keys[: depth + 1])} holds a value, so it has "
                "no keys to set",
                hint=_OVERRIDE_HINT,
            )
    cursor[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """The validated config of a YAML file with `field.path=value` overrides applied.

    `CROSSVENUE_PG_DSN` in the environment overrides `store.dsn`, so the DSN can carry
    a password that no config file holds; `Config.to_yaml` leaves the DSN out of the
    config a run saves. No other environment variable is read.

    A `store` section that is not a mapping is left for the schema to refuse.

    Raises:
        UserError: Nothing is at `path`, or a directory is. An override of `overrides` has no `=`,
            an empty key, a key under a value that is not a section, or a value that is not valid
            YAML. Or the schema rejects a value: the error names the file, or every override that
            supplied a rejected value (`_overrides_at_fault`).
    """
    require_file(path, hint="pass --config with an existing config file")
    raw = read_mapping(
        path, hint="write the config as sections of keys, as configs/default.yaml does"
    )
    from_file = copy.deepcopy(raw)
    for override in overrides or []:
        _set_nested(raw, override)
    for data in (raw, from_file):
        _apply_env_dsn(data)

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        culprits = _overrides_at_fault(overrides or [], exc, from_file)
        if culprits:
            keys = listing.prose([f"--set {_override_key(c)}" for c in culprits])
            raise invalid_file(
                ", ".join(f"--set {c}" for c in culprits),
                exc,
                hint=f"change {keys} to a value the error allows, or drop "
                + ("it" if len(culprits) == 1 else "them"),
            ) from exc
        raise invalid_file(
            path,
            exc,
            hint=f"change each key the error names in {path} to a value it allows "
            "(docs/reference/config.md)",
        ) from exc


def _apply_env_dsn(raw: dict[str, object]) -> None:
    """Puts `CROSSVENUE_PG_DSN`, when set, into `store.dsn` of `raw`."""
    env_dsn = os.environ.get("CROSSVENUE_PG_DSN")
    if env_dsn:
        store = raw.get("store")
        if store is None:
            raw["store"] = {"dsn": env_dsn}
        elif isinstance(store, dict):
            store["dsn"] = env_dsn


def _override_key(override: str) -> str:
    """The dotted key of a `section.field=value` override."""
    return override.partition("=")[0].strip()


def _overrides_at_fault(
    overrides: list[str], exc: ValidationError, from_file: dict[str, object]
) -> list[str]:
    """Every override that set a value the schema rejects; empty when the file is at fault.

    An override is at fault when a rejected location lies at or under its key, or when a
    check of a whole section names its key, as `generation.judge 'x' is not one of
    generation.judges` does. For the same key the last override counts, because a later
    `--set` replaces the earlier value. When the file alone validates, every override
    under a rejected section that no such override explains is at fault too (a section
    check such as `hnsw_ef_construction must be at least 2 times hnsw_m`), and all of them
    when none is found.
    """
    rejected = [(tuple(str(part) for part in error["loc"]), error["msg"]) for error in exc.errors()]
    last = {_override_key(o): o for o in overrides}

    def explains(dotted: str, loc: tuple[str, ...], msg: str) -> bool:
        keys = tuple(dotted.split("."))
        return loc[: len(keys)] == keys or dotted in msg

    at_fault = [
        override
        for dotted, override in last.items()
        if any(explains(dotted, loc, msg) for loc, msg in rejected)
    ]
    if not overrides:
        return []
    try:
        Config.model_validate(from_file)
    except ValidationError:
        return at_fault
    unexplained = [
        loc
        for loc, msg in rejected
        if not any(explains(_override_key(o), loc, msg) for o in at_fault)
    ]
    under = [
        override
        for dotted, override in last.items()
        if override not in at_fault
        and any(tuple(dotted.split("."))[: len(loc)] == loc for loc in unexplained)
    ]
    return [*at_fault, *under] or list(last.values())
