"""The source registry: data/sources.yaml read into validated specs.

The file also carries the structural metadata rules that read `environment`,
`api_version`, `product` and `doc_type` off a document's path and headings;
they are data because they describe each vendor's documentation layout, which
changes with the vendor.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from .. import formatting, listing
from ..confined_paths import confined, plain_name_problem
from ..errors import UserError
from ..models import METADATA_FIELDS, MetadataField, VenueType
from ..yaml_files import read_model

# Closed vocabularies. `api_version` is open: every vendor numbers differently.
ALLOWED_VALUES: dict[str, set[str]] = {
    "environment": {"mainnet", "testnet"},
    "product": {"spot", "perp"},
    "doc_type": {"reference", "guide", "faq", "changelog"},
}
if not set(ALLOWED_VALUES) <= set(METADATA_FIELDS):
    raise RuntimeError(
        f"ALLOWED_VALUES names fields outside METADATA_FIELDS: "
        f"{listing.prose(sorted(set(ALLOWED_VALUES) - set(METADATA_FIELDS)))}"
    )

# A group reference in a metadata rule's value: `{1}` is the first group of its `path` pattern.
GROUP_REF_RE = re.compile(r"\{(\d+)\}")


class MetadataRule(BaseModel):
    """One metadata rule: the `value` a field takes where the rule matches.

    `value` applies when `path` matches the doc path and/or `heading` matches
    any one heading of the chunk's breadcrumb (both regexes, `re.search`).
    `value` may reference capture groups of `path` as `{1}`; `value: null` is a
    stop rule: it matches and assigns nothing, so later rules cannot fire.
    """

    model_config = {"extra": "forbid"}

    value: str | None
    path: str | None = None
    heading: str | None = None

    @model_validator(mode="after")
    def _well_formed(self) -> MetadataRule:
        if self.path is None and self.heading is None:
            raise ValueError("a metadata rule needs `path`, `heading` or both")
        groups = 0
        for pattern in (self.path, self.heading):
            if pattern is not None:
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"bad regex {pattern!r}: {exc}") from exc
                if pattern is self.path:
                    groups = compiled.groups
        for ref in GROUP_REF_RE.findall(self.value or ""):
            if int(ref) > groups:
                raise ValueError(f"value {self.value!r} references group {ref} of `path`")
        return self


class SourceSpec(BaseModel):
    """One venue's documentation source; an unknown key is refused, as in `MetadataRule`."""

    model_config = {"extra": "forbid"}

    venue: str
    venue_type: VenueType
    kind: Literal["git", "gitbook"]
    license: str
    homepage: str = ""
    notes: str = ""

    # Git sources only.
    repo: str | None = None
    ref: str = "main"
    sparse_paths: list[str] = []
    include: list[str] = []
    exclude: list[str] = []

    # GitBook sources only.
    llms_index: str | None = None
    llms_section: str | None = None

    # field -> ordered rules; the first matching rule wins.
    metadata: dict[MetadataField, list[MetadataRule]] = {}
    # How questions name this venue (regexes, whole-word, case-insensitive).
    aliases: list[str] = []

    @field_validator("venue")
    @classmethod
    def _venue_is_a_plain_name(cls, v: str) -> str:
        # The venue names the source's directory under paths.raw, which a fetch empties.
        problem = plain_name_problem(v)
        if problem is not None:
            raise ValueError(f"venue {v!r} is not a plain directory name: it {problem}")
        return v

    @field_validator("aliases")
    @classmethod
    def _aliases_compile(cls, v: list[str]) -> list[str]:
        for alias in v:
            try:
                compiled = re.compile(alias)
            except re.error as exc:
                raise ValueError(f"bad alias regex {alias!r}: {exc}") from exc
            # An alias that matches nothing at all would find the venue in every question.
            if compiled.search("") is not None:
                raise ValueError(f"alias regex {alias!r} matches the empty string")
        return v

    @model_validator(mode="after")
    def _required_fields_per_kind(self) -> SourceSpec:
        if self.kind == "git" and not self.repo:
            raise ValueError(f"source {self.venue!r}: kind=git requires `repo`")
        if self.kind == "gitbook" and not self.llms_index:
            raise ValueError(f"source {self.venue!r}: kind=gitbook requires `llms_index`")
        return self

    @field_validator("metadata")
    @classmethod
    def _closed_vocabularies(
        cls, v: dict[str, list[MetadataRule]]
    ) -> dict[str, list[MetadataRule]]:
        return _check_vocabularies(v)


def _check_vocabularies(rules: dict[str, list[MetadataRule]]) -> dict[str, list[MetadataRule]]:
    for name, field_rules in rules.items():
        allowed = ALLOWED_VALUES.get(name)
        for rule in field_rules:
            if allowed is not None and rule.value is not None and rule.value not in allowed:
                raise ValueError(
                    f"{name}: {rule.value!r} is not one of {', '.join(sorted(allowed))}"
                )
    return rules


def _sources_hint(path: str | Path) -> str:
    """Where a problem in the sources file at `path` is fixed."""
    return f"fix the entry in {path} (docs/reference/corpus.md)"


class SourcesFile(BaseModel):
    """data/sources.yaml: the sources, and the metadata rules tried after every venue's own.

    An unknown top-level key is refused, as in `SourceSpec`.
    """

    model_config = {"extra": "forbid"}

    sources: list[SourceSpec] = []
    # `metadata:` with nothing under it reads as no rules.
    metadata: dict[str, list[MetadataRule]] | None = None


def venue_directory(raw_root: Path, spec: SourceSpec) -> Path:
    """The directory under `raw_root` that `spec`'s files are fetched into.

    Raises:
        UserError: `spec.venue` is empty or blank, holds a control character, is absolute,
            or leads to `raw_root` itself or outside it (a `..` segment, a symlink).
    """
    return confined(
        raw_root,
        spec.venue,
        what="venue",
        hint="fix the venue's name in the sources file (`paths.sources`, docs/reference/corpus.md)",
    )


def load_global_metadata_rules(path: str | Path) -> dict[str, list[MetadataRule]]:
    """The top-level `metadata:` block: rules tried after every venue's own.

    Raises:
        UserError: A rule names an unknown field or a value outside its field's vocabulary.
    """
    rules = dict(read_model(path, SourcesFile, hint=_sources_hint(path)).metadata or {})
    unknown = set(rules) - set(METADATA_FIELDS)
    if unknown:
        raise UserError(
            f"{formatting.plural(len(unknown), 'unknown metadata field')} in {path}: "
            f"{', '.join(sorted(unknown))}",
            hint=f"the fields are {', '.join(METADATA_FIELDS)}",
        )
    try:
        return _check_vocabularies(rules)
    except ValueError as exc:
        raise UserError(f"{path}: {exc}", hint=_sources_hint(path)) from exc


def venue_types(path: str | Path) -> dict[str, str]:
    """Each venue of sources.yaml mapped to its venue type."""
    return {spec.venue: spec.venue_type for spec in load_sources(path)}


def load_sources(path: str | Path) -> list[SourceSpec]:
    """The sources of sources.yaml.

    Raises:
        UserError: A venue repeats, which is refused since venue is a join key.
    """
    specs = read_model(path, SourcesFile, hint=_sources_hint(path)).sources

    seen: set[str] = set()
    for spec in specs:
        if spec.venue in seen:
            raise UserError(f"duplicate venue in {path}: {spec.venue!r}", hint=_sources_hint(path))
        seen.add(spec.venue)
    return specs
