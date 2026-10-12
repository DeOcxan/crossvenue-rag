"""Structural metadata for chunk sources, derived by the rules in data/sources.yaml.

Each field is read off a document's path and headings: the venue's own rules
are tried in order, then the global ones, and the first match wins. No match
leaves the field `None`, which means "the layout does not say" and is never a
default guess.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..models import METADATA_FIELDS, Chunk, ChunkSource
from .artifacts import FieldCoverage, MetadataCoverage
from .sources import (
    GROUP_REF_RE,
    MetadataRule,
    SourceSpec,
    load_global_metadata_rules,
    load_sources,
)

# The decimals of a field's coverage share in corpus_stats.json.
SHARE_DECIMALS = 4


@dataclass(slots=True)
class _Compiled:
    value: str | None
    path: re.Pattern[str] | None
    heading: re.Pattern[str] | None


def _compile(rules: Sequence[MetadataRule]) -> list[_Compiled]:
    return [
        _Compiled(
            r.value,
            re.compile(r.path) if r.path is not None else None,
            re.compile(r.heading) if r.heading is not None else None,
        )
        for r in rules
    ]


class MetadataRules:
    """Compiled rules for every venue, ready to annotate chunk sources.

    The fields and their values: `environment` (mainnet, testnet), `api_version`
    (open, such as v5 or infinity), `product` (spot, perp) and `doc_type`
    (reference, guide, faq, changelog), each `None` where no rule matches.
    """

    def __init__(
        self,
        specs: Sequence[SourceSpec],
        global_rules: dict[str, list[MetadataRule]] | None = None,
    ) -> None:
        """Compiles each venue's own rules in `specs`, followed by `global_rules`."""
        global_rules = global_rules or {}
        self._rules: dict[str, dict[str, list[_Compiled]]] = {}
        for spec in specs:
            own: dict[str, list[MetadataRule]] = {str(k): v for k, v in spec.metadata.items()}
            self._rules[spec.venue] = {
                name: _compile([*own.get(name, []), *global_rules.get(name, [])])
                for name in METADATA_FIELDS
            }
        self._global = {name: _compile(global_rules.get(name, [])) for name in METADATA_FIELDS}

    @classmethod
    def from_file(cls, sources_path: str | Path) -> MetadataRules:
        """The rules of every venue and the global rules in `sources_path`."""
        return cls(load_sources(sources_path), load_global_metadata_rules(sources_path))

    def derive(
        self, venue: str, doc_path: str, heading_path: Sequence[str]
    ) -> dict[str, str | None]:
        """Each metadata field's value for one occurrence; unknown venues get the global rules."""
        rules = self._rules.get(venue, self._global)
        return {name: _first_match(rules[name], doc_path, heading_path) for name in METADATA_FIELDS}

    def annotate(self, source: ChunkSource) -> None:
        """Sets every metadata field of `source` in place from its venue, path and headings."""
        for name, value in self.derive(source.venue, source.doc_path, source.heading_path).items():
            setattr(source, name, value)


def _first_match(
    rules: Sequence[_Compiled], doc_path: str, heading_path: Sequence[str]
) -> str | None:
    for rule in rules:
        path_match = rule.path.search(doc_path) if rule.path is not None else None
        if rule.path is not None and path_match is None:
            continue
        if rule.heading is not None and not any(rule.heading.search(h) for h in heading_path):
            continue
        if rule.value is None:
            return None
        if path_match is None:
            return rule.value
        return _expand(rule.value, path_match)
    return None


def _expand(value: str, match: re.Match[str]) -> str:
    """Substitutes `{n}` in a rule value with capture group n of the path match."""
    return GROUP_REF_RE.sub(lambda ref: match.group(int(ref.group(1))) or "", value)


def coverage(chunks: Sequence[Chunk]) -> MetadataCoverage:
    """Share of chunks each field was assigned on, and where it was not.

    Measured on the canonical chunk's primary occurrence, the value a
    citation carries. `unassigned_by_venue` is where the rules did not fire.

    Saved precision: shares at four decimals.
    """
    total = len(chunks)
    fields: dict[str, FieldCoverage] = {}
    for name in METADATA_FIELDS:
        values = Counter(getattr(c, name) for c in chunks)
        unassigned: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            if getattr(chunk, name) is None:
                unassigned[chunk.venue] += 1
        assigned = total - values.get(None, 0)
        fields[name] = FieldCoverage(
            assigned=assigned,
            share=round(assigned / total, SHARE_DECIMALS) if total else 0.0,
            values={str(k): v for k, v in sorted(values.items(), key=lambda kv: -kv[1]) if k},
            unassigned_by_venue=dict(sorted(unassigned.items())),
        )
    return MetadataCoverage(chunks=total, fields=fields)
