"""Retrieval filters, resolved once into the allow-lists every retriever side receives.

`venue` and `venue_type` resolve to a venue allow-list and the structural
metadata fields to a chunk-id allow-list, so the sparse and dense sides search the
same chunks. A metadata filter matches a chunk if any of its occurrences
matches, and `None` in a filter value matches chunks where no rule assigned the
field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .. import formatting
from ..errors import UserError
from ..models import METADATA_FIELDS, Chunk, MetadataField

# `{"venue": "bybit"}`, `{"venue": [...], "venue_type": "dex"}`,
# `{"environment": "testnet", "api_version": ["v5"]}`. Spelled out rather than
# `dict[str, Any]`: the keys and the two shapes are the whole contract, and
# `Any` here would hide a typo'd filter until it silently matched nothing.
Filters = dict[str, Sequence[str | None] | str | None]


VENUE_FILTER_KEYS = ("venue", "venue_type")


FILTER_KEYS = (*VENUE_FILTER_KEYS, *METADATA_FIELDS)


def _require_venue_options(
    venues: Sequence[str], venue_type: str | None, venue_types: Mapping[str, str]
) -> None:
    """Refuses a `--venue` or `--venue-type` outside `venue_types`, naming the option."""
    unknown = sorted(set(venues) - set(venue_types))
    if unknown:
        raise UserError(
            f"--venue {', '.join(unknown)}: {formatting.plural(len(unknown), 'unknown venue')}",
            hint="pass --venue one of the venues of the sources file (`paths.sources`): "
            f"{', '.join(sorted(venue_types))}",
        )
    types = sorted(set(venue_types.values()))
    if venue_type is not None and venue_type not in types:
        raise UserError(
            f"--venue-type {venue_type!r} is not a venue type",
            hint="pass --venue-type one of the venue types of the sources file "
            f"(`paths.sources`): {' or '.join(types)}",
        )
    if venue_type is not None and venues and not any(venue_types[v] == venue_type for v in venues):
        raise UserError(
            f"no --venue given ({', '.join(sorted(venues))}) is of --venue-type {venue_type}",
            hint="drop one of the two options, or pass a venue of that type",
        )


def filters_from_options(
    venues: Sequence[str],
    venue_type: str | None,
    metadata: Sequence[str],
    venue_types: Mapping[str, str],
) -> Filters | None:
    """The filter dict of `crossvenue retrieve`'s options, `None` when there is none.

    `metadata` holds `FIELD=VALUE` strings: a field given twice ORs its values,
    and the value `none` matches chunks where no rule assigned the field.
    `venue_types` maps each venue of data/sources.yaml to its type; the venue
    options are checked against it here, before any model is loaded.

    Raises:
        UserError: A `--venue` is not a venue, `--venue-type` is not a venue type, or no
            venue given is of the type given. A filter is not `FIELD=VALUE`, or its field
            is not one of `METADATA_FIELDS`: venues are restricted by `--venue` and
            `--venue-type`.
    """
    _require_venue_options(venues, venue_type, venue_types)
    filters: Filters = {}
    if venues:
        filters["venue"] = list(venues)
    if venue_type:
        filters["venue_type"] = venue_type
    values: dict[str, list[str | None]] = {}
    for raw in metadata:
        name, sep, value = raw.partition("=")
        if not sep:
            raise UserError(
                f"bad --filter {raw!r}",
                hint="pass --filter as FIELD=VALUE, for example environment=testnet",
            )
        if name.strip() not in METADATA_FIELDS:
            raise UserError(
                f"bad --filter {raw!r}: {name.strip() or 'the field'} is not a metadata field",
                hint=f"the fields are {', '.join(METADATA_FIELDS)}; restrict venues with "
                "--venue and --venue-type",
            )
        values.setdefault(name.strip(), []).append(
            None if value.strip().lower() == "none" else value.strip()
        )
    filters.update(values)
    return filters or None


def _names(filters: Filters, key: str) -> list[str]:
    """A venue filter's value as a list of names; None is not a venue."""
    value = filters[key]
    names = [value] if isinstance(value, str) else list(value or [])
    if value is None or any(name is None for name in names):
        raise ValueError(f"{key} filter: None is not a {key}")
    return [name for name in names if name is not None]


def resolve_venues(filters: Filters | None, venue_types: Mapping[str, str]) -> list[str] | None:
    """A filter dict as an explicit venue allow-list (or None for "all").

    `venue_type` is expanded here rather than pushed down, so BM25, which has no
    venue types, and Postgres apply byte-identical filters. The venues
    a user names are checked where they enter (`filters_from_options`), against the
    same map of data/sources.yaml that `venue_types` holds.

    Raises:
        ValueError: A key is not in `FILTER_KEYS`, a venue or venue type is not in
            `venue_types`, or the venues and the type select no venue together: both
            indexes read an empty allow-list as "no filter".
    """
    if not filters:
        return None

    unknown = set(filters) - set(FILTER_KEYS)
    if unknown:
        raise ValueError(f"unsupported filter keys: {', '.join(sorted(unknown))}")

    selected: set[str] | None = None

    if "venue" in filters:
        wanted = _names(filters, "venue")
        missing = set(wanted) - set(venue_types)
        if missing:
            raise ValueError(f"venues not in the venue map: {', '.join(sorted(missing))}")
        selected = set(wanted)

    if "venue_type" in filters:
        wanted_type = _names(filters, "venue_type")
        by_type = {v for v, t in venue_types.items() if t in wanted_type}
        if not by_type:
            raise ValueError(f"no venues of type {', '.join(wanted_type)}")
        if selected is not None and not selected & by_type:
            raise ValueError(
                f"no venue is both in {', '.join(sorted(selected))} and of type "
                f"{', '.join(wanted_type)}"
            )
        selected = by_type if selected is None else selected & by_type

    return sorted(selected) if selected is not None else None


def resolve_metadata(filters: Filters | None, chunks: Sequence[Chunk]) -> set[str] | None:
    """The metadata part of a filter dict as a chunk-id allow-list.

    A chunk matches if any of its occurrences matches: a text deduplicated from
    `rest-api.md` and `testnet/rest-api.md` is testnet documentation as much as
    mainnet documentation. None when no metadata key is present. Several keys
    are ANDed; the values of one key are ORed. A value no chunk carries is an
    error rather than an empty result, for the same reason an unknown venue is.

    Raises:
        UserError: A metadata value of `filters` is carried by no chunk.
    """
    wanted: dict[MetadataField, set[str | None]] = {}
    for name in METADATA_FIELDS:
        if filters and name in filters:
            value = filters[name]
            wanted[name] = {value} if value is None or isinstance(value, str) else set(value)
    if not wanted:
        return None

    for name, values in wanted.items():
        present = set().union(*(c.metadata_values(name) for c in chunks)) if chunks else set()
        missing = values - present
        if missing:
            known = sorted(v for v in present if v is not None)
            raise UserError(
                f"no chunk has {name} in {', '.join(sorted(map(str, missing)))}",
                hint=f"known values: {', '.join(known)}",
            )

    return {
        chunk.chunk_id
        for chunk in chunks
        if all(chunk.metadata_values(name) & values for name, values in wanted.items())
    }
