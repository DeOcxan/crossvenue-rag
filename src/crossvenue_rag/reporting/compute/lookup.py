"""A value of a saved table or mapping by its key, for the compute functions."""

from __future__ import annotations

from collections.abc import Hashable, Mapping


def entry[K: Hashable, V](table: Mapping[K, V], key: K, what: str) -> V:
    """The value of `key` in `table`, which `what` names (`summary.csv of <run>`).

    Raises:
        ValueError: `table` has no `key`; the message names both.
    """
    if key not in table:
        raise ValueError(f"{what} has no {key!r}")
    return table[key]
