"""The one value several things share, as a document states it once for all of them."""

from __future__ import annotations

from collections.abc import Hashable, Iterable


def shared_value[T: Hashable](values: Iterable[T]) -> T | None:
    """The one distinct value of `values`; None when they differ or there are none."""
    found = set(values)
    return found.pop() if len(found) == 1 else None
