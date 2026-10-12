"""Caching a compute function per `Sources`."""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Hashable
from typing import Concatenate

from ..sources import Sources


def per_sources[**P, R](
    fn: Callable[Concatenate[Sources, P], R],
) -> Callable[Concatenate[Sources, P], R]:
    """`fn`, computed once per `Sources` object and set of further arguments.

    The further arguments must be hashable. The cache holds the `Sources` weakly, so a
    test's fixture root is not kept alive. A cached value is shared between callers, who
    must not change it.
    """
    cache: weakref.WeakKeyDictionary[Sources, dict[Hashable, R]] = weakref.WeakKeyDictionary()

    @functools.wraps(fn)
    def cached(src: Sources, /, *args: P.args, **kwargs: P.kwargs) -> R:
        results = cache.setdefault(src, {})
        key = (args, tuple(sorted(kwargs.items())))
        if key not in results:
            results[key] = fn(src, *args, **kwargs)
        return results[key]

    return cached
