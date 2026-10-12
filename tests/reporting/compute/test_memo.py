"""Tests for `crossvenue_rag.reporting.compute.memo`: caching a compute function per `Sources`."""

from __future__ import annotations

import gc
import weakref

from crossvenue_rag.reporting.compute.memo import per_sources
from crossvenue_rag.reporting.sources import Sources


def counting():
    """A per-Sources function returning a fresh list per computation, and its call log."""
    calls = []

    @per_sources
    def compute(src, *args, **kwargs):
        calls.append((src, args, kwargs))
        return [len(calls)]

    return compute, calls


def test_a_second_call_on_the_same_sources_returns_the_cached_value(tmp_path):
    compute, calls = counting()
    src = Sources(tmp_path)
    first = compute(src)
    assert compute(src) is first
    assert len(calls) == 1


def test_another_sources_object_is_computed_again_even_on_the_same_root(tmp_path):
    compute, calls = counting()
    first = compute(Sources(tmp_path))
    second = compute(Sources(tmp_path))
    assert first is not second
    assert len(calls) == 2


def test_each_set_of_further_arguments_is_cached_on_its_own(tmp_path):
    compute, calls = counting()
    src = Sources(tmp_path)
    one = compute(src, "a", limit=1)
    two = compute(src, "b", limit=1)
    assert one is not two
    assert compute(src, "a", limit=1) is one
    assert len(calls) == 2


def test_keyword_arguments_in_another_order_hit_the_same_entry(tmp_path):
    compute, calls = counting()
    src = Sources(tmp_path)
    value = compute(src, x=1, y=2)
    assert compute(src, y=2, x=1) is value
    assert len(calls) == 1


def test_the_cache_does_not_keep_a_sources_object_alive(tmp_path):
    compute, calls = counting()
    src = Sources(tmp_path)
    compute(src)
    ref = weakref.ref(src)
    calls.clear()
    del src
    gc.collect()
    assert ref() is None


def test_the_wrapper_keeps_the_function_name_and_docstring():
    @per_sources
    def stated_limits(src):
        """The limits a document states."""
        return src

    assert stated_limits.__name__ == "stated_limits"
    assert stated_limits.__doc__ == "The limits a document states."
