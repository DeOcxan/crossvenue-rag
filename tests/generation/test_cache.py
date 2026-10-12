"""Tests for `generation.cache`: canonical keys and the on-disk store."""

from __future__ import annotations

import json

from crossvenue_rag.generation.cache import CacheEntry, DiskCache, cache_key
from crossvenue_rag.generation.records import CallRecord


def test_the_key_ignores_key_order_and_changes_with_any_value():
    a = cache_key({"model": "m", "params": {"t": 0, "max": 5}})
    assert a == cache_key({"params": {"max": 5, "t": 0}, "model": "m"})
    assert a != cache_key({"model": "m", "params": {"t": 0, "max": 6}})
    assert len(a) == 64


RECORD = {
    "requested_model": "m",
    "actual_model": "m-1",
    "provider": "p",
    "input_tokens": 10,
    "output_tokens": 5,
    "cost_usd": 0.1,
    "latency_ms": 12.5,
    "retries": 0,
    "cache_key": "k",
    "params": {"max_tokens": 100},
}


def test_an_entry_round_trips_as_a_typed_entry_marked_cached_and_a_miss_is_none(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    key = cache_key({"q": "binance ban"})
    assert cache.get(key) is None
    entry = {"request": {"model": "m"}, "content": {"answer": "\u00e9"}, "record": RECORD}
    path = cache.put(key, entry)
    assert path == tmp_path / "cache" / key[:2] / f"{key}.json"
    hit = cache.get(key)
    assert isinstance(hit, CacheEntry)
    assert hit.content == {"answer": "\u00e9"}
    assert hit.record == CallRecord.from_cache(RECORD) and hit.record.cached
    assert json.loads(path.read_text(encoding="utf-8")) == entry
    assert not list(path.parent.glob("*.tmp"))


def test_an_entry_without_a_call_record_is_a_user_error_naming_the_file(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError

    cache = DiskCache(tmp_path)
    key = cache_key({"a": 1})
    cache.put(key, {"content": {"answer": "x"}, "record": {"cost_usd": 0.1}})
    with pytest.raises(UserError, match="requested_model") as caught:
        cache.get(key)
    assert str(cache.path(key)) in caught.value.hint


def test_an_entry_that_is_not_json_is_a_user_error_naming_the_file(tmp_path):
    import pytest

    from crossvenue_rag.errors import UserError

    cache = DiskCache(tmp_path)
    key = cache_key({"a": 1})
    cache.path(key).parent.mkdir(parents=True)
    cache.path(key).write_text("{not json", encoding="utf-8")
    with pytest.raises(UserError, match="Invalid JSON") as caught:
        cache.get(key)
    assert str(cache.path(key)) in caught.value.hint
