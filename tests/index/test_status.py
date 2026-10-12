"""Tests for `index.status`: what `crossvenue check` reports."""

from __future__ import annotations

from crossvenue_rag.index.status import build_status

UNREACHABLE = "postgresql://x@127.0.0.1:1/x"


def test_a_missing_build_and_an_unreachable_database_are_reported_not_raised(cfg):
    cfg.store.dsn = UNREACHABLE
    status = build_status(cfg)
    assert status.venues > 0
    assert [(a.name, a.present) for a in status.artifacts] == [
        ("manifest", False),
        ("chunks", False),
        ("bm25 index", False),
    ]
    assert isinstance(status.store, str)
    assert status.index_version is None
    assert status.inconsistency is not None and "no index metadata" in status.inconsistency
    assert status.agreeing == "config, manifest and index_meta.json"
