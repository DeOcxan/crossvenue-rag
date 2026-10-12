"""Tests for `crossvenue_rag.index_version`."""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.manifest import ManifestRow, write_manifest
from crossvenue_rag.errors import UserError
from crossvenue_rag.index_version import (
    compute_index_version,
    require_current_index,
    write_index_meta,
)


def manifest(cfg) -> None:
    write_manifest(
        cfg.paths.manifest,
        [ManifestRow("bybit", "cex", "https://x", "bybit/a.md", "t", "0" * 40, "0" * 64)],
    )


def test_index_version_moves_with_chunking_and_model(cfg):
    manifest(cfg)
    base = compute_index_version(cfg)
    assert compute_index_version(cfg) == base
    cfg.chunking.max_tokens += 1
    assert compute_index_version(cfg) != base
    cfg.chunking.max_tokens -= 1
    cfg.embedding.model_name = "other/model"
    assert compute_index_version(cfg) != base


def test_a_stale_or_split_index_is_refused(cfg):
    manifest(cfg)
    with pytest.raises(UserError, match="no index is built: no index metadata"):
        require_current_index(cfg)
    version = write_index_meta(cfg, chunks=1)
    assert require_current_index(cfg, [version]) == version
    with pytest.raises(UserError, match="Postgres rows carry"):
        require_current_index(cfg, ["something-else"])
    with pytest.raises(UserError) as two:
        require_current_index(cfg, ["aaaa", "bbbb"])
    assert str(two.value).startswith("Postgres rows carry index_version aaaa, bbbb, the BM25")
    with pytest.raises(UserError, match="index_version none, the BM25"):
        require_current_index(cfg, [])
    cfg.chunking.overlap_tokens += 1
    with pytest.raises(UserError, match="this config and manifest imply"):
        require_current_index(cfg)


def test_model_revisions_are_recorded_beside_the_version_not_in_it(cfg):
    import json

    manifest(cfg)
    base = compute_index_version(cfg)
    cfg.embedding.revision = "rev-a"
    cfg.reranker.revision = "rev-b"
    assert compute_index_version(cfg) == base
    write_index_meta(cfg, chunks=1)
    meta = json.loads((cfg.paths.index / "index_meta.json").read_text())
    assert meta["model_revisions"] == {"embedding": "rev-a", "reranker": "rev-b"}


def test_the_version_of_a_fixed_config_and_manifest_is_pinned(tmp_path):
    """The hash covers the components' canonical JSON; a change to how they are
    serialized would move every recorded `index_version`."""
    from crossvenue_rag.config import Config

    manifest = tmp_path / "m.csv"
    manifest.write_text("venue,url\nbybit,https://example.invalid\n", encoding="utf-8")
    cfg = Config()
    cfg.paths.manifest = manifest
    assert compute_index_version(cfg) == "9c450c5ea637f9a9"


def test_the_recorded_version_is_read_back_and_a_missing_file_is_none(cfg):
    from crossvenue_rag.index_version import index_meta_path, recorded_index_version

    manifest(cfg)
    assert recorded_index_version(index_meta_path(cfg)) is None
    version = write_index_meta(cfg, chunks=1)
    assert recorded_index_version(index_meta_path(cfg)) == version


@pytest.mark.parametrize("content", [b"{", b"\xff\xfe", b"[]", b"{}", b'{"index_version": 3}'])
def test_a_damaged_index_meta_is_a_user_error_naming_the_file(tmp_path, content):
    from crossvenue_rag.index_version import recorded_index_version

    path = tmp_path / "index_meta.json"
    path.write_bytes(content)
    with pytest.raises(UserError) as refused:
        recorded_index_version(path)
    assert str(path) in str(refused.value)
    assert "make ingest index" in refused.value.hint


def test_a_directory_where_index_meta_or_the_manifest_should_be_is_a_user_error(cfg):
    from crossvenue_rag.index_version import (
        index_components,
        index_meta_path,
        recorded_index_version,
    )

    index_meta_path(cfg).mkdir(parents=True)
    with pytest.raises(UserError, match="is a directory"):
        recorded_index_version(index_meta_path(cfg))
    cfg.paths.manifest.mkdir(parents=True)
    with pytest.raises(UserError, match="is a directory"):
        index_components(cfg)
