"""Tests for `corpus.manifest`: reading, writing and fingerprinting the manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from crossvenue_rag.corpus.manifest import (
    ManifestRow,
    fingerprint,
    preserve_timestamps,
    read_manifest,
    write_manifest,
)
from crossvenue_rag.errors import UserError
from crossvenue_rag.run_files import file_sha256
from tests.conftest import REPO


def row(path: str, sha: str = "s1", fetched: str = "2026-09-20T00:00:00Z") -> ManifestRow:
    return ManifestRow("bybit", "cex", f"https://x/{path}", path, fetched, "c1", sha)


def test_rows_round_trip_sorted(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [row("b.md"), row("a.md")])
    assert [r.local_path for r in read_manifest(path)] == ["a.md", "b.md"]
    assert read_manifest(tmp_path / "absent.csv") == []


def test_a_venue_type_outside_cex_and_dex_is_refused(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [row("a.md")])
    path.write_text(path.read_text(encoding="utf-8").replace(",cex,", ",cefi,"), encoding="utf-8")
    with pytest.raises(
        UserError, match=r"manifest\.csv:2: venue_type must be cex or dex, got 'cefi'"
    ):
        read_manifest(path)


def test_an_unchanged_document_keeps_its_fetch_time():
    old = [row("a.md", fetched="2026-01-01T00:00:00Z"), row("b.md", fetched="2026-01-01T00:00:00Z")]
    new = [row("a.md"), row("b.md", sha="s2")]
    kept = preserve_timestamps(new, old)
    assert [r.fetched_at for r in kept] == ["2026-01-01T00:00:00Z", "2026-09-20T00:00:00Z"]


def test_the_fingerprint_is_the_file_hash_and_the_document_count(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [row("a.md"), row("b.md")])
    print_ = fingerprint(path)
    assert print_.manifest_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert print_.manifest_sha256 == file_sha256(path)
    assert print_.documents == 2


def test_the_fingerprint_counts_the_documents_the_manifest_reader_reads(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [row("a.md"), row("b.md")])
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
    assert fingerprint(path).documents == len(read_manifest(path)) == 2


@pytest.mark.parametrize(
    "meta", sorted((REPO / "evals" / "datasets").glob("*.meta.json")), ids=lambda p: p.name
)
def test_every_committed_corpus_snapshot_reads_back_unchanged(meta):
    import json

    from crossvenue_rag.corpus.manifest import CorpusSnapshot

    recorded = json.loads(meta.read_text(encoding="utf-8"))["corpus_snapshot"]
    snapshot = CorpusSnapshot.model_validate(recorded)
    assert json.dumps(snapshot.model_dump(exclude_unset=True)) == json.dumps(recorded)


def fetched_row(**patch: Any) -> ManifestRow:
    """A bybit manifest row fetched at a fixed time, with `patch` applied."""
    base = {
        "venue": "bybit",
        "venue_type": "cex",
        "source_url": "https://example.invalid/a.md",
        "local_path": "bybit/a.md",
        "fetched_at": "2026-01-01T00:00:00Z",
        "commit_sha": "abc123",
        "sha256": "deadbeef",
    }
    return ManifestRow(**{**base, **patch})


def test_refetching_unchanged_content_reproduces_the_manifest_exactly(tmp_path: Path):
    """Idempotence, so a diff in the manifest always means real drift in a source."""
    path = tmp_path / "manifest.csv"
    write_manifest(path, [fetched_row(local_path="bybit/b.md"), fetched_row()])
    first = path.read_text(encoding="utf-8")
    assert [r.local_path for r in read_manifest(path)] == ["bybit/a.md", "bybit/b.md"]

    refetched = [
        fetched_row(fetched_at="2026-06-01T12:00:00Z"),
        fetched_row(local_path="bybit/b.md"),
    ]
    write_manifest(path, preserve_timestamps(refetched, read_manifest(path)))
    assert path.read_text(encoding="utf-8") == first


def test_changed_content_gets_a_new_timestamp(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [fetched_row()])
    changed = [fetched_row(sha256="feedface", fetched_at="2026-06-01T12:00:00Z")]
    merged = preserve_timestamps(changed, read_manifest(path))
    assert merged[0].fetched_at == "2026-06-01T12:00:00Z"


def test_a_manifest_that_is_not_utf8_is_a_user_error(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UserError, match="is not UTF-8 text") as refused:
        read_manifest(path)
    assert refused.value.hint.startswith(f"restore {path},")


def test_a_manifest_that_is_not_utf8_is_a_user_error_when_fingerprinted(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_bytes(b"\xff\xfe")
    with pytest.raises(UserError, match="is not UTF-8 text") as refused:
        fingerprint(path)
    assert refused.value.hint.startswith(f"restore {path},")
