"""Tests for `crossvenue_rag.index.migrate`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crossvenue_rag.corpus.dedup import deduplicate
from crossvenue_rag.errors import UserError
from crossvenue_rag.index.migrate import (
    build_id_map,
    remap_jsonl,
)
from tests.conftest import occurrence


def old_chunk_line(chunk_id: str, venue: str, text: str) -> str:
    return json.dumps({"chunk_id": chunk_id, "venue": venue, "text": text})


def test_id_map_collapses_copies_onto_one_canonical_id(tmp_path: Path):
    old = tmp_path / "old.jsonl"
    old.write_text(
        "\n".join(
            [
                old_chunk_line("old-main", "binance", "limit 1200"),
                old_chunk_line("old-test", "binance", "limit 1200"),
                old_chunk_line("old-gone", "binance", "text the rebuild dropped"),
            ]
        )
        + "\n"
    )
    [canonical], _ = deduplicate([occurrence("binance", "a.md", "limit 1200")])
    id_map, orphans = build_id_map(old, [canonical], "v2")
    assert id_map.mapping == {"old-main": canonical.chunk_id, "old-test": canonical.chunk_id}
    assert orphans == ["old-gone"]


def test_remap_changes_only_the_id_fields_byte_for_byte(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    records = [
        {
            "id": "q1",
            "question": "na\u00efve \u2013 \u00fc?",
            "source_chunk_ids": ["a", "b"],
            "gold_chunk_ids": [],
        },
        {"id": "q2", "question": "other", "source_chunk_ids": ["zz"], "gold_chunk_ids": ["a"]},
    ]
    source = tmp_path / "in.jsonl"
    source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    target = tmp_path / "out.jsonl"
    report = remap_jsonl(
        source,
        target,
        IdMap({"a": "N", "b": "N"}, old_chunks_sha256="", to_index_version=""),
        ["source_chunk_ids", "gold_chunk_ids"],
        only=None,
    )

    out = [json.loads(line) for line in target.read_text().splitlines()]
    assert out[0]["source_chunk_ids"] == ["N"]  # two copies became one canonical id
    assert out[1]["source_chunk_ids"] == ["zz"]  # unmapped: left as it was
    assert out[1]["gold_chunk_ids"] == ["N"]
    assert report.unmapped == {"zz"}
    assert report.records_changed == 2
    assert target.read_text().endswith("\n")
    for before, after in zip(records, out, strict=True):
        for key in before:
            if key not in ("source_chunk_ids", "gold_chunk_ids"):
                assert json.dumps(before[key], ensure_ascii=False) == json.dumps(
                    after[key], ensure_ascii=False
                )


def test_remap_refuses_a_file_that_would_be_reserialized(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    source = tmp_path / "in.jsonl"
    source.write_text('{"id": "q1",  "source_chunk_ids": ["a"]}\n')  # two spaces
    with pytest.raises(UserError, match="does not round-trip"):
        remap_jsonl(
            source,
            tmp_path / "out.jsonl",
            IdMap({"a": "N"}, old_chunks_sha256="", to_index_version=""),
            ["source_chunk_ids"],
            only=None,
        )


def test_remap_names_a_missing_field_as_the_cause(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps({"id": "q1", "gold_chunk_ids": ["a"]}) + "\n")
    with pytest.raises(UserError, match="record lacks 1 field to remap: source_chunk_ids"):
        remap_jsonl(
            source,
            tmp_path / "out.jsonl",
            IdMap({"a": "N"}, old_chunks_sha256="", to_index_version=""),
            ["source_chunk_ids", "gold_chunk_ids"],
            only=None,
        )
    assert not (tmp_path / "out.jsonl").exists()


def test_a_map_or_a_dataset_that_is_not_utf8_is_a_user_error_naming_it(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe{}")
    with pytest.raises(UserError, match=r"bad\.json is not UTF-8 text"):
        IdMap.load(bad)
    with pytest.raises(UserError, match=r"bad\.json is not UTF-8 text"):
        remap_jsonl(
            bad,
            tmp_path / "out.jsonl",
            IdMap({}, old_chunks_sha256="", to_index_version=""),
            ["gold_chunk_ids"],
            only=None,
        )
    assert not (tmp_path / "out.jsonl").exists()


def test_a_field_named_twice_is_rewritten_and_counted_once(tmp_path: Path):
    from crossvenue_rag.index.migrate import IdMap

    source = tmp_path / "in.jsonl"
    source.write_text(json.dumps({"id": "a", "xs": ["old1"]}) + "\n")
    report = remap_jsonl(
        source,
        tmp_path / "out.jsonl",
        IdMap({"old1": "new1"}, old_chunks_sha256="", to_index_version=""),
        ["xs", "xs"],
        only=None,
    )
    assert (report.ids_before, report.ids_after) == (1, 1)


def test_write_id_map_writes_the_map_to_the_built_index_and_refuses_lost_text(cfg, tmp_path):
    from crossvenue_rag.index.migrate import IdMap, write_id_map
    from crossvenue_rag.index_version import write_index_meta
    from tests.conftest import build_corpus

    chunks = build_corpus(cfg)
    old = cfg.paths.processed / "chunks.jsonl"
    with pytest.raises(UserError, match="no index is built") as unbuilt:
        write_id_map(cfg, old, tmp_path / "map.json")
    assert unbuilt.value.hint == "build the index with `make ingest index`"
    version = write_index_meta(cfg, len(chunks))
    mapping = write_id_map(cfg, old, tmp_path / "map.json")
    assert mapping.canonical_ids == len(chunks)
    assert IdMap.load(tmp_path / "map.json").to_index_version == version
    lost = tmp_path / "lost.jsonl"
    lost.write_text(old_chunk_line("gone", "binance", "no such text") + "\n")
    with pytest.raises(UserError, match="1 old chunk without identical text"):
        write_id_map(cfg, lost, tmp_path / "lost-map.json")
    assert not (tmp_path / "lost-map.json").exists()
