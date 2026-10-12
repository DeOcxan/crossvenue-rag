"""Tests for `cli.index`: index --skip-dense, id-map and the remap-ids version check."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from crossvenue_rag.cli import app
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.corpus.artifacts import stats_path
from crossvenue_rag.errors import UserError
from crossvenue_rag.index.migrate import IdMap
from tests.cli.conftest import current_version


def test_the_bm25_index_is_built_and_stamped_with_the_version(config_file):
    result = CliRunner().invoke(app, ["index", "-c", str(config_file), "--skip-dense"])
    assert result.exit_code == 0, result.output
    assert "skipped dense index" in result.output
    cfg = load_config(config_file)
    meta = json.loads((cfg.paths.index / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["index_version"] == current_version(config_file)


def test_chunks_ingested_under_another_config_are_not_indexed(config_file):
    cfg = load_config(config_file)
    stats = json.loads(stats_path(cfg).read_text(encoding="utf-8"))
    stats["index_version"] = "0000000000000000"
    stats_path(cfg).write_text(json.dumps(stats), encoding="utf-8")
    result = CliRunner().invoke(app, ["index", "-c", str(config_file), "--skip-dense"])
    assert isinstance(result.exception, UserError)
    assert "was ingested as index_version 0000000000000000" in str(result.exception)


def test_the_same_chunks_map_onto_themselves(config_file, tmp_path):
    cfg = load_config(config_file)
    out = tmp_path / "map.json"
    old = cfg.paths.processed / "chunks.jsonl"
    result = CliRunner().invoke(app, ["id-map", str(old), "-o", str(out), "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    mapping = IdMap.load(out)
    assert mapping.mapping and all(k == v for k, v in mapping.mapping.items())
    assert mapping.to_index_version == current_version(config_file)


def test_a_map_to_another_index_is_refused(config_file, tmp_path):
    path = tmp_path / "map.json"
    IdMap({"a": "b"}, to_index_version="other", old_chunks_sha256="").save(path)
    source = tmp_path / "q.jsonl"
    source.write_text("", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["remap-ids", str(source), str(tmp_path / "out.jsonl"), "--map", str(path),
              "-c", str(config_file)],
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert "the map targets index other" in str(result.exception)


def test_remap_ids_with_no_index_built_names_the_build_as_the_fix(config_file, tmp_path):
    from crossvenue_rag.index_version import BUILD_INDEX_HINT, index_meta_path

    index_meta_path(load_config(config_file)).unlink()
    path = tmp_path / "map.json"
    IdMap({"a": "b"}, to_index_version="v2", old_chunks_sha256="").save(path)
    source = tmp_path / "q.jsonl"
    source.write_text("", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["remap-ids", str(source), str(tmp_path / "out.jsonl"), "--map", str(path),
              "-c", str(config_file)],
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert str(result.exception).startswith("no index is built: no index metadata at ")
    assert result.exception.hint == BUILD_INDEX_HINT


def test_remap_ids_writes_the_dataset_and_names_unmapped_ids(config_file, tmp_path):
    path = tmp_path / "map.json"
    IdMap({"a": "N"}, to_index_version=current_version(config_file), old_chunks_sha256="").save(
        path
    )
    source = tmp_path / "q.jsonl"
    source.write_text(json.dumps({"id": "q1", "gold_chunk_ids": ["a", "zz"]}) + "\n")
    target = tmp_path / "out.jsonl"
    result = CliRunner().invoke(
        app, ["remap-ids", str(source), str(target), "--map", str(path),
              "--field", "gold_chunk_ids", "--only", "q1", "-c", str(config_file)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "1 record, 1 changed; ids 2 -> 2; unmapped (left as-is): 1" in result.output
    assert "unmapped zz" in result.output
    assert json.loads(target.read_text())["gold_chunk_ids"] == ["N", "zz"]


def test_remap_ids_rewrites_a_sidecar_in_place(config_file, tmp_path):
    version = current_version(config_file)
    path = tmp_path / "map.json"
    IdMap({"a": "N"}, to_index_version=version, old_chunks_sha256="").save(path)
    sidecar = tmp_path / "q.rounds.json"
    sidecar.write_text(json.dumps({"q1": {"1": {"pool": ["a"], "relevant": ["a"]}}}))
    result = CliRunner().invoke(
        app, ["remap-ids", str(sidecar), "--map", str(path), "-c", str(config_file)]
    )
    assert result.exit_code == 0, result.output
    assert str(sidecar.name) in result.output.replace("\n", "")
    entry = json.loads(sidecar.read_text())["q1"]["1"]
    assert entry["pool"] == ["N"] and entry["index_version"] == version


def test_id_map_refuses_old_chunks_whose_text_is_gone(config_file, tmp_path):
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"chunk_id": "gone", "venue": "binance", "text": "no such"}) + "\n")
    out = tmp_path / "map.json"
    result = CliRunner().invoke(app, ["id-map", str(old), "-o", str(out), "-c", str(config_file)])
    assert isinstance(result.exception, UserError)
    assert "1 old chunk without identical text" in str(result.exception)
    assert not out.exists()


def test_the_dense_index_is_written_with_the_version(config_file, monkeypatch):
    written = []

    def fake_index_chunks(cfg, chunks, version):
        written.append(version)
        return len(chunks)

    monkeypatch.setattr("crossvenue_rag.index.build.index_chunks", fake_index_chunks)
    result = CliRunner().invoke(app, ["index", "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    assert written == [current_version(config_file)]
    assert "pgvector" in result.output and "index_version" in result.output


def test_remap_ids_refuses_a_field_for_a_journal(config_file, tmp_path):
    path = tmp_path / "map.json"
    IdMap({"a": "N"}, to_index_version=current_version(config_file), old_chunks_sha256="").save(
        path
    )
    sidecar = tmp_path / "q.rounds.json"
    sidecar.write_text(json.dumps({"q1": {"1": {"pool": ["a"], "relevant": ["a"]}}}))
    result = CliRunner().invoke(
        app, ["remap-ids", str(sidecar), "--map", str(path), "--field", "gold_chunk_ids",
              "-c", str(config_file)],
    )  # fmt: skip
    assert isinstance(result.exception, UserError)
    assert "--field gold_chunk_ids would be ignored" in str(result.exception)


@pytest.mark.parametrize("chunk_id", ["[/x]", "[bold]"])
def test_remap_ids_prints_a_bracketed_unmapped_id_and_path_as_written(
    chunk_id, config_file, tmp_path
):
    path = tmp_path / "map.json"
    IdMap({"a": "N"}, to_index_version=current_version(config_file), old_chunks_sha256="").save(
        path
    )
    source = tmp_path / "q.jsonl"
    source.write_text(json.dumps({"id": "q1", "gold_chunk_ids": [chunk_id]}) + "\n")
    field = ["--field", "gold_chunk_ids"]
    folder = tmp_path / "[" / "x]" if chunk_id == "[/x]" else tmp_path / "[bold]"
    folder.mkdir(parents=True)
    target = folder / "out.jsonl"
    result = CliRunner().invoke(
        app,
        ["remap-ids", str(source), str(target), "--map", str(path), *field, "-c", str(config_file)],
    )
    assert result.exit_code == 0, result.output
    assert f"unmapped {chunk_id}" in result.output
    assert str(target) in "".join(result.output.split("\n"))
