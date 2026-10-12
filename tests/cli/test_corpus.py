"""Tests for `crossvenue_rag.cli.corpus`."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from crossvenue_rag.cli import app
from crossvenue_rag.cli import corpus as _under_test

DDL = ("CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE", "GRANT")


class RecordingConnection:
    """Stands in for a psycopg connection and records every statement."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.read_only = False

    def __enter__(self) -> RecordingConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, query, params=None):
        self.statements.append(repr(query))
        return self

    def fetchone(self):
        return (0,)

    def fetchall(self):
        return []


def test_check_opens_a_read_only_connection_and_issues_no_ddl(monkeypatch):
    import pgvector.psycopg
    import psycopg

    connections: list[RecordingConnection] = []

    def fake_connect(dsn):
        connections.append(RecordingConnection())
        return connections[-1]

    monkeypatch.setattr(psycopg, "connect", fake_connect)
    monkeypatch.setattr(pgvector.psycopg, "register_vector", lambda conn: None)

    result = CliRunner().invoke(app, ["check", "--config", "configs/default.yaml"])

    assert result.exit_code == 0, result.output
    assert "pgvector     connected" in " ".join(result.output.split("\n"))
    [conn] = connections
    assert conn.read_only is True
    assert conn.statements, "check should query the store"
    assert not [s for s in conn.statements if any(word in s.upper() for word in DDL)]


def _manifest_row(venue: str, sha: str):
    from crossvenue_rag.corpus.manifest import ManifestRow

    return ManifestRow(venue, "cex", "https://example.invalid/a", "a.md", "t", sha, "0" * 64)


def _fake_fetch(*, written: bool):
    from crossvenue_rag.corpus.fetch import FetchReport

    def fetch_all(cfg, *, only=None, follow_branch=False, allow_missing=False):
        rows = [_manifest_row("bybit", "a" * 40), _manifest_row("okx", "")]
        return FetchReport(rows=rows, matched={"bybit": 1, "okx": 1}, manifest_written=written)

    return fetch_all


def test_fetch_prints_the_summary_and_leaves_the_manifest_alone(config_file, monkeypatch):

    monkeypatch.setattr(_under_test, "fetch_all", _fake_fetch(written=False))
    result = CliRunner().invoke(app, ["fetch", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "bybit" in result.output and "okx" in result.output
    assert "(2 documents); the manifest is unchanged" in " ".join(result.output.split())


def test_fetch_follow_branch_reports_the_rewritten_manifest(config_file, monkeypatch):

    monkeypatch.setattr(_under_test, "fetch_all", _fake_fetch(written=True))
    args = ["fetch", "--config", str(config_file), "--follow-branch", "--venue", "bybit"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "manifest rewritten" in result.output


def test_ingest_chunks_the_corpus_and_prints_its_statistics(config_file, monkeypatch):
    import crossvenue_rag.corpus.ingest as ingest
    from tests.conftest import count_words

    # The name ingest resolves: patching `corpus.tokens` would load the real tokenizer.
    counters: list[str] = []

    def counter_for(name, *args, **kwargs):
        counters.append(name)
        return count_words

    monkeypatch.setattr(ingest, "token_counter_for", counter_for)
    args = ["ingest", "--config", str(config_file)]
    args += ["--set", "chunking.max_tokens=80", "--set", "chunking.min_tokens=10"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "chunks ->" in result.output and "corpus by venue" in result.output
    assert len(counters) == 1


def test_stats_prints_every_section_the_statistics_file_holds(config_file, tmp_path):
    import json

    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.corpus.artifacts import stats_path

    path = stats_path(load_config(config_file))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["dedup"] = {
        "chunks_before": 3,
        "chunks_after": 2,
        "removed": 1,
        "groups": 1,
        "occurrences_in_groups": 2,
        "cross_venue_groups": 0,
        "cross_venue_examples": [],
        "groups_by_venue": {},
        "testnet_mirror_groups": 0,
    }
    data["metadata_coverage"] = {"chunks": 2, "fields": {}}
    data["index_version"] = "v-test"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = CliRunner().invoke(app, ["stats", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "chunk length distribution" in result.output
    assert "exact deduplication" in result.output and "metadata coverage" in result.output
    assert "index_version (of this ingest): v-test" in result.output


def test_stats_without_an_ingest_is_a_user_error(config_file):
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.corpus.artifacts import stats_path
    from crossvenue_rag.errors import UserError

    stats_path(load_config(config_file)).unlink()
    result = CliRunner().invoke(app, ["stats", "--config", str(config_file)])
    assert result.exit_code == 1
    assert isinstance(result.exception, UserError)
    assert "no corpus statistics" in str(result.exception)


def test_count_term_counts_chunks_per_venue(config_file):
    result = CliRunner().invoke(app, ["count-term", "the", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert 'chunks containing "the"' in result.output and "total" in result.output


class _UnreachableStore:
    def __init__(self, cfg) -> None:
        raise ConnectionError("no database here")


def test_check_reports_an_unreachable_database_and_a_consistent_index(config_file, monkeypatch):
    import crossvenue_rag.index.store as store

    monkeypatch.setattr(store, "ChunkStore", _UnreachableStore)
    result = CliRunner().invoke(app, ["check", "--config", str(config_file)])
    out = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "pgvector unavailable no database here" in out
    assert "index consistent" in out and "(config, manifest and index_meta.json)" in out


def test_check_reports_an_inconsistent_index_instead_of_raising(config_file, monkeypatch):
    import crossvenue_rag.index.store as store
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.index_version import index_meta_path

    index_meta_path(load_config(config_file)).unlink()
    monkeypatch.setattr(store, "ChunkStore", _UnreachableStore)
    result = CliRunner().invoke(app, ["check", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "index        inconsistent" in result.output


def test_check_includes_postgres_when_the_store_holds_the_version(config_file, monkeypatch):
    import contextlib

    import crossvenue_rag.index.store as store
    from tests.cli.conftest import current_version

    version = current_version(config_file)

    class FakeStore:
        def __init__(self, cfg) -> None:
            pass

        def connect_read_only(self):
            return contextlib.nullcontext(object())

        def stored_index_versions(self, conn):
            return [version]

        def count(self, conn):
            return 5

        def stored_models(self, conn):
            return ["m"]

    monkeypatch.setattr(store, "ChunkStore", FakeStore)
    result = CliRunner().invoke(app, ["check", "--config", str(config_file)])
    out = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert "5 rows" in out and "index_meta.json and Postgres" in out


# Two ways square brackets break rich markup: a closing tag with no opening raises,
# a known style tag disappears. As path segments, "[" / "x]" spells "[/x]".
BRACKETED_DIRS = (("[", "x]"), ("[bold]",))


def _bracketed(tmp_path, parts) -> str:
    folder = tmp_path.joinpath(*parts)
    folder.mkdir(parents=True)
    return str(folder)


@pytest.mark.parametrize("written", [False, True])
@pytest.mark.parametrize("parts", BRACKETED_DIRS)
def test_fetch_prints_a_bracketed_manifest_path_as_written(
    written, parts, config_file, tmp_path, monkeypatch
):
    manifest = f"{_bracketed(tmp_path, parts)}/manifest.csv"
    monkeypatch.setattr(_under_test, "fetch_all", _fake_fetch(written=written))
    args = ["fetch", "--config", str(config_file), "--set", f"paths.manifest={manifest}"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert manifest in "".join(result.output.split("\n"))


@pytest.mark.parametrize("parts", BRACKETED_DIRS)
def test_check_prints_a_bracketed_config_path_as_written(parts, config_file, tmp_path, monkeypatch):
    import shutil

    config = f"{_bracketed(tmp_path, parts)}/config.yaml"
    shutil.copy(config_file, config)
    monkeypatch.setattr("crossvenue_rag.index.store.ChunkStore", _UnreachableStore)
    result = CliRunner().invoke(app, ["check", "--config", config])
    assert result.exit_code == 0, result.output
    assert f"config ok: {config}" in "".join(result.output.split("\n"))


@pytest.mark.parametrize("term", ["[/x]", "[bold]"])
def test_count_term_prints_a_bracketed_term_as_typed(term, config_file):
    result = CliRunner().invoke(app, ["count-term", term, "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert f'"{term}"' in result.output


def test_check_reports_a_damaged_index_meta_as_a_line(config_file, monkeypatch):
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.index_version import index_meta_path

    index_meta_path(load_config(config_file)).write_text("{", encoding="utf-8")
    monkeypatch.setattr("crossvenue_rag.index.store.ChunkStore", _UnreachableStore)
    result = CliRunner().invoke(app, ["check", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    assert "index        inconsistent  invalid" in result.output
