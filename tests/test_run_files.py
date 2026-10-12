"""Tests for `run_files`: the index.json and CSV writers every run kind shares."""

from __future__ import annotations

import hashlib
import json

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.run_files import (
    INDEX_FILE,
    RUNS_REGISTRY,
    RunIndex,
    index_json,
    question_fields,
    read_index,
    read_registry,
    registry_json,
    write_csv,
    write_index,
)
from tests.conftest import REPO, committed_with


def test_an_index_is_read_as_written_with_or_without_a_final_newline(tmp_path):
    for final_newline in (True, False):
        write_index(tmp_path, {"index_version": "v", "k": 20}, final_newline=final_newline)
        assert read_index(tmp_path) == RunIndex(index_version="v", k=20)
        assert (tmp_path / "index.json").read_text().endswith("\n") is final_newline


def test_a_run_without_an_index_reads_as_none(tmp_path):
    assert read_index(tmp_path) is None


def test_csv_tables_use_crlf_and_minimal_quoting(tmp_path):
    write_csv(tmp_path / "t.csv", ["a", "b"], [["x, y", None]])
    assert (tmp_path / "t.csv").read_bytes() == b'a,b\r\n"x, y",\r\n'


def test_a_saved_config_is_read_from_the_file_alone(tmp_path, monkeypatch):
    from crossvenue_rag.run_files import read_run_config

    (tmp_path / "config.yaml").write_text("retrieval:\n  k: 7\n", encoding="utf-8")
    monkeypatch.setenv("RETRIEVAL", '{"k": 9}')
    config = read_run_config(tmp_path)
    assert config.retrieval.k == 7
    assert config.model_fields_set == {"retrieval"}


@pytest.mark.parametrize("run", committed_with(INDEX_FILE), ids=lambda p: p.name)
def test_a_committed_index_is_written_back_with_the_same_bytes(run):
    original = (run / INDEX_FILE).read_text(encoding="utf-8")
    index = read_index(run)
    assert index is not None
    assert index_json(index, final_newline=original.endswith("\n")) == original


def test_the_committed_registry_is_written_back_with_the_same_bytes():
    path = REPO / RUNS_REGISTRY
    assert registry_json(read_registry(path)) == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("run", committed_with("config.yaml"), ids=lambda p: p.name)
def test_every_committed_run_config_is_read_with_its_retired_keys_dropped(run):
    from crossvenue_rag.run_files import read_run_config

    assert read_run_config(run).retrieval.k > 0


def test_a_run_config_with_an_unknown_key_that_was_never_in_the_schema_is_refused(tmp_path):

    from crossvenue_rag.run_files import read_run_config

    (tmp_path / "config.yaml").write_text("eval:\n  never_a_key: 1\n", encoding="utf-8")
    with pytest.raises(UserError, match="never_a_key"):
        read_run_config(tmp_path)


def test_a_run_directory_is_created_once_and_an_existing_one_is_refused(tmp_path):
    from crossvenue_rag.errors import UserError
    from crossvenue_rag.run_files import create_run_dir, run_path

    run_dir = run_path(tmp_path, "_x")
    assert run_dir.name.endswith("_x") and not run_dir.exists()
    assert create_run_dir(run_dir).is_dir()
    with pytest.raises(UserError, match="exists already"):
        create_run_dir(run_dir)


@pytest.mark.parametrize("run", committed_with("source.json"), ids=lambda p: p.name)
def test_a_committed_source_json_is_written_back_with_the_same_bytes(run, tmp_path):
    from crossvenue_rag.run_files import read_run_source, write_run_source

    source = read_run_source(run)
    assert source is not None and source.runs()
    write_run_source(tmp_path, source)
    assert (tmp_path / "source.json").read_bytes() == (run / "source.json").read_bytes()


def test_a_run_records_its_index_and_its_question_file(tmp_path):
    questions = tmp_path / "q.jsonl"
    questions.write_text('{"id": "cv-1"}\n', encoding="utf-8")
    write_index(tmp_path, {"index_version": "v1", **question_fields(questions)}, final_newline=True)
    record = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert record == {
        "index_version": "v1",
        "questions": str(questions),
        "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
    }


def test_a_count_cell_is_a_whole_number():
    from crossvenue_rag.run_files import CsvRow

    row = CsvRow.model_validate({"calls": "12.0", "mean": "2.5", "empty": ""})
    assert row.count("calls") == 12 and isinstance(row.count("calls"), int)
    with pytest.raises(ValueError, match=r"holds 2\.5"):
        row.count("mean")
    with pytest.raises(ValueError, match="empty"):
        row.count("empty")


def test_a_count_cell_that_is_no_whole_number_is_named_through_the_formatting_module(
    monkeypatch,
):
    from crossvenue_rag.run_files import CsvRow

    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(ValueError, match=r"holds <2\.5>"):
        CsvRow.model_validate({"mean": "2.5"}).count("mean")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("method,recall@10,mrr\r\nbm25,0.5\r\n", "short.csv:2: a row of 2 cells under 3 columns"),
        ("method,recall@10\r\nbm25,0.5,0.7\r\n", "short.csv:2: a row of 3 cells under 2 columns"),
    ],
)
def test_a_ragged_row_of_a_table_is_refused_naming_its_line(tmp_path, body, message):
    from crossvenue_rag.run_files import CsvRow, read_table

    class Row(CsvRow):
        method: str

    path = tmp_path / "short.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(UserError, match=message):
        read_table(path, Row)


def test_a_column_the_row_neither_declares_nor_holds_is_a_value_error_naming_it():
    from crossvenue_rag.run_files import CsvRow

    row = CsvRow.model_validate({"mrr": "0.5"})
    for read in (row.text, row.number, row.value):
        with pytest.raises(ValueError, match="no column recall@10"):
            read("recall@10")
