"""Tests for `crossvenue_rag.jsonl`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from crossvenue_rag import jsonl
from crossvenue_rag.errors import UserError


def test_a_written_file_reads_back_and_keeps_non_ascii_text(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows: list[dict[str, Any]] = [
        {"id": "a", "text": "\u0423\u0440 \u2212 0.5"},
        {"id": "b", "n": 2},
    ]
    jsonl.write(path, rows)
    assert path.read_text(encoding="utf-8") == (
        '{"id": "a", "text": "\u0423\u0440 \u2212 0.5"}\n{"id": "b", "n": 2}\n'
    )
    assert jsonl.read(path) == rows


def test_ascii_only_escapes_non_ascii_characters(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    jsonl.write(path, [{"minus": "\u2212"}], ascii_only=True)
    assert path.read_text(encoding="utf-8") == '{"minus": "\\u2212"}\n'


def test_append_adds_lines_and_creates_the_file(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    jsonl.append(path, [{"i": 1}])
    jsonl.append(path, [{"i": 2}, {"i": 3}])
    assert [r["i"] for r in jsonl.read(path)] == [1, 2, 3]


def test_blank_lines_are_skipped_and_numbered_rows_keep_file_line_numbers(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n  \n{"a": 2}\n', encoding="utf-8")
    assert jsonl.read(path) == [{"a": 1}, {"a": 2}]
    assert list(jsonl.iter_rows(path)) == [(1, {"a": 1}), (4, {"a": 2})]


def test_a_malformed_line_names_its_file_and_line(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(UserError, match=r"rows\.jsonl:2: not JSON"):
        list(jsonl.iter_rows(path))


def test_a_malformed_line_is_named_by_file_and_line(tmp_path):

    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": \n', encoding="utf-8")
    with pytest.raises(UserError, match=r"rows\.jsonl:3: not JSON"):
        jsonl.read(path)


def test_a_byte_that_is_not_utf8_is_named_on_its_own_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"a": 1}\n{"a": 2}\n{"a": "\xff"}\n{"a": 4}\n')
    with pytest.raises(UserError, match=r"rows\.jsonl:3: not UTF-8 text"):
        jsonl.read(path)
