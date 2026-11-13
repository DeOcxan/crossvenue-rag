"""Tests for `crossvenue_rag.atomic_files`: a text file written through a `.tmp` file."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.atomic_files import write_text_atomically


def test_a_file_is_written_into_a_folder_it_creates_and_no_tmp_file_is_left(tmp_path: Path):
    path = tmp_path / "annotations" / "q.rounds.json"
    write_text_atomically(path, "{}\n")
    assert path.read_text(encoding="utf-8") == "{}\n"
    assert list(path.parent.iterdir()) == [path]


def test_a_failed_write_leaves_the_last_complete_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "entry.json"
    write_text_atomically(path, "complete")

    def fail(self: Path, *_: object, **__: object) -> None:
        self.write_bytes(b"cut sh")
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomically(path, "replacement")
    assert path.read_text(encoding="utf-8") == "complete"
