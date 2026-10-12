"""Tests for `crossvenue_rag.dataset.added_chunks`: the file `--add-chunks` reads."""

from __future__ import annotations

import pytest

from crossvenue_rag.dataset.added_chunks import read_added_chunks
from crossvenue_rag.errors import UserError


def test_added_chunks_are_read_per_question(tmp_path):
    listing = tmp_path / "added.jsonl"
    listing.write_text(
        '{"id": "q1", "chunk_ids": ["x1", "x2"]}\n\n{"id": "q2", "chunk_ids": []}\n',
        encoding="utf-8",
    )
    assert read_added_chunks(listing) == {"q1": ["x1", "x2"], "q2": []}
    for bad, problem in (
        ('{"id": "q1"}', ":1: chunk_ids"),
        ("not json", ":1: "),
        ('{"chunk_ids": []}', ":1: id"),
        ('{"id": "q1", "chunk_ids": [1]}', ":1: chunk_ids"),
        ('{"id": 1, "chunk_ids": []}', ":1: id"),
    ):
        listing.write_text(bad + "\n", encoding="utf-8")
        with pytest.raises(UserError, match=problem) as refused:
            read_added_chunks(listing)
        assert refused.value.hint
    with pytest.raises(UserError, match="not found"):
        read_added_chunks(tmp_path / "missing.jsonl")
