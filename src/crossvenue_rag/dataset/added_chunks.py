"""The added-chunks file: per question, the chunk ids `--relabel` shows beside its recorded pool.

`crossvenue label --relabel --add-chunks` reads it; `evals/selection/d23_answer_chunks.jsonl`,
the answer chunks of deviation D23, is one.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .. import jsonl


class _AddedChunks(BaseModel):
    """One line of the file: a question id and the chunk ids to show beside its pool."""

    id: str
    chunk_ids: list[str]


def read_added_chunks(path: str | Path) -> dict[str, list[str]]:
    """Question id -> chunk ids to show beside its recorded pool, from a JSON Lines file.

    One object per line: `{"id": "cv-0115", "chunk_ids": ["...", "..."]}`.

    Raises:
        UserError: Nothing is at `path`, or a directory is. A line is not UTF-8, not JSON, or
            not an object with a string `id` and a list of string `chunk_ids`.
    """
    records = jsonl.iter_models(
        path,
        _AddedChunks,
        hint='write one question per line: {"id": "cv-0115", "chunk_ids": ["..."]}',
    )
    return {record.id: record.chunk_ids for _, record in records}
