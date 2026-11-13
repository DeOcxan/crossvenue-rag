"""Writing a text file atomically, so a crash mid-write leaves the last complete file in place."""

from __future__ import annotations

from pathlib import Path


def write_text_atomically(path: Path, text: str) -> None:
    """Writes `text` to `path` as UTF-8 through a `.tmp` file beside it, creating the folder.

    The `.tmp` file replaces `path` only when it is complete, so an interrupt or a crash
    mid-write leaves the last complete file in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
