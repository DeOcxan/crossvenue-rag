"""Shared helpers for tests/corpus/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from crossvenue_rag.corpus.sources import SourceSpec

# Each venue the sources file refuses, and a symlink it does not check, as `fetch` would join them;
# ABSOLUTE stands for the absolute path of data/outside.
UNSAFE_VENUES = ["", "   ", "..", "a/../..", "ABSOLUTE", "linked"]


def spec(**metadata: Any) -> SourceSpec:
    return SourceSpec(
        venue="bybit",
        venue_type="cex",
        kind="git",
        repo="https://example.invalid/repo",
        license="MIT",
        metadata=metadata,
    )


def raw_with_a_sibling(tmp_path: Path) -> tuple[Path, Path]:
    """data/raw holding another venue's file, next to data/outside holding a file."""
    raw, outside = tmp_path / "data" / "raw", tmp_path / "data" / "outside"
    (raw / "binance").mkdir(parents=True)
    (raw / "binance" / "keep.md").write_text("binance", encoding="utf-8")
    outside.mkdir()
    (outside / "keep.md").write_text("outside", encoding="utf-8")
    (raw / "linked").symlink_to(outside, target_is_directory=True)
    return raw, outside
