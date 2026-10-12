"""Shared helpers for tests/corpus/."""

from __future__ import annotations

from typing import Any

from crossvenue_rag.corpus.sources import SourceSpec


def spec(**metadata: Any) -> SourceSpec:
    return SourceSpec(
        venue="bybit",
        venue_type="cex",
        kind="git",
        repo="https://example.invalid/repo",
        license="MIT",
        metadata=metadata,
    )
