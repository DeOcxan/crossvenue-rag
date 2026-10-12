"""What results/runs.json records of the committed runs the documents cite."""

from __future__ import annotations

from collections.abc import Sequence

from ...run_files import RUNS_REGISTRY, read_registry
from ..sources import Sources
from .corpus import index_version


def run_index_version(src: Sources, run: str) -> str | None:
    """The `index_version` results/runs.json records for `run`; None if it records none."""
    entry = read_registry(src.path(RUNS_REGISTRY)).runs.get(run)
    return None if entry is None else entry.index_version


def runs_on_index(src: Sources, runs: Sequence[str]) -> list[tuple[str, bool]]:
    """Each of `runs` with whether results/runs.json records it against the built index."""
    current = index_version(src)
    return [(run, run_index_version(src, run) == current) for run in runs]


def runs_on_current_index(src: Sources, runs: Sequence[str]) -> bool:
    """Whether results/runs.json records every one of `runs` against the built index."""
    return all(on_index for _, on_index in runs_on_index(src, runs))
