"""What `crossvenue check` reports: the sources, the built artifacts and the index's agreement.

Every part is read, never written, so the report is safe against any database.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..corpus.artifacts import chunks_path
from ..corpus.sources import load_sources
from ..errors import UserError
from ..index_version import INDEX_META_FILENAME, require_current_index
from .bm25_index import INDEX_DIRNAME
from .store import StoreState, store_state


@dataclass(frozen=True)
class Artifact:
    """One built artifact: its name, its path and whether it exists."""

    name: str
    path: Path
    present: bool


@dataclass(frozen=True)
class BuildStatus:
    """The state `check` reports.

    `store` is the Postgres state, or why it cannot be read. `index_version` is set when
    the config, manifest, index_meta.json and any Postgres rows agree; `inconsistency`
    says why they do not otherwise.
    """

    venues: int
    artifacts: list[Artifact]
    store: StoreState | str
    index_version: str | None
    inconsistency: str | None

    @property
    def agreeing(self) -> str:
        """What agrees on `index_version`: Postgres too when its rows carry a version."""
        if isinstance(self.store, StoreState) and self.store.index_versions:
            return f"config, manifest, {INDEX_META_FILENAME} and Postgres"
        return f"config, manifest and {INDEX_META_FILENAME}"


def build_status(cfg: Config) -> BuildStatus:
    """The sources, the artifacts, the Postgres state and whether the index is consistent."""
    venues = len(load_sources(cfg.paths.sources))
    artifacts = [
        Artifact(name, path, path.exists())
        for name, path in (
            ("manifest", cfg.paths.manifest),
            ("chunks", chunks_path(cfg)),
            ("bm25 index", cfg.paths.index / INDEX_DIRNAME),
        )
    ]
    store = store_state(cfg)
    versions = store.index_versions if isinstance(store, StoreState) else None
    try:
        return BuildStatus(venues, artifacts, store, require_current_index(cfg, versions), None)
    except UserError as exc:
        return BuildStatus(venues, artifacts, store, None, exc.with_hint())
