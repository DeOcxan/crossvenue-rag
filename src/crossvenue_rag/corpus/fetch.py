"""Corpus acquisition: every selected source into data/raw, compared with manifest.csv.

`fetch_all` chooses the sources, downloads each through its kind's module
(`git_source`, `gitbook_source`), and compares the result with the manifest or
rewrites it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .. import formatting, listing
from ..config import Config
from ..confined_paths import file_on_path
from ..errors import UserError
from ..run_files import file_sha256
from .git_source import fetch_git_source
from .gitbook_source import Http, fetch_gitbook_source
from .manifest import (
    ManifestRow,
    preserve_timestamps,
    read_manifest,
    write_manifest,
)
from .sources import SourceSpec, load_sources

log = logging.getLogger(__name__)


@dataclass
class FetchReport:
    """What a fetch found, per venue.

    Files matching the manifest, files whose content differs, manifest rows with
    no file, files the manifest lacks, and pages that could not be downloaded.
    """

    rows: list[ManifestRow]
    matched: dict[str, int] = field(default_factory=dict)
    changed: dict[str, list[str]] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)
    extra: dict[str, list[str]] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)
    manifest_written: bool = False

    @property
    def differs(self) -> bool:
        """True when any venue has a changed, missing or extra file."""
        return any(self.changed.values()) or any(self.missing.values()) or any(self.extra.values())


def _compare(
    venue: str,
    fresh: list[ManifestRow],
    recorded: list[ManifestRow],
    report: FetchReport,
    raw_root: Path,
) -> None:
    """Fills `report` for one venue.

    A recorded file that this fetch did not download (a failed GitBook page) but
    that is still on disk with its recorded hash counts as matched.
    """
    by_path = {row.local_path: row for row in recorded}
    seen = {row.local_path: row for row in fresh}
    kept = {
        path
        for path, row in by_path.items()
        if path not in seen
        and (raw_root / path).is_file()
        and file_sha256(raw_root / path) == row.sha256
    }
    report.matched[venue] = len(kept) + sum(
        1 for path, row in seen.items() if path in by_path and by_path[path].sha256 == row.sha256
    )
    report.changed[venue] = sorted(
        path for path, row in seen.items() if path in by_path and by_path[path].sha256 != row.sha256
    )
    report.missing[venue] = sorted(set(by_path) - set(seen) - kept)
    report.extra[venue] = sorted(set(seen) - set(by_path))


def _require_raw_directory(raw: Path) -> None:
    """`paths.raw` is a directory or can be made one: no file stands on its path.

    Raises:
        UserError: `raw` or one of its parents is a file, so the sources cannot be written.
    """
    blocking = file_on_path(raw)
    if blocking is not None:
        raise UserError(
            f"paths.raw is {raw}, and {blocking} is a file; the fetched sources need a directory",
            hint="set paths.raw to a directory (--set paths.raw=<dir>) or move the file",
        )


def fetch_all(
    cfg: Config,
    only: list[str] | None,
    *,
    follow_branch: bool,
    allow_missing: bool,
    http: Http | None = None,
) -> FetchReport:
    """Fetches every source (`only` None), or the sources of the `only` venues.

    Without `follow_branch`, git sources are checked out at the commit the manifest
    records and every file is compared with its recorded sha256;
    GitBook pages, which have no commit, are compared the same way. The
    manifest is left as it is, and a difference raises `UserError` after all
    sources are fetched, because a changed file under an unchanged manifest
    would change chunk ids without changing `index_version`.

    With `follow_branch`, sources are fetched at the head of the branch
    `sources.yaml` names and the manifest is rewritten; a rerun whose content
    hashes and commits are unchanged rewrites it byte for byte, including `fetched_at`.
    A page that cannot be downloaded raises unless `allow_missing`. `http` downloads
    the GitBook pages (`gitbook_source.fetch_gitbook_source`).

    Raises:
        UserError: A venue of `only` is unknown.
    """
    specs = _selected_sources(cfg, only)
    _require_raw_directory(cfg.paths.raw)
    cfg.paths.raw.mkdir(parents=True, exist_ok=True)
    existing = read_manifest(cfg.paths.manifest)
    if not follow_branch:
        _require_manifest(cfg, existing)
    report = FetchReport(rows=[])
    recorded = None if follow_branch else _by_venue(existing)
    fresh = _download(cfg, specs, recorded, http, report)
    if not allow_missing:
        _require_every_page(report.failed)
    touched = {s.venue for s in specs}
    if follow_branch:
        # A partial fetch must not drop the rows of venues it did not fetch.
        kept = _other_venues(existing, touched) if only else []
        _rewrite_manifest(cfg, report, fresh + kept, existing)
        return report
    report.rows = [row for row in existing if row.venue in touched]
    _require_no_difference(report, cfg.paths.manifest)
    return report


def _download(
    cfg: Config,
    specs: list[SourceSpec],
    recorded: dict[str, list[ManifestRow]] | None,
    http: Http | None,
    report: FetchReport,
) -> list[ManifestRow]:
    """Downloads every source and returns the fresh manifest rows.

    Given `recorded` (the manifest's rows by venue), each source is pinned to
    its rows and compared with them into `report`.
    """
    fresh: list[ManifestRow] = []
    for spec in specs:
        log.info("fetching %s (%s, %s)", spec.venue, spec.venue_type, spec.kind)
        pinned = None if recorded is None else recorded.get(spec.venue, [])
        rows = _fetch_source(spec, cfg, pinned, http, report.failed)
        if pinned is not None:
            _compare(spec.venue, rows, pinned, report, cfg.paths.raw)
        fresh += rows
    return fresh


def _require_manifest(cfg: Config, existing: list[ManifestRow]) -> None:
    if not existing:
        raise UserError(
            f"{cfg.paths.manifest} is empty or missing, so there is nothing to fetch from",
            hint="run `crossvenue fetch --follow-branch` to fetch the branch heads and write it",
        )


def _require_every_page(failed: list[str]) -> None:
    if failed:
        raise UserError(
            f"{formatting.plural(len(failed), 'page')} could not be downloaded: "
            f"{listing.first_items(failed)}",
            hint="run again later, or pass --allow-missing to continue without them",
        )


def _selected_sources(cfg: Config, only: list[str] | None) -> list[SourceSpec]:
    """The sources in `data/sources.yaml`, or those of the `only` venues."""
    specs = load_sources(cfg.paths.sources)
    if not only:
        return specs
    unknown = set(only) - {s.venue for s in specs}
    if unknown:
        raise UserError(
            f"unknown venues: {', '.join(sorted(unknown))}",
            hint=f"pass --venue one of the venues in {cfg.paths.sources}: "
            f"{', '.join(s.venue for s in specs)}",
        )
    return [s for s in specs if s.venue in only]


def _fetch_source(
    spec: SourceSpec,
    cfg: Config,
    pinned: list[ManifestRow] | None,
    http: Http | None,
    failed: list[str],
) -> list[ManifestRow]:
    """Downloads one source and returns its manifest rows.

    The source is fetched at its recorded commit or hashes when `pinned` is the
    manifest's rows for it, and at the head of its branch when `pinned` is `None`.
    """
    if spec.kind == "git":
        return fetch_git_source(spec, cfg.paths.raw, _recorded_commit(spec, pinned))
    hashes = None if pinned is None else {row.source_url: row.sha256 for row in pinned}
    return fetch_gitbook_source(spec, cfg.paths.raw, cfg, http, failed, hashes)


def _recorded_commit(spec: SourceSpec, pinned: list[ManifestRow] | None) -> str | None:
    """The one commit the manifest records for a git source; `None` follows the branch."""
    if pinned is None:
        return None
    shas = {row.commit_sha for row in pinned if row.commit_sha}
    if len(shas) != 1:
        raise UserError(
            f"{spec.venue}: the manifest records {formatting.plural(len(shas), 'commit')}, "
            "expected one",
            hint="fetch it with `--follow-branch` to record a commit",
        )
    return shas.pop()


def _other_venues(rows: list[ManifestRow], venues: set[str]) -> list[ManifestRow]:
    return [row for row in rows if row.venue not in venues]


def _by_venue(rows: list[ManifestRow]) -> dict[str, list[ManifestRow]]:
    grouped: dict[str, list[ManifestRow]] = {}
    for row in rows:
        grouped.setdefault(row.venue, []).append(row)
    return grouped


def _rewrite_manifest(
    cfg: Config, report: FetchReport, rows: list[ManifestRow], existing: list[ManifestRow]
) -> None:
    """Writes `rows` as the manifest, keeping the timestamps of unchanged files."""
    report.rows = preserve_timestamps(rows, existing)
    write_manifest(cfg.paths.manifest, report.rows)
    report.manifest_written = True


def _require_no_difference(report: FetchReport, manifest: Path) -> None:
    """Refuses a fetch whose files differ from the manifest it was pinned to."""
    if not report.differs:
        return
    details = [
        f"{venue}: {formatting.count(len(report.changed[venue]))} changed, "
        f"{formatting.count(len(report.missing[venue]))} missing, "
        f"{formatting.count(len(report.extra[venue]))} not in the manifest"
        for venue in report.changed
        if report.changed[venue] or report.missing[venue] or report.extra[venue]
    ]
    raise UserError(
        f"the fetched sources differ from {manifest}: " + "; ".join(details),
        hint="the corpus cannot be rebuilt as recorded; `--follow-branch` fetches the "
        "current sources and rewrites the manifest, which changes index_version",
    )
