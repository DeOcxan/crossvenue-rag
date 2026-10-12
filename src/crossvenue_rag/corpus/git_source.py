"""Fetching a git documentation source by a blobless shallow sparse clone, a manifest row per file.

A source is checked out at the commit the manifest records, or at the head of
its branch when the manifest is being rewritten; its include and exclude globs
choose the files.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path

from .. import formatting
from ..clock import utc_now
from ..errors import UserError
from ..run_files import file_sha256
from .manifest import ManifestRow
from .sources import SourceSpec, venue_directory

log = logging.getLogger(__name__)


def _glob_match(rel_path: str, pattern: str) -> bool:
    """fnmatch, with `**/` also matching zero directories.

    fnmatch's `*` crosses separators, so `docs/**/*.md` matches nested paths,
    but `**/*_CN.md` would not match the top-level `rest-api_CN.md` of the
    Binance repo. Matching the prefix-stripped pattern too makes `**/` match
    zero directories as well.
    """
    if fnmatch(rel_path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch(rel_path, pattern.removeprefix("**/"))


def matches(rel_path: str, include: list[str], exclude: list[str]) -> bool:
    """Glob filter over a source-relative path.

    An empty `include` means "everything not excluded"; `exclude` always wins.
    """
    if any(_glob_match(rel_path, pattern) for pattern in exclude):
        return False
    if not include:
        return True
    return any(_glob_match(rel_path, pattern) for pattern in include)


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    """The output of a git command; UserError naming the command and git's last message.

    A clone or fetch fails on a network or a repository the user can fix, as a page
    download does in `gitbook_source`.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        said = (exc.stderr or "").strip().splitlines()
        raise UserError(
            f"git {args[0]} failed with exit status {formatting.own_form(exc.returncode)}"
            + (f": {said[-1]}" if said else ""),
            hint="check the network and the source's `repo` in the sources file "
            "(`paths.sources`), then fetch again",
        ) from exc
    return result.stdout.strip()


def clone_or_update(spec: SourceSpec, dest: Path) -> str:
    """Shallow, blobless, sparse clone (or fast-forward) of a docs repo.

    Returns the checked-out commit SHA. ``--filter=blob:none`` matters for ccxt,
    whose repository holds its generated SDKs; with a sparse-checkout of
    ``wiki`` only the wiki blobs are ever downloaded.
    """
    if (dest / ".git").exists():
        _run_git(["fetch", "--depth", "1", "origin", spec.ref], cwd=dest)
        _run_git(["checkout", "--force", "FETCH_HEAD"], cwd=dest)
    else:
        _fresh_clone(spec, dest, ["--branch", spec.ref])

    return _run_git(["rev-parse", "HEAD"], cwd=dest)


def checkout_commit(spec: SourceSpec, dest: Path, sha: str) -> str:
    """Blobless, sparse checkout of one recorded commit (not the branch head).

    A fresh clone starts without a checkout, then fetches exactly `sha`; the
    hosts in `sources.yaml` serve a commit by id to a shallow fetch.
    """
    if not (dest / ".git").exists():
        _fresh_clone(spec, dest, ["--no-checkout"])
    _run_git(["fetch", "--depth", "1", "origin", sha], cwd=dest)
    _run_git(["checkout", "--force", sha], cwd=dest)
    return _run_git(["rev-parse", "HEAD"], cwd=dest)


def _fresh_clone(spec: SourceSpec, dest: Path, mode: list[str]) -> None:
    """A shallow, blobless clone into an emptied `dest`, sparse when the spec names paths.

    `mode` is what differs between the callers: `--branch <ref>` or `--no-checkout`.
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--depth", "1", "--filter=blob:none", *mode]
    if spec.sparse_paths:
        args.append("--sparse")
    _run_git([*args, spec.repo or "", str(dest)])
    if spec.sparse_paths:
        _run_git(["sparse-checkout", "set", "--no-cone", *spec.sparse_paths], cwd=dest)


def _git_blob_url(repo: str, sha: str, rel_path: str) -> str:
    return f"{repo.rstrip('/')}/blob/{sha}/{rel_path}"


def fetch_git_source(spec: SourceSpec, raw_root: Path, commit: str | None) -> list[ManifestRow]:
    """Rows for every matching file of a git source.

    The checkout is at `commit` when given, else at the head of the branch the
    source names.

    Raises:
        UserError: `spec.venue` names no directory under `raw_root` (`venue_directory`).
    """
    dest = venue_directory(raw_root, spec)
    sha = checkout_commit(spec, dest, commit) if commit else clone_or_update(spec, dest)
    log.info("%s: checked out %s", spec.venue, sha)

    rows: list[ManifestRow] = []
    now = utc_now()
    for path in sorted(dest.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(dest).as_posix()
        if not matches(rel, spec.include, spec.exclude):
            continue
        rows.append(
            ManifestRow(
                venue=spec.venue,
                venue_type=spec.venue_type,
                source_url=_git_blob_url(spec.repo or "", sha, rel),
                local_path=path.relative_to(raw_root).as_posix(),
                fetched_at=now,
                commit_sha=sha,
                sha256=file_sha256(path),
            )
        )
    return rows
