"""Tests for `crossvenue_rag.corpus.fetch`."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import httpx
import pytest

from crossvenue_rag.config import Config
from crossvenue_rag.corpus.fetch import fetch_all
from crossvenue_rag.corpus.gitbook_source import Http
from crossvenue_rag.corpus.manifest import (
    ManifestRow,
    read_manifest,
    write_manifest,
)
from crossvenue_rag.errors import UserError
from crossvenue_rag.run_files import file_sha256
from tests.conftest import default_config


# Fetching from the manifest: a local upstream repository, no network.
def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    """A repository with two commits: docs/a.md at "v1", then "v2" plus docs/b.md."""
    repo = tmp_path / "upstream"
    (repo / "docs").mkdir(parents=True)
    _git("init", "-q", "-b", "main", cwd=repo)
    for key, value in (
        ("user.email", "t@example.invalid"),
        ("user.name", "t"),
        ("uploadpack.allowAnySHA1InWant", "true"),
        ("uploadpack.allowFilter", "true"),
    ):
        _git("config", key, value, cwd=repo)
    (repo / "docs/a.md").write_text("# A\n\nv1\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "one", cwd=repo)
    first = _git("rev-parse", "HEAD", cwd=repo)
    (repo / "docs/a.md").write_text("# A\n\nv2\n")
    (repo / "docs/b.md").write_text("# B\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "two", cwd=repo)
    return repo, first, _git("rev-parse", "HEAD", cwd=repo)


def _config(tmp_path: Path, sources: str) -> Config:
    cfg = default_config(tmp_path)
    cfg.paths.raw = tmp_path / "raw"
    cfg.paths.manifest = tmp_path / "manifest.csv"
    cfg.paths.sources = tmp_path / "sources.yaml"
    cfg.paths.sources.write_text(sources)
    cfg.fetch.request_delay_seconds = 0
    cfg.fetch.max_retries = 0
    return cfg


def _git_config(tmp_path: Path, repo: Path) -> Config:
    return _config(
        tmp_path,
        f"""sources:
  - venue: bybit
    venue_type: cex
    kind: git
    license: MIT
    repo: file://{repo}
    ref: main
    include: ["docs/*.md"]
""",
    )


def _pin(cfg: Config, rows: list[tuple[str, str, str]]) -> None:
    """Writes a manifest of (local_path, commit, content) rows."""
    write_manifest(
        cfg.paths.manifest,
        [
            ManifestRow(
                "bybit", "cex", f"https://x/{path}", path, "2026-01-01T00:00:00Z", commit,
                hashlib.sha256(content.encode()).hexdigest(),
            )
            for path, commit, content in rows
        ],
    )  # fmt: skip


def test_fetch_checks_out_the_recorded_commit_and_leaves_the_manifest(tmp_path, upstream):
    repo, first, _ = upstream
    cfg = _git_config(tmp_path, repo)
    _pin(cfg, [("bybit/docs/a.md", first, "# A\n\nv1\n")])
    before = cfg.paths.manifest.read_bytes()
    report = fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)
    assert (cfg.paths.raw / "bybit/docs/a.md").read_text() == "# A\n\nv1\n"
    assert not (cfg.paths.raw / "bybit/docs/b.md").exists()  # only in the later commit
    assert report.matched == {"bybit": 1} and not report.differs
    assert cfg.paths.manifest.read_bytes() == before and not report.manifest_written


def test_following_the_branch_takes_the_head_and_rewrites_the_manifest(tmp_path, upstream):
    repo, first, head = upstream
    cfg = _git_config(tmp_path, repo)
    _pin(cfg, [("bybit/docs/a.md", first, "# A\n\nv1\n")])
    report = fetch_all(cfg, only=None, follow_branch=True, allow_missing=False)
    assert report.manifest_written
    assert {r.commit_sha for r in read_manifest(cfg.paths.manifest)} == {head}
    assert [r.local_path for r in read_manifest(cfg.paths.manifest)] == [
        "bybit/docs/a.md",
        "bybit/docs/b.md",
    ]


def test_a_file_that_differs_from_its_recorded_hash_fails_the_fetch(tmp_path, upstream):
    repo, first, _ = upstream
    cfg = _git_config(tmp_path, repo)
    _pin(cfg, [("bybit/docs/a.md", first, "something else")])
    with pytest.raises(UserError, match="1 changed"):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)


def test_an_unknown_venue_is_refused_with_a_hint_listing_the_venues(tmp_path, upstream):
    repo, _, _ = upstream
    cfg = _git_config(tmp_path, repo)
    with pytest.raises(UserError, match="unknown venues: bogus") as refused:
        fetch_all(cfg, only=["bogus"], follow_branch=False, allow_missing=False)
    assert refused.value.hint == f"pass --venue one of the venues in {cfg.paths.sources}: bybit"


def test_fetch_from_the_manifest_needs_a_manifest(tmp_path, upstream):
    repo, _, _ = upstream
    with pytest.raises(UserError, match="nothing to fetch from"):
        fetch_all(_git_config(tmp_path, repo), only=None, follow_branch=False, allow_missing=False)


@pytest.mark.parametrize("blocker", ["raw", "blocker"])
def test_a_file_where_the_raw_directory_goes_is_refused_naming_paths_raw(
    tmp_path, upstream, blocker
):
    """The file is `paths.raw` itself, or a parent directory `mkdir` would need."""
    repo, _, _ = upstream
    cfg = _git_config(tmp_path, repo)
    cfg.paths.raw = tmp_path / "raw" if blocker == "raw" else tmp_path / "blocker" / "raw"
    (tmp_path / blocker).write_text("not a directory")
    with pytest.raises(UserError, match=r"paths\.raw is .*is a file") as refused:
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)
    assert str(tmp_path / blocker) in str(refused.value)


GITBOOK = """sources:
  - venue: pancakeswap
    venue_type: dex
    kind: gitbook
    license: no license found
    llms_index: https://docs.example.invalid/llms.txt
"""
INDEX = (
    "- [One](https://docs.example.invalid/space/one.md)\n"
    "- [Two](https://docs.example.invalid/space/two.md)\n"
)


def _gitbook(tmp_path: Path, pages: dict[str, tuple[int, str]]) -> tuple[Config, Http]:
    cfg = _config(tmp_path, GITBOOK)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("llms.txt"):
            return httpx.Response(200, text=INDEX)
        status, body = pages[request.url.path.rsplit("/", 1)[-1]]
        return httpx.Response(status, text=body)

    return cfg, Http(httpx.Client(transport=httpx.MockTransport(handler)), lambda _seconds: None)


def _pin_gitbook(cfg: Config, contents: dict[str, str]) -> None:
    write_manifest(
        cfg.paths.manifest,
        [
            ManifestRow(
                "pancakeswap", "dex", f"https://docs.example.invalid/space/{name}",
                f"pancakeswap/{name}", "2026-01-01T00:00:00Z", "",
                hashlib.sha256(text.encode()).hexdigest(),
            )
            for name, text in contents.items()
        ],
    )  # fmt: skip


def test_a_drifted_gitbook_page_is_reported_and_never_overwrites_the_recorded_copy(tmp_path):
    cfg, http = _gitbook(tmp_path, {"one.md": (200, "one"), "two.md": (200, "two, edited")})
    _pin_gitbook(cfg, {"one.md": "one", "two.md": "two"})
    recorded = cfg.paths.raw / "pancakeswap/two.md"
    recorded.parent.mkdir(parents=True)
    recorded.write_text("two")
    with pytest.raises(UserError, match="1 changed"):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False, http=http)
    assert recorded.read_text() == "two"
    assert (cfg.paths.raw / "pancakeswap/one.md").read_text() == "one"


def test_a_page_that_cannot_be_downloaded_fails_unless_allowed(tmp_path):
    cfg, http = _gitbook(tmp_path, {"one.md": (200, "one"), "two.md": (500, "")})
    _pin_gitbook(cfg, {"one.md": "one", "two.md": "two"})
    with pytest.raises(UserError, match="could not be downloaded"):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False, http=http)
    # Allowed, and the recorded copy is still on disk: it counts as matched.
    kept = cfg.paths.raw / "pancakeswap/two.md"
    kept.write_text("two")
    report = fetch_all(cfg, only=None, follow_branch=False, allow_missing=True, http=http)
    assert report.failed and report.matched == {"pancakeswap": 2}
    assert file_sha256(kept) == read_manifest(cfg.paths.manifest)[1].sha256


# A count the messages below format with a thousands separator.
MANY = 1000


def _many_rows(commit_of, sha: str) -> list[ManifestRow]:
    return [
        ManifestRow("bybit", "cex", f"https://x/{i}", f"bybit/docs/{i}.md", "t", commit_of(i), sha)
        for i in range(MANY)
    ]


def test_a_manifest_with_many_commits_for_a_source_is_refused_counting_them(tmp_path):
    cfg = _git_config(tmp_path, tmp_path / "unused")
    write_manifest(cfg.paths.manifest, _many_rows(lambda i: f"{i:040x}", "0" * 64))
    with pytest.raises(UserError, match="records 1,000 commits, expected one"):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)


def test_the_difference_report_formats_its_counts(tmp_path, monkeypatch):
    import crossvenue_rag.corpus.fetch as fetch

    cfg = _git_config(tmp_path, tmp_path / "unused")
    write_manifest(cfg.paths.manifest, _many_rows(lambda i: "a" * 40, "0" * 64))
    fresh = _many_rows(lambda i: "a" * 40, "1" * 64)
    monkeypatch.setattr(fetch, "fetch_git_source", lambda spec, raw, commit: fresh)
    with pytest.raises(UserError, match="bybit: 1,000 changed, 0 missing, 0 not in the manifest"):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)
    with pytest.raises(UserError, match=f"differ from {re.escape(str(cfg.paths.manifest))}: "):
        fetch_all(cfg, only=None, follow_branch=False, allow_missing=False)
