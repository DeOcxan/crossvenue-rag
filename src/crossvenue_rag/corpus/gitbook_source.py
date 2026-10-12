"""GitBook documentation sources: the per-page markdown a space's `llms.txt` lists.

The markdown is what the site renders, so no HTML cleanup is needed; requests
are serialized with a configurable delay and retried with backoff.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import formatting
from ..clock import utc_now
from ..config import Config
from ..confined_paths import confined
from ..errors import UserError
from ..hashes import text_sha256
from ..retries import NETWORK_ERROR, Attempt, Backoff, with_retries
from ..run_files import file_sha256
from .manifest import ManifestRow
from .sources import SourceSpec, venue_directory

log = logging.getLogger(__name__)

# GitBook's llms.txt is a markdown list of `- [Title](https://.../page.md)` links,
# grouped under `## <Language>` headings.
_LINK_RE = re.compile(r"^\s*-\s*\[(?P<title>[^\]]*)\]\((?P<url>https?://[^)\s]+)\)", re.MULTILINE)


_SECTION_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)


def parse_llms_index(text: str, section: str | None) -> list[tuple[str, str]]:
    """The ``(title, url)`` pairs of a GitBook llms.txt.

    When ``section`` is given, only links under that ``## <section>`` heading are
    returned: PancakeSwap lists fifteen translations of the same pages, and a
    corpus of the same text in fifteen languages measures the wrong thing.

    Raises:
        UserError: `section` is given and the index has no such heading.
    """
    if section:
        bounds = [(m.group("name").strip(), m.start(), m.end()) for m in _SECTION_RE.finditer(text)]
        chosen: str | None = None
        for index, (name, _start, end) in enumerate(bounds):
            if name.lower() == section.lower():
                stop = bounds[index + 1][1] if index + 1 < len(bounds) else len(text)
                chosen = text[end:stop]
                break
        if chosen is None:
            raise UserError(
                f"section {section!r} not found in llms.txt",
                hint="correct the source's `llms_section` in the sources file (`paths.sources`)",
            )
        text = chosen

    seen: set[str] = set()
    links: list[tuple[str, str]] = []
    for match in _LINK_RE.finditer(text):
        url = match.group("url")
        if not url.endswith(".md") or url in seen:
            continue
        seen.add(url)
        links.append((match.group("title").strip(), url))
    return links


def url_to_local_path(url: str) -> str:
    """The filesystem path under the venue directory that a page URL maps onto.

    ``https://host/space/a/b.md`` -> ``a/b.md``. The space prefix is dropped so
    every document's path starts at the first real section. A ``..`` segment is
    kept, so the result can lead outside the venue directory: the writer refuses
    such a path (`confined`).
    """
    without_scheme = url.split("://", 1)[-1]
    parts = [p for p in without_scheme.split("/")[1:] if p]  # the host is not part of the path
    if len(parts) > 1:
        parts = parts[1:]  # the first part is the GitBook space slug
    safe = [re.sub(r"[^A-Za-z0-9._-]", "_", p) for p in parts]
    return "/".join(safe) or "index.md"


@dataclass(frozen=True)
class Http:
    """The HTTP client the pages are downloaded with, and the function that waits so many seconds.

    The waits are the delay between two requests and the backoff before a retry.
    """

    client: httpx.Client
    sleep: Callable[[float], None] = time.sleep


def fetch_gitbook_source(
    spec: SourceSpec,
    raw_root: Path,
    cfg: Config,
    http: Http | None,
    failures: list[str],
    recorded: dict[str, str] | None,
) -> list[ManifestRow]:
    """Rows for every page `llms.txt` lists.

    A page that cannot be downloaded is left out and its URL appended to
    `failures`. `http` is a client built from `cfg.fetch` that waits with `time.sleep`
    unless given. `recorded` maps a URL to the sha256 the manifest holds for it. A
    local file that still has that hash is kept when the page has changed
    upstream, so a drifted page never overwrites the recorded copy; the row
    carries the upstream hash, which the caller reports.

    Raises:
        UserError: The index cannot be downloaded, or `spec.venue` names no directory under
            `raw_root` (`venue_directory`).
    """
    dest = venue_directory(raw_root, spec)
    dest.mkdir(parents=True, exist_ok=True)

    owns_client = http is None
    http = http or Http(
        httpx.Client(
            headers={"User-Agent": cfg.fetch.user_agent},
            timeout=cfg.fetch.timeout_seconds,
            follow_redirects=True,
        )
    )
    try:
        try:
            index_text = _get_with_retries(http, spec.llms_index or "", cfg).text
        except httpx.HTTPError as exc:
            # Unlike a page, the index cannot be left out: it lists every page.
            raise UserError(
                f"{spec.venue}: could not download {spec.llms_index} ({exc})",
                hint="check the network, then fetch again",
            ) from exc
        links = parse_llms_index(index_text, spec.llms_section)
        log.info("%s: llms.txt lists %s", spec.venue, formatting.plural(len(links), "page"))
        # Every link is checked before the first page is written.
        targets = [(url, _page_target(spec, dest, url)) for _title, url in links]

        rows: list[ManifestRow] = []
        now = utc_now()
        for position, (url, target) in enumerate(targets, start=1):
            if position > 1:
                http.sleep(cfg.fetch.request_delay_seconds)
            try:
                body = _get_with_retries(http, url, cfg).text
            except httpx.HTTPError as exc:
                log.warning("%s: could not download %s (%s)", spec.venue, url, exc)
                failures.append(url)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            body_sha = text_sha256(body)
            keep = (
                recorded is not None
                and url in recorded
                and body_sha != recorded[url]
                and target.exists()
                and file_sha256(target) == recorded[url]
            )
            if not keep:
                target.write_text(body, encoding="utf-8")
            rows.append(
                ManifestRow(
                    venue=spec.venue,
                    venue_type=spec.venue_type,
                    source_url=url,
                    local_path=target.relative_to(raw_root).as_posix(),
                    fetched_at=now,
                    # No commit to pin to; the content hash is the only anchor.
                    commit_sha="",
                    sha256=body_sha,
                )
            )
        return rows
    finally:
        if owns_client:
            http.client.close()


def _page_target(spec: SourceSpec, dest: Path, url: str) -> Path:
    """Where the page at `url` is written: `url_to_local_path` under `dest`, or a UserError."""
    return confined(
        dest,
        url_to_local_path(url),
        what=f"the page path of {url}",
        hint=f"the llms.txt at {spec.llms_index} links a page outside its space; "
        "fetch the other venues with --venue until that index is fixed",
    )


def _get_with_retries(http: Http, url: str, cfg: Config) -> httpx.Response:
    """GET, retrying what `retries.is_retryable` allows, with backoff from `fetch.backoff_seconds`.

    Raises:
        httpx.HTTPError: The last attempt failed with a network error.
    """

    def attempt() -> Attempt[httpx.Response | httpx.TransportError]:
        try:
            response = http.client.get(url)
        except httpx.TransportError as exc:
            return Attempt(exc, NETWORK_ERROR)
        return Attempt(response, response.status_code, response.headers.get("retry-after"))

    backoff = Backoff(cfg.fetch.max_retries, cfg.fetch.backoff_seconds, http.sleep)
    last, _ = with_retries(attempt, backoff)
    if isinstance(last.result, httpx.TransportError):
        raise last.result
    last.result.raise_for_status()
    return last.result
