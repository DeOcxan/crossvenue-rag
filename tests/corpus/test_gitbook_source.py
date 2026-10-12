"""Tests for `crossvenue_rag.corpus.gitbook_source`."""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.gitbook_source import Http, parse_llms_index, url_to_local_path
from crossvenue_rag.corpus.sources import SourceSpec
from crossvenue_rag.errors import UserError

LLMS_TXT = """# PancakeSwap

## English

- [Product Overview](https://docs.example.invalid/welcome/readme.md)
- [How-to Guides](https://docs.example.invalid/welcome/guides.md)
- [Again](https://docs.example.invalid/welcome/readme.md)
- [Not markdown](https://docs.example.invalid/welcome/image.png)

## Français

- [Aperçu](https://docs.example.invalid/fr/apercu.md)
"""


def test_parse_llms_index_restricts_to_a_language_section():
    """PancakeSwap lists fifteen translations of the same pages, and a corpus of
    the same text in fifteen languages measures the wrong thing."""
    links = parse_llms_index(LLMS_TXT, section="English")
    urls = [url for _title, url in links]
    assert urls == [
        "https://docs.example.invalid/welcome/readme.md",
        "https://docs.example.invalid/welcome/guides.md",
    ]
    assert not any("/fr/" in url for url in urls)  # other languages dropped
    assert not any(url.endswith(".png") for url in urls)  # non-markdown dropped
    assert len(urls) == len(set(urls))  # repeated link counted once


def test_a_section_the_index_lacks_names_the_sources_yaml_key_that_sets_it():
    with pytest.raises(UserError, match="section 'Deutsch' not found") as refused:
        parse_llms_index(LLMS_TXT, section="Deutsch")
    key = refused.value.hint.split("`")[1]
    assert key == "llms_section" and key in SourceSpec.model_fields
    assert "`paths.sources`" in refused.value.hint


def test_url_to_local_path_drops_host_and_space_slug():
    assert (
        url_to_local_path("https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding.md")
        == "trading/funding.md"
    )
    assert "?" not in url_to_local_path("https://h/space/a b?x=1/c.md")


def test_an_index_that_cannot_be_downloaded_is_a_user_error_naming_the_venue(tmp_path):
    import httpx
    import pytest

    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source
    from crossvenue_rag.corpus.sources import SourceSpec
    from crossvenue_rag.errors import UserError

    spec = SourceSpec(
        venue="pancakeswap",
        venue_type="dex",
        kind="gitbook",
        license="no license found",
        llms_index="https://docs.example.invalid/llms.txt",
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(404)))
    with pytest.raises(UserError, match="pancakeswap: could not download https://docs"):
        fetch_gitbook_source(spec, tmp_path, Config(), Http(client), [], None)


def test_a_retry_waits_the_backoff_even_with_no_pacing(tmp_path):
    import httpx

    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus import gitbook_source
    from crossvenue_rag.corpus.sources import SourceSpec

    spec = SourceSpec(
        venue="pancakeswap",
        venue_type="dex",
        kind="gitbook",
        license="no license found",
        llms_index="https://docs.example.invalid/llms.txt",
    )
    statuses = iter([500, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), text="")

    waits: list[float] = []
    cfg = Config()
    cfg.fetch.request_delay_seconds = 0.0
    cfg.fetch.backoff_seconds = 0.25
    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert (
        gitbook_source.fetch_gitbook_source(
            spec, tmp_path, cfg, gitbook_source.Http(client, waits.append), [], None
        )
        == []
    )
    assert waits == [0.25]


INDEX_URL = "https://docs.example.invalid/llms.txt"


def _gitbook_spec():
    from crossvenue_rag.corpus.sources import SourceSpec

    return SourceSpec(
        venue="pancakeswap",
        venue_type="dex",
        kind="gitbook",
        license="no license found",
        llms_index=INDEX_URL,
    )


def _serving(pages: list[str], requested: list[str]):
    """A client serving an llms.txt that lists `pages`, and "fetched" for every page."""
    import httpx

    index = "\n".join(f"- [Page](https://docs.example.invalid/{page})" for page in pages)

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, text=index if str(request.url) == INDEX_URL else "fetched")

    return Http(httpx.Client(transport=httpx.MockTransport(handler)))


def _raw_with_a_sibling(tmp_path):
    """data/raw holding another venue's file, next to data/outside holding a file."""
    raw, outside = tmp_path / "data" / "raw", tmp_path / "data" / "outside"
    (raw / "binance").mkdir(parents=True)
    (raw / "binance" / "keep.md").write_text("binance", encoding="utf-8")
    outside.mkdir()
    (outside / "keep.md").write_text("outside", encoding="utf-8")
    (raw / "linked").symlink_to(outside, target_is_directory=True)
    return raw, outside


def _unchanged(raw, outside):
    return (raw / "binance" / "keep.md").read_text(encoding="utf-8") == "binance" and (
        outside / "keep.md"
    ).read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("venue", ["", "   ", "..", "a/../..", "ABSOLUTE", "linked"])
def test_a_venue_outside_the_raw_directory_is_refused_before_anything_is_fetched(tmp_path, venue):
    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source
    from crossvenue_rag.errors import UserError

    raw, outside = _raw_with_a_sibling(tmp_path)
    # model_copy skips validation, as a spec that got past the sources file would;
    # ABSOLUTE stands for the absolute path of data/outside.
    unsafe = _gitbook_spec().model_copy(
        update={"venue": str(outside) if venue == "ABSOLUTE" else venue}
    )
    requested: list[str] = []
    with pytest.raises(UserError, match="names no path"):
        fetch_gitbook_source(
            unsafe, raw, Config(), _serving(["space/keep.md"], requested), [], None
        )
    assert requested == []
    assert _unchanged(raw, outside)


@pytest.mark.parametrize(
    "page",
    [
        # From data/raw/pancakeswap: `..` reaches data/raw, `../..` reaches data.
        "space/../../outside/keep.md",
        "space/../binance/keep.md",
        "space/a/../../../outside/keep.md",
        "space/../linked/keep.md",
    ],
)
def test_a_page_path_outside_the_venue_directory_is_refused_before_any_page_is_written(
    tmp_path, page
):
    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source
    from crossvenue_rag.errors import UserError

    raw, outside = _raw_with_a_sibling(tmp_path)
    requested: list[str] = []
    client = _serving(["space/first.md", page], requested)
    with pytest.raises(UserError, match="names no path inside") as refused:
        fetch_gitbook_source(_gitbook_spec(), raw, Config(), client, [], None)
    assert page in str(refused.value)
    assert requested == [INDEX_URL]
    assert not (raw / "pancakeswap" / "first.md").exists()
    assert _unchanged(raw, outside)


def test_a_symlinked_page_directory_that_points_outside_is_refused(tmp_path):
    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source
    from crossvenue_rag.errors import UserError

    raw, outside = _raw_with_a_sibling(tmp_path)
    (raw / "pancakeswap").mkdir()
    (raw / "pancakeswap" / "guides").symlink_to(outside, target_is_directory=True)
    client = _serving(["space/guides/keep.md"], [])
    with pytest.raises(UserError, match=r"'guides/keep\.md' names no path inside"):
        fetch_gitbook_source(_gitbook_spec(), raw, Config(), client, [], None)
    assert _unchanged(raw, outside)


def test_the_page_count_is_logged_with_its_noun_in_the_right_number(tmp_path, caplog):
    import logging

    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source

    caplog.set_level(logging.INFO, logger="crossvenue_rag.corpus.gitbook_source")
    rows = fetch_gitbook_source(
        _gitbook_spec(), tmp_path, Config(), _serving(["space/guides/one.md"], []), [], None
    )
    assert [row.local_path for row in rows] == ["pancakeswap/guides/one.md"]
    assert "pancakeswap: llms.txt lists 1 page" in caplog.messages


def test_the_delay_between_two_pages_is_spent_by_the_given_sleep(tmp_path):
    from crossvenue_rag.config import Config
    from crossvenue_rag.corpus.gitbook_source import fetch_gitbook_source

    cfg = Config()
    cfg.fetch.request_delay_seconds = 0.5
    served = _serving(["space/one.md", "space/two.md"], [])
    waits: list[float] = []
    rows = fetch_gitbook_source(
        _gitbook_spec(), tmp_path, cfg, Http(served.client, waits.append), [], None
    )
    assert len(rows) == 2 and waits == [0.5]
