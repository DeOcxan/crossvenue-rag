"""Tests for `crossvenue_rag.corpus.sources`."""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.sources import load_sources
from tests.corpus.conftest import spec


def test_sources_yaml_is_valid_and_covers_both_venue_types():
    specs = load_sources("data/sources.yaml")
    assert len(specs) >= 8
    assert {s.venue_type for s in specs} == {"cex", "dex"}
    # Every source records a license verbatim, including its absence.
    assert all(s.license for s in specs)
    assert all(s.repo for s in specs if s.kind == "git")
    assert all(s.llms_index for s in specs if s.kind == "gitbook")


def test_closed_vocabularies_are_enforced():
    with pytest.raises(ValueError, match="not one of"):
        spec(environment=[{"value": "staging", "path": "."}])
    with pytest.raises(ValueError, match="references group"):
        spec(api_version=[{"value": "{2}", "path": r"(v\d)"}])
    with pytest.raises(ValueError, match="needs `path`, `heading`"):
        spec(product=[{"value": "spot"}])


def test_a_misspelled_key_in_a_source_entry_is_refused(tmp_path):
    from crossvenue_rag.errors import UserError

    path = tmp_path / "sources.yaml"
    path.write_text(
        "sources:\n"
        "  - venue: binance\n    venue_type: cex\n    kind: git\n    license: MIT\n"
        "    repo: https://example.invalid/r\n    exlude: ['**/*_CN.md']\n",
        encoding="utf-8",
    )
    with pytest.raises(UserError, match="exlude"):
        load_sources(path)


@pytest.mark.parametrize(
    "text",
    [
        "sources:\n  - binance\n",
        "sources:\n  binance: {venue_type: cex}\n",
        "metadata:\n  environment:\n    - testnet\n",
        "metadata:\n  - environment\n",
    ],
)
def test_a_wrongly_shaped_sources_file_is_refused_with_its_hint(tmp_path, text):
    from crossvenue_rag.corpus.sources import load_global_metadata_rules
    from crossvenue_rag.errors import UserError

    path = tmp_path / "sources.yaml"
    path.write_text(text, encoding="utf-8")
    for load in (load_sources, load_global_metadata_rules):
        with pytest.raises(UserError, match=f"invalid {path}") as refused:
            load(path)
        assert refused.value.hint.startswith(f"fix the entry in {path}")


def test_a_misspelled_top_level_key_is_refused(tmp_path):
    from crossvenue_rag.errors import UserError

    path = tmp_path / "sources.yaml"
    path.write_text("sources: []\nmetdata: {}\n", encoding="utf-8")
    with pytest.raises(UserError, match="metdata"):
        load_sources(path)


@pytest.mark.parametrize("alias", ["", "okx?|", "a*", "^"])
def test_an_alias_that_matches_the_empty_string_is_refused(alias):
    from crossvenue_rag.corpus.sources import SourceSpec

    with pytest.raises(ValueError, match="matches the empty string"):
        SourceSpec(
            venue="okx", venue_type="cex", kind="git", repo="r", license="MIT", aliases=[alias]
        )


@pytest.mark.parametrize("venue", ["", "   ", ".", "..", "a/../..", "/tmp/elsewhere", "a\\b"])
def test_a_venue_that_is_not_a_plain_directory_name_is_refused_at_load(tmp_path, venue):
    from crossvenue_rag.errors import UserError

    path = tmp_path / "sources.yaml"
    path.write_text(
        f"sources:\n  - venue: {venue!r}\n    venue_type: cex\n    kind: git\n"
        "    license: MIT\n    repo: https://example.invalid/r\n",
        encoding="utf-8",
    )
    with pytest.raises(UserError, match="is not a plain directory name") as refused:
        load_sources(path)
    assert refused.value.hint.startswith(f"fix the entry in {path}")
