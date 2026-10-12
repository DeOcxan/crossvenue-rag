"""Tests for `crossvenue_rag.retrieval.filters`."""

from __future__ import annotations

import pytest

from crossvenue_rag.corpus.dedup import deduplicate
from crossvenue_rag.errors import UserError
from crossvenue_rag.retrieval.filters import filters_from_options, resolve_metadata, resolve_venues
from tests.conftest import occurrence

VENUE_TYPES = {
    "binance": "cex",
    "bybit": "cex",
    "uniswap": "dex",
    "pancakeswap": "dex",
}


def test_venue_filters_resolve_to_an_explicit_allow_list():
    """venue_type is expanded here so BM25 and Postgres apply identical filters."""
    assert resolve_venues(None, VENUE_TYPES) is None
    assert resolve_venues({}, VENUE_TYPES) is None
    assert resolve_venues({"venue": "bybit"}, VENUE_TYPES) == ["bybit"]
    assert resolve_venues({"venue_type": "dex"}, VENUE_TYPES) == ["pancakeswap", "uniswap"]
    assert resolve_venues({"venue": ["bybit", "uniswap"], "venue_type": "cex"}, VENUE_TYPES) == [
        "bybit"
    ]


def test_unknown_venue_is_an_error_not_an_empty_result():
    """Silently returning nothing for a typo is how eval runs quietly become zero."""
    with pytest.raises(ValueError, match="venues not in the venue map: bitmex"):
        resolve_venues({"venue": "bitmex"}, VENUE_TYPES)
    with pytest.raises(ValueError, match="no venues of type"):
        resolve_venues({"venue_type": "cefi"}, VENUE_TYPES)


def test_unsupported_filter_key_is_rejected():
    with pytest.raises(ValueError, match="unsupported filter keys"):
        resolve_venues({"heading": "Rate Limits"}, VENUE_TYPES)


def test_metadata_filter_matches_any_occurrence_and_rejects_unknown_values():
    [merged], _ = deduplicate(
        [
            occurrence("binance", "rest-api.md", "same", env="mainnet"),
            occurrence("binance", "testnet/rest-api.md", "same", env="testnet"),
        ]
    )
    other = occurrence("binance", "x.md", "different", env="mainnet")
    chunks = [merged, other]
    assert resolve_metadata({"environment": "testnet"}, chunks) == {merged.chunk_id}
    assert resolve_metadata({"environment": ["mainnet"]}, chunks) == {
        merged.chunk_id,
        other.chunk_id,
    }
    assert resolve_metadata({"venue": "binance"}, chunks) is None
    with pytest.raises(UserError, match="no chunk has environment"):
        resolve_metadata({"environment": "staging"}, chunks)


def test_a_venue_outside_the_venue_type_is_an_error_not_every_venue():
    """Both indexes read an empty allow-list as no filter at all."""
    with pytest.raises(ValueError, match="no venue is both in binance and of type dex"):
        resolve_venues({"venue": ["binance"], "venue_type": "dex"}, VENUE_TYPES)


def test_no_option_is_no_filter():
    assert filters_from_options([], None, [], VENUE_TYPES) is None


def test_options_become_one_filter_dict_with_repeated_fields_ored():
    filters = filters_from_options(
        ["binance", "bybit"],
        "cex",
        ["environment=testnet", " environment = None ", "api_version=v5"],
        VENUE_TYPES,
    )
    assert filters == {
        "venue": ["binance", "bybit"],
        "venue_type": "cex",
        "environment": ["testnet", None],
        "api_version": ["v5"],
    }


def test_a_filter_without_an_equals_sign_is_a_user_error_with_an_example():
    with pytest.raises(UserError, match="bad --filter 'testnet'") as info:
        filters_from_options([], None, ["testnet"], VENUE_TYPES)
    assert "pass --filter as FIELD=VALUE, for example environment=testnet" in (
        info.value.hint or ""
    )


def test_a_filter_on_a_venue_field_is_refused_so_it_cannot_replace_venue():
    from crossvenue_rag.retrieval.filters import filters_from_options

    for raw in ("venue=bybit", "venue_type=dex", "venue=none"):
        with pytest.raises(UserError, match="is not a metadata field"):
            filters_from_options(["binance"], None, [raw], VENUE_TYPES)


@pytest.mark.parametrize(
    ("venues", "venue_type", "message", "hint"),
    [
        (
            ["bitmex"],
            None,
            "--venue bitmex: 1 unknown venue",
            "pass --venue one of the venues of the sources file (`paths.sources`): ",
        ),
        (
            [],
            "cefi",
            "--venue-type 'cefi' is not a venue type",
            "pass --venue-type one of the venue types of the sources file (`paths.sources`): "
            "cex or dex",
        ),
        (["binance"], "dex", r"no --venue given \(binance\) is of --venue-type dex", "drop one"),
    ],
)
def test_a_venue_option_outside_its_domain_is_refused_naming_the_option(
    venues, venue_type, message, hint
):
    with pytest.raises(UserError, match=message) as refused:
        filters_from_options(venues, venue_type, [], VENUE_TYPES)
    assert hint in refused.value.hint
