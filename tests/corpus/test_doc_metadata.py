"""Tests for `crossvenue_rag.corpus.doc_metadata`."""

from __future__ import annotations

from crossvenue_rag.corpus.doc_metadata import MetadataRules, coverage
from crossvenue_rag.corpus.sources import MetadataRule
from tests.conftest import occurrence
from tests.corpus.conftest import spec


def test_first_matching_rule_wins_and_capture_groups_fill_the_value():
    rules = MetadataRules(
        [
            spec(
                api_version=[{"value": "{1}", "path": r"^docs/(v\d+)/"}],
                doc_type=[
                    {"value": "guide", "path": r"guide\.mdx$"},
                    {"value": "reference", "path": r"^docs/"},
                ],
            )
        ]
    )
    assert rules.derive("bybit", "docs/v5/guide.mdx", []) == {
        "environment": None,
        "api_version": "v5",
        "product": None,
        "doc_type": "guide",
    }
    assert rules.derive("bybit", "docs/v3/order.mdx", [])["doc_type"] == "reference"
    assert rules.derive("bybit", "other/page.md", [])["api_version"] is None


def test_a_null_rule_stops_later_rules_including_global_ones():
    global_rules = {"environment": [MetadataRule(value="testnet", heading=r"(?i)^testnet")]}
    rules = MetadataRules([spec(environment=[{"value": None, "path": "^demo/"}])], global_rules)
    assert rules.derive("bybit", "demo/x.md", ["Testnet"])["environment"] is None
    assert rules.derive("bybit", "docs/x.md", ["Testnet"])["environment"] == "testnet"
    assert rules.derive("bybit", "docs/x.md", ["Not testnet"])["environment"] is None


def test_the_real_sources_file_loads_and_labels_known_paths():
    rules = MetadataRules.from_file("data/sources.yaml")
    assert rules.derive("binance", "testnet/rest-api.md", [])["environment"] == "testnet"
    assert rules.derive("binance", "rest-api.md", [])["environment"] == "mainnet"
    assert rules.derive("binance", "demo-mode/general-info.md", [])["environment"] is None
    assert rules.derive("uniswap", "content/protocols/v4/guides/x.mdx", [])["api_version"] == "v4"
    assert (
        rules.derive("uniswap", "content/sdks/v3/g.mdx", ["Local Node with a Mainnet Fork"])[
            "environment"
        ]
        is None
    )


def test_coverage_reports_share_and_unassigned_venues():
    a = occurrence("binance", "a.md", "one", env="mainnet")
    b = occurrence("bybit", "b.md", "two")
    report = coverage([a, b])
    env = report.fields["environment"]
    assert env.assigned == 1
    assert env.share == 0.5
    assert env.unassigned_by_venue == {"bybit": 1}
