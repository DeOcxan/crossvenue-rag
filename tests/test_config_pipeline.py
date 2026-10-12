"""Tests for `crossvenue_rag.config_pipeline`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossvenue_rag.config_loading import load_config
from crossvenue_rag.config_pipeline import ChunkingConfig, StoreConfig
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_overlap_must_be_smaller_than_max_tokens():
    """Otherwise chunking never advances."""
    with pytest.raises(UserError, match="overlap_tokens"):
        load_config(DEFAULT, ["chunking.overlap_tokens=450"])


def test_a_refused_config_value_keeps_its_own_form_in_the_message():
    """Style rule 11: a config value is quoted as written, with no thousands separator."""
    with pytest.raises(
        UserError, match=r"overlap_tokens \(1300\) must be smaller than max_tokens \(1200\)"
    ):
        load_config(DEFAULT, ["chunking.max_tokens=1200", "chunking.overlap_tokens=1300"])


def test_the_refused_chunking_and_hnsw_values_are_written_through_the_formatting_module(
    monkeypatch,
):
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(
        ValidationError, match=r"\(<1300>\) must be smaller than max_tokens \(<1200>\)"
    ):
        ChunkingConfig(max_tokens=1200, overlap_tokens=1300)
    with pytest.raises(
        ValidationError, match=r"min_tokens \(<500>\) must not exceed max_tokens \(<450>\)"
    ):
        ChunkingConfig(min_tokens=500)
    with pytest.raises(ValidationError, match=r"\(<64>\) must be at least 2 times hnsw_m \(<40>\)"):
        StoreConfig(hnsw_m=40)


def test_table_name_must_be_an_identifier():
    """The table name is spliced into DDL, so it may not be arbitrary text."""
    with pytest.raises(UserError, match="ASCII letters, digits and underscores"):
        load_config(DEFAULT, ["store.table='chunks; DROP TABLE users'"])


@pytest.mark.parametrize("table", ["chunks\u00e9", "t\u00e4ble", "a-b"])
def test_a_table_name_outside_ascii_identifiers_is_refused(table):
    """The 49-character limit is Postgres's byte limit only for ASCII names."""
    with pytest.raises(ValidationError, match="ASCII letters, digits and underscores"):
        StoreConfig(table=table)


def test_dense_candidates_may_exceed_hnsw_ef_search():
    """A search deeper than ef_search is exact (`index.store`), so nothing is cut."""
    assert (
        load_config(DEFAULT, ["retrieval.dense_candidates=150"]).retrieval.dense_candidates == 150
    )


def test_hnsw_construction_is_at_least_twice_the_degree():
    with pytest.raises(UserError, match="at least 2 times"):
        load_config(DEFAULT, ["store.hnsw_m=40", "store.hnsw_ef_construction=64"])


def test_user_agent_names_the_package_version():
    from importlib.metadata import version

    from crossvenue_rag.config_pipeline import FetchConfig

    assert FetchConfig().user_agent == (
        f"crossvenue-rag/{version('crossvenue-rag')} (research corpus builder)"
    )


# Each key given a value outside its domain through `--set`, and the key the refusal names.
KEY_OUT_OF_DOMAIN = [
    ("chunking.max_tokens=0", "chunking.max_tokens"),
    ("embedding.model_name=' '", "embedding.model_name"),
    ("embedding.device=tpu", "embedding.device"),
    ("embedding.revision=''", "embedding.revision"),
    ("embedding.batch_size=0", "embedding.batch_size"),
    ("reranker.model_name=''", "reranker.model_name"),
    ("reranker.revision=''", "reranker.revision"),
    ("retrieval.k=0", "retrieval.k"),
    ("retrieval.rrf_k=0", "retrieval.rrf_k"),
    ("store.dsn=notadsn", "store.dsn"),
    ("store.table=" + "t" * 50, "store.table"),
    ("store.hnsw_m=1", "store.hnsw_m"),
    ("store.hnsw_m=101", "store.hnsw_m"),
    ("store.hnsw_ef_construction=2000", "store.hnsw_ef_construction"),
    ("store.hnsw_ef_search=2000", "store.hnsw_ef_search"),
    ("fetch.request_delay_seconds=-1", "fetch.request_delay_seconds"),
    ("fetch.timeout_seconds=.inf", "fetch.timeout_seconds"),
    ("fetch.user_agent=' '", "fetch.user_agent"),
    ("fetch.max_retries=11", "fetch.max_retries"),
    ("fetch.backoff_seconds=0", "fetch.backoff_seconds"),
]


@pytest.mark.parametrize(("override", "key"), KEY_OUT_OF_DOMAIN)
def test_a_key_outside_its_domain_is_refused_naming_it(override, key):
    with pytest.raises(UserError) as refused:
        load_config(DEFAULT, [override])
    assert str(refused.value).startswith(f"invalid --set {override}: ")
    assert key in str(refused.value)
