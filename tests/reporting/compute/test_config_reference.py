"""Tests for `reporting.compute.config_reference`: the schema's domains and the YAML's values."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.reporting.compute import config_reference as schema
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_file


@pytest.mark.parametrize(
    ("section", "key", "domain"),
    [
        ("chunking", "max_tokens", "int > 0"),
        ("chunking", "overlap_tokens", "int >= 0"),
        ("store", "hnsw_m", "int in [2, 100]"),
        ("store", "table", "str, at most 49 characters"),
        ("fetch", "timeout_seconds", "finite float > 0"),
        ("eval", "duplicate_threshold", "float in [0, 1]"),
        ("generation", "answer_schema_version", "1 or 2"),
        ("generation", "calibration_seed", "int"),
    ],
)
def test_a_number_key_has_the_domain_its_field_sets(section, key, domain):
    assert schema.field_domain(section, key) == domain


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("paths", "raw"),
        ("eval", "methods"),
        ("generation", "primary_scope"),
        ("fetch", "user_agent"),
    ],
)
def test_a_key_whose_domain_is_no_number_or_length_has_none(section, key):
    assert schema.field_domain(section, key) is None


def test_a_number_value_keeps_the_form_the_yaml_writes(tmp_path):
    write_file(
        tmp_path,
        "configs/default.yaml",
        "fetch:\n  timeout_seconds: 30\n  backoff_seconds: 0.5\n"
        "eval:\n  bootstrap_resamples: 10000\n  methods: [bm25, dense]\n"
        "chunking:\n  keep_code_blocks_intact: true\n",
    )
    src = Sources(tmp_path)
    assert schema.number_value(src, "fetch", "timeout_seconds") == "30"
    assert schema.number_value(src, "fetch", "backoff_seconds") == "0.5"
    assert schema.number_value(src, "eval", "bootstrap_resamples") == "10000"
    # A boolean and a list are no number; a key the file does not set has no value.
    assert schema.number_value(src, "chunking", "keep_code_blocks_intact") is None
    assert schema.number_value(src, "eval", "methods") is None
    assert schema.number_value(src, "fetch", "max_retries") is None
    assert schema.yaml_length(src, "eval", "methods") == 2


def test_a_prompt_pin_has_sixty_four_hex_digits():
    assert schema.sha256_digits() == 64


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: schema.section_model("absent"), "Config schema has no 'absent'"),
        (lambda: schema.field_domain("chunking", "absent"), "section chunking has no 'absent'"),
        (lambda: schema.schema_default_length("eval", "absent"), "section eval has no 'absent'"),
    ],
)
def test_a_section_or_key_the_schema_lacks_is_a_value_error(call, message):
    with pytest.raises(ValueError, match=message):
        call()


def test_a_dimension_without_a_threshold_is_a_value_error(tmp_path):
    write_file(
        tmp_path,
        "configs/default.yaml",
        "generation:\n  calibration_thresholds:\n    supported: 0.61\n",
    )
    with pytest.raises(ValueError, match="calibration_thresholds has no 'venue_correct'"):
        schema.shared_threshold(Sources(tmp_path), ["supported", "venue_correct"])


def test_a_threshold_keeps_the_decimals_of_a_config_threshold(tmp_path):
    write_file(tmp_path, "configs/default.yaml", "eval:\n  duplicate_threshold: 0.9\n")
    assert schema.number_value(Sources(tmp_path), "eval", "duplicate_threshold") == "0.90"


def test_a_config_section_that_is_no_mapping_is_a_user_error_naming_the_fix(tmp_path):
    write_file(tmp_path, "configs/default.yaml", "eval:\n  - duplicate_threshold\n")
    with pytest.raises(
        UserError, match=r"section eval of configs/default\.yaml is not a mapping"
    ) as e:
        schema.default_yaml(Sources(tmp_path))
    assert e.value.hint == "write the config as sections of keys"
