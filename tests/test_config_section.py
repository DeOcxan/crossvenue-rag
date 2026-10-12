"""Tests for `crossvenue_rag.config_section`."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_a_misspelled_key_inside_a_section_is_refused(tmp_path):
    with pytest.raises(UserError, match="kk"):
        load_config(DEFAULT, ["retrieval.kk=20"])
    path = tmp_path / "typo.yaml"
    path.write_text("generation:\n  generators:\n    g:\n      model: m/x\n      modle: m/y\n")
    with pytest.raises(UserError, match="modle"):
        load_config(path)


@pytest.mark.parametrize(
    "override",
    [
        "embedding.model_name=' '",
        "reranker.model_name=''",
        "fetch.user_agent=' '",
        "generation.generators.gemini.provider_order=['']",
    ],
)
def test_a_blank_name_is_refused_as_blank(override):
    with pytest.raises(UserError, match="must not be empty or blank"):
        load_config(DEFAULT, [override])


@pytest.mark.parametrize(
    "override",
    [
        "retrieval.k=yes",
        "retrieval.k=true",
        "fetch.timeout_seconds=on",
        "eval.k_values=[1, yes]",
        "eval.duplicate_threshold=true",
        "generation.calibration_thresholds={supported: yes}",
        "generation.answer_schema_version=true",
        "generation.generator_params.temperature=no",
    ],
)
def test_a_number_key_refuses_a_yaml_boolean(override):
    """YAML reads yes, on and true as True, which pydantic would take as the number 1."""
    with pytest.raises(UserError, match="true or false where a number is required"):
        load_config(DEFAULT, [override])


def test_a_float_key_still_takes_an_integer():
    cfg = load_config(DEFAULT, ["fetch.timeout_seconds=30", "generation.budget_usd=2"])
    assert (cfg.fetch.timeout_seconds, cfg.generation.budget_usd) == (30.0, 2.0)


def test_a_repeated_value_is_refused_naming_the_key_and_the_value():
    from crossvenue_rag.config_section import distinct

    assert distinct([1, 2], "k_values") == [1, 2]
    with pytest.raises(ValueError, match="k_values repeats 5"):
        distinct([5, 5, 10], "k_values")


def test_a_repeated_value_is_named_through_the_formatting_module(monkeypatch):
    from crossvenue_rag.config_section import distinct

    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(ValueError, match="k_values repeats <1000>"):
        distinct([1000, 1000], "k_values")
