"""Tests for `crossvenue_rag.config_generation`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crossvenue_rag.config_generation import GenerationConfig
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_matrix_names_are_unique_and_the_primary_pair_names_two_of_them():
    matrix = (
        "generation.matrix=[{name: a, generator: gemini, retriever: bm25},"
        " {name: a, generator: qwen, retriever: bm25}]"
    )
    with pytest.raises(UserError, match=r"generation\.matrix names repeats a"):
        load_config(DEFAULT, [matrix, "generation.primary_pair=[]"])
    for pair in ("[gemini@venue_split]", "[gemini@venue_split, gemini@venue_split]"):
        with pytest.raises(UserError, match="primary_pair names 2 different configurations"):
            load_config(DEFAULT, [f"generation.primary_pair={pair}"])


def test_a_threshold_outside_the_agreement_range_is_refused_as_configs_write_it(monkeypatch):
    with pytest.raises(ValidationError, match=r"supported \(5\.00\) does not"):
        GenerationConfig(calibration_thresholds={"supported": 5.0})
    monkeypatch.setattr(
        "crossvenue_rag.formatting.own_form", lambda value, decimals=None: f"<{value} {decimals}>"
    )
    with pytest.raises(ValidationError, match=r"supported \(<5.0 2>\) does not"):
        GenerationConfig(calibration_thresholds={"supported": 5.0})


# Each key given a value outside its domain through `--set`, and the key the refusal names.
KEY_OUT_OF_DOMAIN = [
    ("generation.max_retries=11", "generation.max_retries"),
    ("generation.budget_usd=.nan", "generation.budget_usd"),
    ("generation.timeout_seconds=.inf", "generation.timeout_seconds"),
    ("generation.generator_prompt_sha256=zz", "generation.generator_prompt_sha256"),
    ("generation.judge_prompt_sha256=" + "A" * 64, "generation.judge_prompt_sha256"),
    ("generation.primary_prompt_sha256=zz", "generation.primary_prompt_sha256"),
    ("generation.answer_schema_version=3", "generation.answer_schema_version"),
    ("generation.primary_metrics=[nosuch]", "generation.primary_metrics"),
    ("generation.primary_scope=nosuch", "generation.primary_scope"),
    ("generation.calibration_thresholds={nosuch: 0.5}", "generation.calibration_thresholds"),
    ("generation.calibration_thresholds={supported: 5}", "calibration_thresholds"),
    ("generation.calibration_size=0", "generation.calibration_size"),
    ("generation.generators.gemini.model=' '", "generation.generators.gemini.model"),
    ("generation.generators.gemini.provider_order=['']", "provider_order"),
    ("generation.judge=nosuch", "generation.judge"),
]


@pytest.mark.parametrize(("override", "key"), KEY_OUT_OF_DOMAIN)
def test_a_key_outside_its_domain_is_refused_naming_it(override, key):
    with pytest.raises(UserError) as refused:
        load_config(DEFAULT, [override])
    assert str(refused.value).startswith(f"invalid --set {override}: ")
    assert key in str(refused.value)
