"""Tests for `crossvenue_rag.config_loading`: the file, `--set` overrides and the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError

DEFAULT = Path("configs/default.yaml")


def test_overrides_are_yaml_parsed():
    """`--set` is how an ablation runs without editing code."""
    cfg = load_config(DEFAULT, overrides=["chunking.max_tokens=256", "eval.methods=[bm25, dense]"])
    assert cfg.chunking.max_tokens == 256
    assert cfg.eval.methods == ["bm25", "dense"]


def test_dsn_comes_from_environment(monkeypatch):
    """So a password never has to reach a YAML file."""
    monkeypatch.setenv("CROSSVENUE_PG_DSN", "postgresql://u:p@db:5432/x")
    assert load_config(DEFAULT).store.dsn == "postgresql://u:p@db:5432/x"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("retrieval.k=[", "the value is not valid YAML"),
        ("retrieval.k={a", "the value is not valid YAML"),
        ("retrieval.k=@x", "the value is not valid YAML"),
        ("a..b=1", "has an empty key"),
        ("=1", "has an empty key"),
        ("retrieval.k.x=1", "retrieval.k holds a value, so it has no keys to set"),
        ("retrieval.k: 7", "override must look like section.field=value"),
    ],
)
def test_a_malformed_override_is_refused_naming_it(override, message):
    with pytest.raises(UserError, match=message) as refused:
        load_config(DEFAULT, [override])
    assert override.partition("=")[0].partition(":")[0] in str(refused.value)


def test_a_config_that_is_not_a_yaml_file_of_sections_is_refused(tmp_path: Path):
    with pytest.raises(UserError, match="is a directory where a file is expected"):
        load_config(tmp_path)
    broken = tmp_path / "broken.yaml"
    broken.write_text("retrieval: [\n", encoding="utf-8")
    with pytest.raises(UserError, match=r"broken\.yaml is not valid YAML: .*line 2"):
        load_config(broken)


def test_the_dsn_of_the_environment_fills_an_empty_store_section(tmp_path, monkeypatch):
    path = tmp_path / "c.yaml"
    path.write_text("store:\nretrieval:\n  k: 5\n", encoding="utf-8")
    monkeypatch.setenv("CROSSVENUE_PG_DSN", "postgresql://u:p@db:5432/x")
    assert load_config(path).store.dsn == "postgresql://u:p@db:5432/x"
    monkeypatch.setenv("CROSSVENUE_PG_DSN", "not a dsn")
    with pytest.raises(UserError, match=r"store\.dsn"):
        load_config(path)


def test_no_environment_variable_sets_a_section(tmp_path, monkeypatch):
    path = tmp_path / "partial.yaml"
    path.write_text("retrieval:\n  k: 5\n", encoding="utf-8")
    monkeypatch.setenv("EVAL", '{"k_values": [3], "pool_depth_per_method": 21}')
    monkeypatch.setenv("PATHS", "not json")
    cfg = load_config(path)
    assert cfg.eval.k_values == [1, 5, 10] and cfg.paths.raw == Path("data/raw")


def test_a_config_whose_top_level_key_is_a_number_is_refused_naming_the_file(tmp_path):
    path = tmp_path / "numbered.yaml"
    path.write_text("1: x\n", encoding="utf-8")
    with pytest.raises(UserError, match=r"numbered\.yaml has the top-level key 1") as refused:
        load_config(path)
    assert "configs/default.yaml" in refused.value.hint


def test_an_override_creates_a_missing_section_and_a_store_value_is_left_to_the_schema(
    tmp_path, monkeypatch
):
    path = tmp_path / "c.yaml"
    path.write_text("store: 5\n", encoding="utf-8")
    monkeypatch.setenv("CROSSVENUE_PG_DSN", "postgresql://u:p@db:5432/x")
    with pytest.raises(UserError, match=r"invalid .*c\.yaml: store"):
        load_config(path)
    path.write_text("{}\n", encoding="utf-8")
    assert load_config(path, ["retrieval.k=7"]).retrieval.k == 7


def test_a_value_the_schema_rejects_names_the_override_that_supplied_it():
    with pytest.raises(UserError) as caught:
        load_config(DEFAULT, overrides=["retrieval.k=5", "retrieval.k=yes"])
    message = str(caught.value)
    assert message.startswith("invalid --set retrieval.k=yes: retrieval.k:")
    assert "configs/default.yaml:" not in message
    assert caught.value.hint.startswith("change --set retrieval.k to a value")


def test_a_value_the_schema_rejects_in_the_file_names_the_file_and_the_fix(tmp_path: Path):
    path = tmp_path / "c.yaml"
    path.write_text("retrieval:\n  k: yes\n", encoding="utf-8")
    with pytest.raises(UserError, match=f"invalid {path}: retrieval.k:") as refused:
        load_config(path, overrides=["chunking.max_tokens=256"])
    assert refused.value.hint.startswith(f"change each key the error names in {path} to a value")


def test_a_rejected_value_beside_a_rejected_section_names_both_overrides():
    with pytest.raises(
        UserError, match=r"--set retrieval\.k=abc, --set store\.hnsw_m=40"
    ) as refused:
        load_config(DEFAULT, ["retrieval.k=abc", "store.hnsw_m=40"])
    assert refused.value.hint == (
        "change --set retrieval.k and --set store.hnsw_m to a value the error allows, or drop them"
    )
