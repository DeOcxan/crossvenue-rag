"""Tests for `yaml_files`: a YAML file whose top level must be a mapping."""

from __future__ import annotations

import pytest

from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError
from crossvenue_rag.yaml_files import read_mapping


def test_a_mapping_is_read_and_an_empty_file_reads_as_empty(tmp_path):
    path = tmp_path / "a.yaml"
    path.write_text("retrieval:\n  k: 7\n", encoding="utf-8")
    assert read_mapping(path, hint="h") == {"retrieval": {"k": 7}}
    path.write_text("", encoding="utf-8")
    assert read_mapping(path, hint="h") == {}


@pytest.mark.parametrize("text", ["- retrieval\n", "just words\n"])
def test_a_list_or_a_scalar_at_the_top_is_a_user_error_naming_the_file(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UserError, match="where a mapping of keys is expected") as caught:
        read_mapping(path, hint="fix it here")
    assert str(path) in str(caught.value) and caught.value.hint == "fix it here"
    with pytest.raises(UserError, match="where a mapping of keys is expected"):
        load_config(path)


def test_a_file_that_is_not_utf8_is_refused_naming_it_with_a_hint_to_save_it_as_utf8(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_bytes(b"retrieval:\n  k: \xff\n")
    with pytest.raises(UserError, match="is not UTF-8 text") as caught:
        read_mapping(path, hint="write the config as sections of keys")
    assert str(path) in str(caught.value)
    assert caught.value.hint == f"save {path} with UTF-8 encoding"


def test_a_top_level_key_that_is_not_a_name_is_a_user_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("1: x\n", encoding="utf-8")
    with pytest.raises(UserError, match="top-level key 1, which is not a name"):
        read_mapping(path, hint="h")
