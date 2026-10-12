"""Tests for `crossvenue_rag.generation.api_key`."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.generation import api_key
from crossvenue_rag.generation.api_key import KEY_NAME, load_api_key


def _read_from(monkeypatch, path):
    """Points the dotenv file the key is read from at `path`."""
    monkeypatch.setattr(api_key, "DOTENV_FILE", path)


def test_the_api_key_comes_from_the_environment_first(tmp_path, monkeypatch):

    keys = tmp_path / "keys"
    keys.write_text(f"{KEY_NAME}=from-file\n", encoding="utf-8")
    monkeypatch.setenv(KEY_NAME, "from-env")
    _read_from(monkeypatch, keys)
    assert load_api_key() == "from-env"


@pytest.mark.parametrize(
    ("line", "key"),
    [
        ('OPENROUTER_API_KEY="sk-quoted # not a comment"', "sk-quoted # not a comment"),
        ("OPENROUTER_API_KEY='sk-single'", "sk-single"),
        ("OPENROUTER_API_KEY=sk-abc  # my key", "sk-abc"),
        ("export OPENROUTER_API_KEY=sk-exported", "sk-exported"),
        ("OTHER=x\nOPENROUTER_API_KEY = sk-spaced", "sk-spaced"),
    ],
)
def test_the_key_file_is_read_as_dotenv(tmp_path, monkeypatch, line, key):

    monkeypatch.delenv(KEY_NAME, raising=False)
    keys = tmp_path / "keys"
    keys.write_text(line + "\n", encoding="utf-8")
    _read_from(monkeypatch, keys)
    assert load_api_key() == key


def test_a_missing_key_is_a_user_error_that_names_no_value(tmp_path, monkeypatch):

    monkeypatch.delenv(KEY_NAME, raising=False)
    keys = tmp_path / "keys"
    keys.write_text("OTHER_KEY=sk-other-secret\n", encoding="utf-8")
    _read_from(monkeypatch, keys)
    with pytest.raises(UserError, match=f"{KEY_NAME} is not set") as info:
        load_api_key()
    assert "sk-other-secret" not in str(info.value) + info.value.hint
    _read_from(monkeypatch, tmp_path / "absent")
    with pytest.raises(UserError, match=f"{KEY_NAME} is not set"):
        load_api_key()


def test_a_key_file_that_is_not_utf8_is_a_user_error_that_echoes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_NAME, raising=False)
    keys = tmp_path / "keys"
    keys.write_bytes(b"OPENROUTER_API_KEY=sk-secret-\xff\n")
    _read_from(monkeypatch, keys)
    with pytest.raises(UserError, match="keys is not UTF-8 text") as info:
        load_api_key()
    assert "sk-secret" not in str(info.value) + info.value.hint
