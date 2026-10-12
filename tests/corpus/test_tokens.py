"""Tests for `corpus.tokens`: the tokenizer is loaded at a revision or not at all."""

from __future__ import annotations

import sys
import types

import pytest

from crossvenue_rag.corpus.tokens import token_counter_for
from crossvenue_rag.errors import UserError


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens=True):
        return text.split() + (["[CLS]", "[SEP]"] if add_special_tokens else [])

    def num_special_tokens_to_add(self):
        return 2


def _fake_transformers(monkeypatch, loader):
    fake = types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=loader))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    token_counter_for.cache_clear()


def test_tokens_are_counted_without_special_tokens_at_the_pinned_revision(monkeypatch):
    calls = []

    def load(name, revision=None):
        calls.append((name, revision))
        return FakeTokenizer()

    _fake_transformers(monkeypatch, load)
    count = token_counter_for("m", "abc123", 450)
    assert count("three short words") == 3
    assert calls == [("m", "abc123")]
    token_counter_for.cache_clear()


def test_a_tokenizer_that_cannot_load_is_an_error_not_a_word_count(monkeypatch):
    def load(name, revision=None):
        raise OSError("not in the cache")

    _fake_transformers(monkeypatch, load)
    with pytest.raises(UserError, match="cannot load the tokenizer of m at revision abc"):
        token_counter_for("m", "abc", 450)
    token_counter_for.cache_clear()


@pytest.mark.parametrize("error", [ValueError("bad revision"), ImportError("no transformers")])
def test_a_malformed_revision_or_a_missing_library_is_a_user_error(monkeypatch, error):
    def load(name, revision=None):
        raise error

    _fake_transformers(monkeypatch, load)
    with pytest.raises(UserError, match="cannot load the tokenizer"):
        token_counter_for("m", "abc", 450)
    token_counter_for.cache_clear()


def test_an_unexpected_exception_inside_the_load_keeps_its_traceback(monkeypatch):
    def load(name, revision=None):
        raise RuntimeError("a defect in the loader")

    _fake_transformers(monkeypatch, load)
    with pytest.raises(RuntimeError, match="a defect in the loader"):
        token_counter_for("m", "abc", 450)
    token_counter_for.cache_clear()


@pytest.mark.parametrize("max_tokens", [511, 512])
def test_a_chunk_budget_above_the_window_less_the_special_tokens_is_refused(
    monkeypatch, max_tokens
):
    class WindowedTokenizer(FakeTokenizer):
        model_max_length = 512

    _fake_transformers(monkeypatch, lambda name, revision=None: WindowedTokenizer())
    assert token_counter_for("m", None, 510)("a b") == 2
    with pytest.raises(
        UserError,
        match=rf"chunking\.max_tokens \({max_tokens}\) exceeds the 510 tokens of text m encodes "
        r"\(512 with its special tokens\)",
    ):
        token_counter_for("m", None, max_tokens)
    token_counter_for.cache_clear()


def test_the_refused_budget_and_the_one_to_set_keep_the_config_values_form(monkeypatch):
    class WideTokenizer(FakeTokenizer):
        model_max_length = 2048

    _fake_transformers(monkeypatch, lambda name, revision=None: WideTokenizer())
    with pytest.raises(UserError, match=r"chunking\.max_tokens \(4096\) exceeds") as refused:
        token_counter_for("m", None, 4096)
    assert refused.value.hint == "set chunking.max_tokens to at most 2046"
    token_counter_for.cache_clear()


def test_the_refused_budget_and_the_one_to_set_are_written_through_the_formatting_module(
    monkeypatch,
):
    class WideTokenizer(FakeTokenizer):
        model_max_length = 2048

    _fake_transformers(monkeypatch, lambda name, revision=None: WideTokenizer())
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    with pytest.raises(UserError, match=r"chunking\.max_tokens \(<4096>\) exceeds") as refused:
        token_counter_for("m", None, 4096)
    assert refused.value.hint == "set chunking.max_tokens to at most <2046>"
    token_counter_for.cache_clear()


def test_a_tokenizer_without_a_pinned_revision_is_named_without_one(monkeypatch):
    def load(name, revision=None):
        raise OSError("not in the cache")

    _fake_transformers(monkeypatch, load)
    with pytest.raises(UserError, match="cannot load the tokenizer of m: not in the cache"):
        token_counter_for("m", None, 450)
    token_counter_for.cache_clear()
