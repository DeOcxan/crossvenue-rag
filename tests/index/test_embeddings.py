"""Tests for `index.embeddings`: device choice and the pinned model revision."""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np

from crossvenue_rag.config_pipeline import EmbeddingConfig
from crossvenue_rag.index.embeddings import Embedder, resolve_device


def test_an_explicit_device_is_used_as_given_else_mps_then_cuda_then_cpu(monkeypatch):
    assert resolve_device("cpu") == "cpu"

    def torch_with(*, mps: bool, cuda: bool):
        return types.SimpleNamespace(
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
            cuda=types.SimpleNamespace(is_available=lambda: cuda),
        )

    for mps, cuda, device in ((True, True, "mps"), (False, True, "cuda"), (False, False, "cpu")):
        monkeypatch.setitem(sys.modules, "torch", torch_with(mps=mps, cuda=cuda))
        assert resolve_device(None) == device


class FakeSentenceTransformer:
    loaded: ClassVar[list[tuple[str, str, str | None]]] = []

    def __init__(self, name, device, revision=None):
        FakeSentenceTransformer.loaded.append((name, device, revision))

    def encode(self, texts, **kwargs):
        self.texts, self.kwargs = texts, kwargs
        return [[1.0, 0.0]] * len(texts)


def test_the_model_loads_at_its_revision_and_queries_carry_the_prefix(monkeypatch):
    fake = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    cfg = EmbeddingConfig(device="cpu", revision="abc123", query_prefix="Q: ")
    embedder = Embedder(cfg)
    vector = embedder.encode_query("funding interval")
    assert FakeSentenceTransformer.loaded[-1] == (cfg.model_name, "cpu", "abc123")
    assert embedder.model.texts == ["Q: funding interval"]
    assert vector.dtype == np.float32 and vector.tolist() == [1.0, 0.0]
    assert embedder.model_fingerprint() == f"{cfg.model_name}@dim{cfg.dimension}"


def test_a_missing_sentence_transformers_is_a_user_error_naming_no_revision(monkeypatch):
    import pytest

    from crossvenue_rag.errors import UserError

    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(UserError, match=r"cannot load embedding\.model_name") as refused:
        _ = Embedder(EmbeddingConfig(device="cpu")).model
    assert "latest" not in str(refused.value) and " at revision" not in str(refused.value)
