"""Tests for `index.model_loading`: the device choice and the hint for a failed load."""

from __future__ import annotations

import sys
import types

from crossvenue_rag.errors import INSTALL_HINT
from crossvenue_rag.index.model_loading import MODEL_HINT, load_hint, resolve_device


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


def test_a_failed_import_hints_the_install_and_a_failed_model_load_the_model():
    assert load_hint(ImportError("no sentence_transformers")) == INSTALL_HINT
    assert load_hint(OSError("not in the cache")) == MODEL_HINT
    assert load_hint(ValueError("malformed")) == MODEL_HINT
