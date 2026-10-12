"""Tests for `retrieval.methods`: the method names and their lookup."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError
from crossvenue_rag.retrieval.methods import METHODS, ORACLE_METHODS, POOL_METHODS, method_named
from tests.conftest import HEAVY_MODULES


def test_a_known_name_is_returned_as_its_method():
    assert [method_named(m) for m in METHODS] == list(METHODS)


def test_an_unknown_name_is_refused_with_every_method_in_the_hint():
    with pytest.raises(UserError, match="--method 'bm52' is not a retrieval method") as caught:
        method_named("bm52")
    assert all(m in (caught.value.hint or "") for m in METHODS)


def test_the_pool_never_holds_the_oracle():
    assert not set(POOL_METHODS) & set(ORACLE_METHODS)
    assert set(POOL_METHODS) | set(ORACLE_METHODS) == set(METHODS)


def test_importing_the_cli_the_service_or_the_labeling_modes_loads_no_heavy_library():
    import subprocess
    import sys

    modules = (
        "crossvenue_rag.cli.app",
        "crossvenue_rag.service.app",
        "crossvenue_rag.evaluation.labels",
        "crossvenue_rag.labeling.modes",
    )
    code = (
        f"import sys\nfor m in {modules!r}: __import__(m)\n"
        f"print(','.join(h for h in {HEAVY_MODULES!r} if h in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_an_ad_hoc_question_can_use_every_method_but_an_oracle():
    from crossvenue_rag.retrieval.methods import AD_HOC_METHODS, ad_hoc_method

    assert [ad_hoc_method(m, "--retriever") for m in AD_HOC_METHODS] == list(AD_HOC_METHODS)
    with pytest.raises(UserError, match="--retriever venue_split_oracle: it reads a question's"):
        ad_hoc_method("venue_split_oracle", "--retriever")
    with pytest.raises(UserError, match="--retriever 'bm52' is not a retrieval method") as typo:
        ad_hoc_method("bm52", "--retriever")
    # The hint lists only the methods the option takes.
    assert "venue_split_oracle" not in (typo.value.hint or "")
