"""Tests for `crossvenue_rag.errors`."""

from __future__ import annotations

import pytest

from crossvenue_rag.errors import UserError


def test_a_user_error_carries_its_message_and_hint():
    with pytest.raises(UserError, match="the index is stale") as info:
        raise UserError("the index is stale", hint="run make index")
    assert str(info.value) == "the index is stale"
    assert info.value.hint == "run make index"
    assert UserError("x").hint == ""


def test_an_error_as_one_line_keeps_its_hint():
    from crossvenue_rag.errors import UserError

    assert UserError("x is wrong", hint="pass y").with_hint() == "x is wrong (pass y)"
    assert UserError("x is wrong").with_hint() == "x is wrong"


def test_a_failure_outside_the_program_carries_a_hint_and_is_no_user_error():
    from crossvenue_rag.errors import ExpectedFailure, ExternalFailure

    failure = ExternalFailure("m: HTTP 502 after 3 retries", hint="re-run to retry")
    assert isinstance(failure, ExpectedFailure) and not isinstance(failure, UserError)
    assert failure.with_hint() == "m: HTTP 502 after 3 retries (re-run to retry)"
    assert isinstance(UserError("x"), ExpectedFailure)


def test_a_validation_problem_names_its_field_path_through_the_formatting_module(monkeypatch):
    from pydantic import BaseModel, ValidationError

    from crossvenue_rag.errors import validation_problems

    class Model(BaseModel):
        ks: list[int]

    with pytest.raises(ValidationError) as failed:
        Model.model_validate({"ks": [1, "x"]})
    assert validation_problems(failed.value).startswith("ks.1: ")
    monkeypatch.setattr("crossvenue_rag.formatting.own_form", lambda value: f"<{value}>")
    assert validation_problems(failed.value).startswith("<ks>.<1>: ")


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_a_blank_value_is_refused_naming_what_is_empty_and_the_fix(value):
    from crossvenue_rag.errors import is_blank, not_blank

    assert is_blank(value)
    with pytest.raises(UserError, match=r"^the query is empty$") as refused:
        not_blank("the query", value)
    assert refused.value.hint == "pass the query with some text"


def test_a_value_with_text_is_returned_as_given():
    from crossvenue_rag.errors import is_blank, not_blank

    assert not is_blank(" a ")
    assert not_blank("the query", " a ") == " a "
