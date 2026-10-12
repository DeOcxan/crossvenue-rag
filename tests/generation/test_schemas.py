"""Tests for `generation.schemas`: answer and judgment parsing, the outcome rule."""

from __future__ import annotations

from typing import Any

import pytest

from crossvenue_rag.errors import ExternalFailure, UserError
from crossvenue_rag.generation.schemas import (
    ANSWER_SCHEMA_V2,
    InvalidResponse,
    parse_answer,
    parse_judgement,
)


def test_answer_parses_and_cited_chunk_numbers_are_bounded():
    content: dict[str, Any] = {
        "answer": "Binance bans for 2 minutes to 3 days [1].",
        "claims": [{"text": "bans scale", "venue": "binance", "chunks": [1]}],
        "refused": False,
        "asked_clarification": False,
    }
    assert parse_answer(content, 3, schema_version=1).claims[0].chunks == [1]
    content["claims"][0]["chunks"] = [4]
    with pytest.raises(InvalidResponse, match=r"outside 1\.\.3"):
        parse_answer(content, 3, schema_version=1)
    with pytest.raises(InvalidResponse, match="schema"):
        parse_answer({"claims": []}, 3, schema_version=1)


def test_schema_objects_forbid_extra_properties_on_every_object():
    def objects(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                yield node
            for value in node.values():
                yield from objects(value)

    assert all(o["additionalProperties"] is False for o in objects(ANSWER_SCHEMA_V2))


def test_judgement_needs_one_verdict_per_claim_and_sane_coverage():
    base = {"coverage": [], "behaviour_correct": True, "behaviour_rationale": ""}
    two = [
        {"index": 2, "supported": True, "venue_correct": True, "rationale": ""},
        {"index": 1, "supported": False, "venue_correct": True, "rationale": ""},
    ]
    judged = parse_judgement({**base, "claims": two}, 2, set())
    assert [c.index for c in judged.claims] == [1, 2]
    with pytest.raises(InvalidResponse, match=r"expected 1\.\.3"):
        parse_judgement({**base, "claims": two}, 3, set())
    bad_cov = {
        **base,
        "claims": [],
        "coverage": [{"venue": "binance", "gold_facts": 1, "covered_facts": 2, "rationale": ""}],
    }
    with pytest.raises(InvalidResponse, match="covered_facts 2 > gold_facts 1"):
        parse_judgement(bad_cov, 0, {"binance"})


def test_judgement_must_cover_exactly_the_venues_with_gold():
    def cov(venue):
        return {"venue": venue, "gold_facts": 1, "covered_facts": 1, "rationale": ""}

    content = {
        "claims": [],
        "coverage": [cov("bybit"), cov("hyperliquid")],
        "behaviour_correct": True,
        "behaviour_rationale": "",
    }
    judged = parse_judgement(content, 0, {"bybit", "hyperliquid"})
    assert sorted(c.venue for c in judged.coverage) == ["bybit", "hyperliquid"]
    with pytest.raises(InvalidResponse, match="expected"):
        parse_judgement(content, 0, {"bybit", "hyperliquid", "dydx"})


def v2(**patch):
    base = {
        "answer": "Bybit uses PostOnly [2]; the Binance chunks do not say.",
        "claims": [{"text": "PostOnly is a timeInForce", "venue": "bybit", "chunks": [2]}],
        "refused": False,
        "asked_clarification": False,
        "partial": True,
        "missing": [{"venue": "binance", "missing": "how post-only is set"}],
    }
    return {**base, **patch}


def test_v2_schema_carries_partial_and_missing_and_v1_stays_as_it_was():
    from crossvenue_rag.generation.schemas import ANSWER_SCHEMA_V1, ANSWER_SCHEMA_V2

    assert set(ANSWER_SCHEMA_V2["required"]) == set(ANSWER_SCHEMA_V1["required"]) | {
        "partial",
        "missing",
    }
    assert "partial" not in ANSWER_SCHEMA_V1["properties"]
    missing_item = ANSWER_SCHEMA_V2["properties"]["missing"]["items"]
    assert missing_item["required"] == ["venue", "missing"]
    assert missing_item["additionalProperties"] is False


def test_a_partial_answer_parses_with_its_claims_and_what_is_missing():
    answer = parse_answer(v2(), 3, schema_version=2)
    assert answer.partial and not answer.refused
    assert answer.claims[0].venue == "bybit"
    assert answer.missing[0].venue == "binance"


def test_v2_answers_must_be_consistent():
    with pytest.raises(InvalidResponse, match="both partial and refused"):
        parse_answer(v2(refused=True, claims=[]), 3, schema_version=2)
    with pytest.raises(InvalidResponse, match="must say in `missing`"):
        parse_answer(v2(missing=[]), 3, schema_version=2)
    with pytest.raises(InvalidResponse, match="lacks"):
        parse_answer({k: v for k, v in v2().items() if k != "partial"}, 3, schema_version=2)


def test_a_refusal_and_a_full_answer_parse_under_v2():
    refusal = parse_answer(
        v2(
            partial=False,
            refused=True,
            claims=[],
            missing=[{"venue": "binance", "missing": "funding interval"}],
        ),
        3,
        schema_version=2,
    )
    assert refusal.refused and not refusal.claims
    full = parse_answer(v2(partial=False, missing=[]), 3, schema_version=2)
    assert not full.partial and not full.missing


def test_a_v1_answer_still_parses_and_reads_as_not_partial():
    v1 = {k: v for k, v in v2().items() if k not in ("partial", "missing")}
    answer = parse_answer(v1, 3, schema_version=1)
    assert answer.partial is False and answer.missing == []


def test_a_partial_answer_without_a_claim_counts_as_a_refusal():
    from crossvenue_rag.generation.schemas import GeneratedAnswer, outcome

    claim = {"text": "x", "venue": "bybit", "chunks": [1]}
    missing = [{"venue": "binance", "missing": "the limit"}]
    base = {"answer": "a", "refused": False, "asked_clarification": False}

    def of(**fields):
        return outcome(GeneratedAnswer.model_validate({**base, **fields}))

    assert of(partial=True, missing=missing, claims=[]) == "refused"
    assert of(partial=True, missing=missing, claims=[claim]) == "partial"
    assert of(refused=True, claims=[]) == "refused"
    assert of(asked_clarification=True, claims=[]) == "asked"
    assert of(claims=[claim]) == "full"
    # A v1 answer has no `partial`; the rule applies to it unchanged.
    assert of(claims=[]) == "full"


def test_the_rule_reaches_the_judge_and_the_counts():
    from crossvenue_rag.generation.answer import judge_message
    from crossvenue_rag.generation.answer_rows import summary_rows
    from crossvenue_rag.generation.schemas import GeneratedAnswer
    from crossvenue_rag.models import Question
    from tests.generation.factories import generation_json, record

    empty_partial = {
        "answer": "Only Bybit's limit is documented.",
        "claims": [],
        "refused": False,
        "asked_clarification": False,
        "partial": True,
        "missing": [{"venue": "binance", "missing": "the limit"}],
    }
    q = Question("cv-0001", "binance vs bybit limit?", "gold", ["binance", "bybit"], "comparison")
    answer = GeneratedAnswer.model_validate(empty_partial)
    assert "refused: true" in judge_message(q, answer, {}, {})
    [row] = summary_rows([record("c", generation=generation_json(answer=empty_partial))], ["c"])
    assert (row.refused, row.partial) == (1, 0)


def test_a_reply_that_fails_its_schema_is_named_by_field_without_the_input_value():
    with pytest.raises(InvalidResponse) as refused:
        parse_answer({"answer": "a", "claims": {"text": "t"}}, 3, schema_version=1)
    message = str(refused.value)
    assert message.startswith("generator response does not match the schema: claims: ")
    assert "input_value" not in message and "{" not in message and "https://" not in message
    with pytest.raises(
        InvalidResponse, match="judge response does not match the schema: "
    ) as judge:
        parse_judgement({"claims": "x"}, 1, set())
    assert "input_value" not in str(judge.value)


def test_an_invalid_reply_is_a_failure_outside_the_program_not_a_user_error():
    """The model, not the user, is at fault; the CLI still prints it without a traceback."""
    with pytest.raises(ExternalFailure) as caught:
        parse_answer({"claims": []}, 3, schema_version=1)
    assert not isinstance(caught.value, UserError)


def test_the_numbers_outside_a_context_are_every_claims_in_claim_order():
    from crossvenue_rag.generation.schemas import Claim, GeneratedAnswer

    claims = [Claim(text="a", venue="x", chunks=[2, 5]), Claim(text="b", venue="x", chunks=[0, 1])]
    answer = GeneratedAnswer(answer="a", claims=claims)
    assert answer.numbers_outside(3) == [5, 0]
    assert answer.numbers_outside(5) == [0]
