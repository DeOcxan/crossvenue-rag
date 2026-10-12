"""Tests for `cli.render_answers`: an answer prints as written, a table what it is given."""

from __future__ import annotations

from rich.console import Console

from crossvenue_rag.cli import render_answers
from crossvenue_rag.generation.ask import AskResult, answer_question
from tests.generation.offline import ASK_ANSWER, ask_setup


def text(renderable) -> str:
    console = Console(record=True, width=160)
    console.print(renderable)
    return console.export_text()


def _answer(**overrides) -> AskResult:
    citation = {
        "chunk": 1,
        "venue": "bybit",
        "doc_path": "a.md",
        "environment": "mainnet",
        "api_version": None,
        "product": "spot",
        "url": "https://example.invalid/a",
    }
    bare = {**citation, "environment": None, "product": None, "url": ""}
    result = {
        "question": "q",
        "venues_detected": ["bybit"],
        "retriever": "hybrid",
        "generator": "gen",
        "answer": "Use [recvWindow].",
        "claims": [{"venue": "bybit", "text": "a claim", "citations": [citation, bare]}],
        "outcome": "full",
        "refused": False,
        "asked_clarification": False,
        "partial": False,
        "missing": [],
        "model": "m",
        "provider": "",
        "cost_usd": 0.0123,
        "latency_ms": 1234.5,
        "cached": False,
    }
    return AskResult.model_validate({**result, **overrides})


def _printed(result: AskResult) -> str:

    console = Console(record=True, width=160)
    render_answers.print_answer(console, result)
    return console.export_text()


def test_answer_prints_claims_citations_and_cost():
    out = _printed(_answer())
    assert "Use [recvWindow]." in out and "gen over hybrid" in out
    assert "(1) [bybit] a claim" in out and "(mainnet/spot)" in out
    assert "https://example.invalid/a" in out and "via -" in out and "; 1 s;" in out
    assert "cached" not in out


def test_answer_flags_refusals_questions_and_partial_answers():
    assert "refused" in _printed(_answer(outcome="refused", cached=True, venues_detected=[]))
    assert "asks for clarification" in _printed(_answer(outcome="asked"))
    missing = [{"venue": "okx", "missing": "fees"}]
    assert "okx: fees" in _printed(_answer(outcome="partial", missing=missing))
    assert "documentation: -" in _printed(_answer(outcome="partial"))
    assert "full" in _printed(_answer(outcome="full")).splitlines()


def test_estimate_totals_known_costs_and_flags_unknown_ones():
    from crossvenue_rag.generation.estimate import EstimateRow

    rows = [
        EstimateRow("answers", "m", 10, 2, 0.01, 0.08),
        EstimateRow("judge", "j", 4, None, None, None),
    ]
    out = text(render_answers.estimate_table(rows))
    judge_row = next(line for line in out.splitlines() if "judge" in line)
    assert [c.strip() for c in judge_row.split("\u2502")[1:5]] == ["judge", "j", "4", "-"]
    assert "unknown" in out and "+ unknown" in out and "cost estimate" in out
    assert "+ unknown" not in text(render_answers.estimate_table(rows[:1]))


def test_agreement_table_dashes_missing_statistics():
    from crossvenue_rag.generation.grader_agreement import DimensionAgreement

    row = DimensionAgreement(
        dimension="faithful", n=12, agreement=0.8, kappa=None, pearson=0.5, spearman=None
    )
    out = text(render_answers.agreement_table("judge vs human", [row]))
    assert "judge vs human" in out and "faithful" in out and " - " in out


def test_terminal_output_shows_answer_claims_citations_and_refusal(tmp_path):
    cfg, llm, retriever, _ = ask_setup(tmp_path)
    console = Console(record=True, width=160)
    render_answers.print_answer(
        console,
        answer_question(
            cfg,
            retriever,
            llm,
            "binance testnet ban?",
            retriever_name="venue_split",
            generator_name="gemini",
        ),
    )
    screen = console.export_text()
    assert "Binance bans scale" in screen
    assert "(1) [binance] bans scale 2m to 3d" in screen
    assert "testnet/rest-api.md" in screen and "https://t" in screen

    refused = {**ASK_ANSWER, "answer": "Not in the chunks.", "claims": [], "refused": True}
    cfg, llm, retriever, _ = ask_setup(tmp_path / "r", refused)
    console = Console(record=True, width=160)
    render_answers.print_answer(
        console,
        answer_question(
            cfg,
            retriever,
            llm,
            "binance funding?",
            retriever_name="venue_split",
            generator_name="gemini",
        ),
    )
    assert "refused" in console.export_text()


def test_a_partial_answer_with_claims_names_what_is_missing(tmp_path):
    partial = {**ASK_ANSWER, "partial": True, "missing": [{"venue": "bybit", "missing": "bans"}]}
    cfg, llm, retriever, _ = ask_setup(tmp_path, partial)
    console = Console(record=True, width=120)
    render_answers.print_answer(
        console,
        answer_question(
            cfg,
            retriever,
            llm,
            "binance and bybit ban length?",
            retriever_name="venue_split",
            generator_name="gemini",
        ),
    )
    assert "partial, not in the documentation: bybit: bans" in console.export_text()


def test_an_answer_prints_the_brackets_the_model_and_the_provider_wrote_as_they_are():
    missing = [{"venue": "okx", "missing": "rate limit [/x]"}]
    claims = [{"venue": "bybit", "text": "the [bold]fee[/bold] table", "citations": []}]
    out = _printed(
        _answer(
            outcome="partial",
            missing=missing,
            claims=claims,
            answer="[/a] then [bold]b[/bold]",
            generator="[/g]",
            model="[/m]",
            provider="[bold]p[/bold]",
        )
    )
    assert "okx: rate limit [/x]" in out and "(1) [bybit] the [bold]fee[/bold] table" in out
    assert "[/a] then [bold]b[/bold]" in out and "[/g] over hybrid" in out
    assert "model [/m] via [bold]p[/bold]" in out
