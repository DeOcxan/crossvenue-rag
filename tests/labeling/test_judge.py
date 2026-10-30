"""Tests for `crossvenue_rag.labeling.judge`."""

from __future__ import annotations

from crossvenue_rag.labeling.judge import judge
from crossvenue_rag.labeling.screen import ScreenView
from crossvenue_rag.models import Question
from tests.labeling.conftest import Script, comparison, quiet, result, results


def test_fully_covered_comparison_saves_without_asking():
    judgement = judge(quiet(), comparison(), results(), set(), Script("1 2"))
    assert judgement is not None
    assert judgement.relevant == ["c1", "c2"]
    assert judgement.venue_coverage == {"binance": "covered", "bybit": "covered"}


def test_an_uncovered_venue_must_be_accounted_for_and_back_returns_to_selection():
    console = quiet()
    script = Script("1", "x", "b", "1", "a")
    judgement = judge(console, comparison(), results(), set(), script)
    assert judgement is not None
    assert judgement.relevant == ["c1"]
    assert judgement.venue_coverage == {"binance": "covered", "bybit": "absent_in_docs"}
    out = console.export_text()
    assert "nothing marked from bybit: 1 bybit candidate in the pool (#2)" in out
    assert "Expected b, a or m." in out


def test_none_relevant_on_a_comparison_asks_for_every_venue():
    judgement = judge(quiet(), comparison(), results(), set(), Script("n", "m", "a"))
    assert judgement is not None
    assert judgement.relevant == []
    assert judgement.venue_coverage == {
        "binance": "missing_from_pool",
        "bybit": "absent_in_docs",
    }


def test_non_comparison_and_unanswerable_questions_are_not_asked():
    fact = Question("cv-0002", "binance limit", "silver", ["binance"], "fact", category="fact")
    judged = judge(quiet(), fact, results(), set(), Script("1"))
    assert judged is not None and judged.venue_coverage == {}
    unanswerable = comparison(is_answerable=False, expected_behavior="refuse")
    judged = judge(quiet(), unanswerable, results(), set(), Script("n"))
    assert judged is not None and judged.venue_coverage == {}


def test_version_sensitive_shows_marked_environments_before_saving():
    question = Question(
        "cv-0003",
        "is the ip ban on binance testnet the same",
        "silver",
        ["binance"],
        "fact",
        category="version_sensitive",
        notes="testnet is the trap",
    )
    console = quiet()
    judgement = judge(console, question, results(), set(), Script("3", "b", "1", ""))
    assert judgement is not None and judgement.relevant == ["c1"]
    out = console.export_text()
    assert "marked chunks: environment / api_version" in out
    assert "testnet/rest-api.md" in out  # shown for the first, withdrawn pick
    assert "notes: testnet is the trap" in out


def test_ask_picks_returns_the_selection_the_marker_for_none_or_skip():
    from crossvenue_rag.labeling.judge import NONE_RELEVANT, ask_picks

    def no_toggle() -> None:
        raise AssertionError("not toggled")

    assert ask_picks(quiet(), comparison(), results(), Script("2 1"), "relevant", no_toggle) == [
        1,
        2,
    ]
    assert ask_picks(quiet(), comparison(), results(), Script("n"), "relevant", no_toggle) == (
        NONE_RELEVANT
    )
    assert ask_picks(quiet(), comparison(), results(), Script(""), "relevant", no_toggle) is None


def test_ask_picks_toggles_views_and_reprompts_on_bad_input():
    from crossvenue_rag.labeling.judge import ask_picks

    toggles: list[int] = []
    console = quiet()
    picks = ask_picks(
        console,
        comparison(),
        results(),
        Script("f", "v9", "v1", "x", "2"),
        "relevant",
        lambda: toggles.append(1),
    )
    assert picks == [2]
    assert toggles == [1]
    out = console.export_text()
    assert f"Expected v1 .. v{len(results())}." in out
    assert "Expected numbers like `1 4 7` or `2-5`." in out


def test_a_reversed_range_or_no_number_is_rejected_and_asked_again():
    from crossvenue_rag.labeling.judge import NONE_RELEVANT, ask_picks, parse_selection

    assert parse_selection("5-2", 10) is None
    assert parse_selection(",", 10) is None
    assert parse_selection("1-3,5", 10) == [1, 2, 3, 5]

    def no_toggle() -> None:
        raise AssertionError("not toggled")

    console = quiet()
    picks = ask_picks(
        console, comparison(), results(), Script("5-2", ",", "1-3"), "relevant", no_toggle
    )
    assert picks == [1, 2, 3]
    assert console.export_text().count("Expected numbers like `1 4 7` or `2-5`.") == 2
    assert ask_picks(quiet(), comparison(), results(), Script("n"), "relevant", no_toggle) == (
        NONE_RELEVANT
    )


def test_between_draws_on_every_redraw_of_the_screen():
    drawn: list[str] = []
    console = quiet()
    candidates = [result("a", "text")]
    question = Question("q1", "a question?", "gold", ["binance"], "fact", category="fact")
    view = ScreenView(between=lambda: drawn.append("x"))
    judge(console, question, candidates, set(), Script("f", ""), view=view)
    assert drawn == ["x", "x"]
