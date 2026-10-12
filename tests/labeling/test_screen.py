"""Tests for `crossvenue_rag.labeling.screen`."""

from __future__ import annotations

from rich.console import Console

from crossvenue_rag.dataset.journal import Judgment
from crossvenue_rag.labeling.pool import RETEST_SHUFFLE_SEED, retest_candidates
from crossvenue_rag.labeling.screen import (
    ScreenView,
    first_line,
    highlight,
    match_window,
    question_terms,
    render,
)
from crossvenue_rag.models import Question
from tests.labeling.conftest import CHUNKS, quiet, result

FILLER = " ".join(f"word{i:03d}" for i in range(60))  # 479 chars of non-matching words


def test_question_terms_drop_stopwords_and_keep_identifiers():
    terms = question_terms("whats the largest recvWindow binance will accept on a signed order")
    assert "recvwindow" in terms and "binance" in terms and "signed" in terms
    assert "the" not in terms and "whats" not in terms and "will" not in terms
    assert terms[0] == "recvwindow"  # longest first


def test_highlight_is_case_insensitive_and_markup_safe():
    text = highlight("Set [recvWindow](https://x) to at most 60000.", ["recvwindow"])
    assert text.plain == "Set [recvWindow](https://x) to at most 60000."
    styled = [text.plain[span.start : span.end] for span in text.spans]
    assert styled == ["recvWindow"]


def test_first_line_skips_breadcrumb_and_headings():
    r = result("c1", "Limits\n\n## Signed requests\n\nThe recvWindow may not exceed 60000.\nMore.")
    assert first_line(r) == "The recvWindow may not exceed 60000."


def test_first_line_skips_a_code_fence():
    r = result("c1", "Limits\n\n```console\n$ echo -n 'symbol=BTCUSDT'\n```")
    assert first_line(r) == "$ echo -n 'symbol=BTCUSDT'"


def test_compact_row_marks_the_source_and_names_copies():
    question = Question(
        "q1",
        "whats the largest recvWindow",
        "gold",
        ["binance"],
        "fact",
        source_chunk_ids=["c2"],
    )
    candidates = [
        result("c1", "Limits\n\nFirst body line about weights.\nSecond line."),
        result("c2", "Limits\n\nThe recvWindow may not exceed 60000.", copies=1),
    ]
    console = Console(record=True, width=200)
    render(console, question, candidates, {"c1"}, ScreenView(compact=True))
    out = console.export_text()
    assert "source chunks in pool: 1/1 (#2)" in out
    assert "*1" in out
    assert "2s" in out
    assert "(+1 identical: testnet/rest-api-0.md)" in out
    assert "First body line about weights." in out
    assert "Second line." not in out  # compact: first line only


def test_match_window_is_centered_on_the_first_match_and_trimmed_to_words():
    r = result("c1", f"Limits\n\nFirst line.\n{FILLER} Maximum recvWindow is 60000 ms. {FILLER}")
    window = match_window(r, ["recvwindow"], width=120)
    assert window is not None
    assert window.startswith("... ") and window.endswith(" ...")
    inner = window.removeprefix("... ").removesuffix(" ...")
    assert "recvWindow" in inner
    assert len(inner) <= 120
    # Centered: roughly as much context before the match as after it.
    before, _, after = inner.partition("recvWindow")
    assert abs(len(before) - len(after)) <= 20
    # Whole words only at both edges.
    assert inner.split()[0].startswith("word") and len(inner.split()[0]) == 7


def test_match_window_shifts_inward_at_the_ends_of_the_text():
    near_start = result("c1", f"Limits\n\nFirst line.\nrecvWindow first. {FILLER}")
    window = match_window(near_start, ["recvwindow"], width=120)
    assert window is not None and window.startswith("recvWindow first.")
    assert window.endswith(" ...")

    near_end = result("c2", f"Limits\n\nFirst line.\n{FILLER} ends with recvWindow")
    window = match_window(near_end, ["recvwindow"], width=120)
    assert window is not None and window.endswith("ends with recvWindow")
    assert window.startswith("... ")


def test_match_window_takes_the_earliest_of_several_terms():
    r = result("c1", "Limits\n\nFirst line.\nThe order is signed, then recvWindow applies.")
    assert match_window(r, ["recvwindow", "signed"], width=12) == "... is signed, ..."


def test_a_term_only_on_the_first_line_or_no_term_opens_no_window():
    r = result("c1", "Limits\n\nrecvWindow is in the first line only.\nNothing else here.")
    assert match_window(r, ["recvwindow"]) is None
    assert match_window(r, []) is None
    assert match_window(result("c2", "Limits\n\nOne line."), ["recvwindow"]) is None


def test_compact_row_shows_the_window_under_the_first_line():
    question = Question("q1", "whats the maximum recvWindow", "gold", ["binance"], "fact")
    candidates = [
        result("c1", "Limits\n\nSigned requests need a timestamp.\nMaximum recvWindow is 60000."),
        result("c2", "Limits\n\nNothing relevant.\nStill nothing."),
    ]
    console = Console(record=True, width=200)
    render(console, question, candidates, set(), ScreenView(compact=True))
    out = console.export_text()
    assert "Maximum recvWindow is 60000." in out
    assert "Still nothing." not in out


def test_blind_screen_shows_no_marks_no_found_by_and_no_source_markers():
    question = Question(
        "cv-0001", "binance limits", "gold", ["binance"], "fact", source_chunk_ids=["c1"]
    )
    candidates = retest_candidates(
        {c.chunk_id: c for c in CHUNKS},
        "cv-0001",
        Judgment(pool=["c1", "c2", "c3"]),
        RETEST_SHUFFLE_SEED,
    )
    for c in candidates:
        c.component_ranks = {"bm25": 1}

    seen = quiet()
    render(seen, question, candidates, {"c1"}, ScreenView(blind=False))
    open_screen = seen.export_text()
    assert "found by" in open_screen and "source chunks in pool" in open_screen
    assert "* = marked" in open_screen

    blind = quiet()
    render(blind, question, candidates, {"c1"}, ScreenView(blind=True))
    screen = blind.export_text()
    assert "found by" not in screen
    assert "bm25#1" not in screen
    assert "source chunks in pool" not in screen
    assert "marked" not in screen
    assert "*" not in screen


def test_blind_screen_hides_the_question_notes_and_the_open_screen_shows_them():
    question = Question(
        "cv-0001", "binance limits", "gold", ["binance"], "fact", notes="trap: testnet page"
    )
    candidates = [result("c1", "binance")]

    seen = quiet()
    render(seen, question, candidates, set(), ScreenView(blind=False))
    assert "notes: trap: testnet page" in seen.export_text()

    blind = quiet()
    render(blind, question, candidates, set(), ScreenView(blind=True))
    screen = blind.export_text()
    assert "notes" not in screen
    assert "trap: testnet page" not in screen
    assert "binance limits" in screen and "expected:" in screen
