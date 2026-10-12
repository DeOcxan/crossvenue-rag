"""Tests for `crossvenue_rag.cli.errors`."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError
from rich.console import Console

from crossvenue_rag import cli
from crossvenue_rag.cli.errors import friendly_errors
from crossvenue_rag.config_loading import load_config
from crossvenue_rag.errors import UserError


def _failure(exc: Exception) -> str:
    console = Console(width=200, no_color=True)
    with console.capture() as capture, pytest.raises(SystemExit), friendly_errors(console):
        raise exc
    return capture.get()


def test_a_user_error_becomes_one_readable_line_with_its_hint():
    """Every command has prerequisites, so a failure can be a missing step, not a bug,
    and a traceback shows code when the fix is a missing step."""
    console = Console(width=100, no_color=True)
    with (
        console.capture() as capture,
        pytest.raises(SystemExit) as exit_info,
        friendly_errors(console),
    ):
        raise UserError("no corpus statistics", hint="run `crossvenue ingest` first")
    assert exit_info.value.code == 1
    out = capture.get()
    assert "Error: no corpus statistics" in out
    assert "run `crossvenue ingest` first" in out
    assert "CROSSVENUE_DEBUG" in out


@pytest.mark.parametrize(
    "exc",
    [KeyError("a bug"), ValueError("a bug"), FileNotFoundError("a bug"), RuntimeError("a bug")],
)
def test_anything_but_a_user_error_keeps_its_traceback(exc):
    """A defect is not a user error, and discarding its trace loses the useful part."""
    with pytest.raises(type(exc)), friendly_errors(Console()):
        raise exc


def test_a_validation_error_outside_the_config_keeps_its_traceback():
    class Model(BaseModel):
        k: int

    with pytest.raises(ValidationError), friendly_errors(Console()):
        Model.model_validate({"k": "many"})


def test_an_invalid_config_names_its_file_and_field(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval:\n  k: many\n", encoding="utf-8")
    with pytest.raises(UserError) as info:
        load_config(path)
    out = _failure(info.value)
    assert f"invalid {path}: retrieval.k:" in out and "docs/reference/config.md" in out


def test_a_missing_config_and_a_malformed_override_are_user_errors(tmp_path):
    with pytest.raises(UserError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    path = tmp_path / "ok.yaml"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(UserError, match=r"section\.field=value"):
        load_config(path, ["retrieval.k"])


def test_an_unreachable_database_says_how_to_start_it():
    class OperationalError(Exception):
        pass

    out = _failure(OperationalError("connection refused"))
    assert "cannot reach Postgres: connection refused" in out and "make db" in out


def test_a_failed_model_call_prints_its_message_and_hint_without_a_traceback():
    from crossvenue_rag.generation.openrouter import ModelCallError

    out = _failure(ModelCallError("m: HTTP 502 after 3 retries: upstream", hint="re-run"))
    assert "Error: m: HTTP 502 after 3 retries: upstream" in out and "re-run" in out
    assert "Traceback" not in out


def test_a_value_error_raised_inside_a_command_keeps_its_traceback(config_file, monkeypatch):
    def defect(_cfg):
        raise ValueError("a defect inside the command")

    monkeypatch.setattr("crossvenue_rag.cli.corpus._print_stats", defect)
    monkeypatch.setattr("sys.argv", ["crossvenue", "stats", "--config", str(config_file)])
    with pytest.raises(ValueError, match="a defect inside the command"):
        cli.main()


def test_debug_mode_reraises_expected_failures(monkeypatch):
    monkeypatch.setenv("CROSSVENUE_DEBUG", "1")
    with pytest.raises(UserError, match="bad"), friendly_errors(Console()):
        raise UserError("bad")


@pytest.mark.parametrize("typed", ["[/x]", "[bold]"])
def test_a_message_and_hint_holding_brackets_print_as_written(typed):
    """A message quotes what the user typed; rich markup would raise on `[/x]`
    and drop `[bold]`."""
    out = _failure(UserError(f"--venue {typed!r} is not a venue", hint=f"pass {typed}"))
    assert f"Error: --venue '{typed}' is not a venue" in out and f"pass {typed}" in out


@pytest.mark.parametrize("typed", ["[/x]", "[bold]"])
def test_count_term_names_a_bracketed_unknown_venue_as_typed(
    typed, config_file, monkeypatch, capsys
):
    monkeypatch.setattr(
        "sys.argv",
        ["crossvenue", "count-term", "x", "--venue", typed, "--config", str(config_file)],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 1
    assert f"--venue '{typed}' is not a venue of the corpus" in capsys.readouterr().out
