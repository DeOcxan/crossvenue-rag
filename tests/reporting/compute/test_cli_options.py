"""Tests for `reporting.compute.cli_options`, on a command package written under tmp_path."""

from __future__ import annotations

import pytest

from crossvenue_rag.reporting.compute import cli_options
from crossvenue_rag.reporting.sources import Sources
from tests.reporting.conftest import write_file

REGISTRY = """
from . import tools as _tools
from .app import app

app.command("time-it")(_tools.time_command)
app.command()(_tools.draw_sample)
"""
TOOLS = """
from typing import Annotated

import typer

from ..dataset.sampling import SAMPLE_SEED

LOCAL_SIZE = 25


def time_command(
    rounds: Annotated[int, typer.Option("--rounds", help="Passes, 1 or more.")] = 1,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    at_least("--rounds", rounds, 1)


def draw_sample(
    n: Annotated[int, typer.Option("--n")] = LOCAL_SIZE,
    seed: Annotated[int, typer.Option("--seed")] = SAMPLE_SEED,
) -> None:
    pass
"""


@pytest.fixture
def src(tmp_path):
    write_file(tmp_path, "src/crossvenue_rag/cli/__init__.py", REGISTRY)
    write_file(tmp_path, "src/crossvenue_rag/cli/tools.py", TOOLS)
    write_file(tmp_path, "src/crossvenue_rag/dataset/sampling.py", "SAMPLE_SEED = 20260924\n")
    return Sources(tmp_path)


def test_commands_are_found_under_their_registered_names(src):
    assert set(cli_options.commands(src)) == {"time-it", "draw-sample"}


def test_a_literal_default_is_read_from_the_signature(src):
    assert cli_options.option_default(src, "time-it", "--rounds") == 1
    assert cli_options.option_default(src, "time-it", "--force") is False


def test_a_named_default_is_the_literal_its_module_or_import_binds(src):
    assert cli_options.option_default(src, "draw-sample", "--n") == 25
    assert cli_options.option_default(src, "draw-sample", "--seed") == 20260924


def test_the_least_value_is_the_low_of_the_at_least_call(src):
    assert cli_options.least_value(src, "time-it", "--rounds") == 1
    with pytest.raises(ValueError, match="no lower bound of --n"):
        cli_options.least_value(src, "draw-sample", "--n")


def test_an_option_the_command_does_not_declare_is_refused(src):
    with pytest.raises(ValueError, match="declares no --k"):
        cli_options.option_default(src, "time-it", "--k")


def test_an_unregistered_command_is_refused(src):
    with pytest.raises(ValueError, match="registered commands has no 'time'"):
        cli_options.option_default(src, "time", "--rounds")
    with pytest.raises(ValueError, match="registered commands has no 'time'"):
        cli_options.least_value(src, "time", "--rounds")


@pytest.mark.parametrize(
    ("registration", "message"),
    [
        ("app.command()(_other.draw_sample)", "imports of cli/__init__.py has no '_other'"),
        ("app.command()(_tools.absent)", "cli/tools.py defines no absent"),
    ],
)
def test_a_registration_of_an_unknown_module_or_function_is_refused(
    tmp_path, registration, message
):
    write_file(tmp_path, "src/crossvenue_rag/cli/__init__.py", REGISTRY + registration + "\n")
    write_file(tmp_path, "src/crossvenue_rag/cli/tools.py", TOOLS)
    with pytest.raises(ValueError, match=message):
        cli_options.commands(Sources(tmp_path))
