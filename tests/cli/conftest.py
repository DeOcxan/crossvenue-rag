"""Fixtures and helpers of the CLI tests: a config file with every path but `paths.sources`
inside `tmp_path`, its index version, the in-process command line and the text a console prints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner, Result

from crossvenue_rag.cli import app
from crossvenue_rag.config import Config
from crossvenue_rag.index_version import write_index_meta
from tests.conftest import build_corpus


@pytest.fixture
def config_file(cfg: Config, tmp_path: Path) -> Path:
    """`configs/default.yaml` with the LLM cache, the index and every path but `paths.sources`
    under `tmp_path`; the fixture corpus is ingested and its index metadata written."""
    cfg.generation.cache_dir = tmp_path / "cache"
    cfg.paths.sources = Path("data/sources.yaml")
    chunks = build_corpus(cfg)
    write_index_meta(cfg, len(chunks))
    path = tmp_path / "config.yaml"
    path.write_text(cfg.to_yaml(), encoding="utf-8")
    return path


def current_version(config_file: Path) -> str:
    from crossvenue_rag.config_loading import load_config
    from crossvenue_rag.index_version import require_current_index

    return require_current_index(load_config(config_file))


def invoke(*args: str) -> Result:
    """Runs the `crossvenue` command line with `args` in-process."""
    return CliRunner().invoke(app, list(args))


def recorded_text(write: Callable[[Console], object]) -> str:
    """The text `write` prints on a recording console 160 columns wide."""
    console = Console(record=True, width=160)
    write(console)
    return console.export_text()


def printed_text(renderable: object) -> str:
    """The text a console 160 columns wide prints for `renderable`."""
    return recorded_text(lambda console: console.print(renderable))
