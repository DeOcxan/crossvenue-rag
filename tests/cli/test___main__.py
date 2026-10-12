"""Tests for `python -m crossvenue_rag.cli`."""

from __future__ import annotations

import runpy
import subprocess
import sys

import pytest


def test_the_module_runs_the_cli():
    done = subprocess.run(
        [sys.executable, "-m", "crossvenue_rag.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert "reproduce" in done.stdout and "runs-manifest" in done.stdout


def test_the_module_calls_main_in_process(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["crossvenue", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("crossvenue_rag.cli", run_name="__main__")
    assert exit_info.value.code == 0
    assert "runs-manifest" in capsys.readouterr().out
