"""Tests for `evaluation.machine`: the busy check, the machine state and machine.json."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from crossvenue_rag.evaluation.machine import (
    MACHINE_FILE,
    LoadDuring,
    MachineRecord,
    MachineState,
    Process,
    busy_reasons,
    machine_json,
    machine_state,
    read_machine,
    recordable,
)
from tests.conftest import committed_with

IDLE = MachineState(
    measured_at="t",
    cpu_count=8,
    load_average=(0.5, 0.5, 0.5),
    competing_processes=(),
    busy_processes=(),
)


def test_a_busy_machine_is_named_and_an_idle_one_is_not():
    idle = MachineState(
        measured_at="t",
        cpu_count=8,
        load_average=(1.0, 1.0, 1.0),
        competing_processes=(),
        busy_processes=(),
    )
    assert busy_reasons(idle) == []
    busy = idle.model_copy(
        update={
            "load_average": (6.5, 3.0, 2.0),
            "competing_processes": (Process(pid=7, cpu_percent=90.0, command="crossvenue eval"),),
        }
    )
    reasons = busy_reasons(busy)
    assert "1-minute load average 6.50 on 8 CPUs" in reasons
    assert "pid 7: crossvenue eval" in reasons


def test_machine_state_skips_this_process_and_its_parents_and_sorts_out_the_rest(monkeypatch):
    me, parent = os.getpid(), os.getppid()
    table = "\n".join(
        [
            f"{me} {parent} 99.0 python -m pytest",  # this process
            f"{parent} 1 80.0 uv run pytest",  # the `uv run` that started it
            "11 1 20.0 crossvenue eval",  # a competitor
            "12 1 1.0 crossvenue rate",  # idle: not a competitor
            "13 1 70.0 browser",  # busy but not a run
            "14 1 90.0 zsh -c crossvenue bench-rerank",  # the wrapping shell
            "garbage",
        ]
    )
    monkeypatch.setattr(
        "crossvenue_rag.evaluation.machine.subprocess.run",
        lambda *a, **kw: subprocess.CompletedProcess(a, 0, stdout=table),
    )
    monkeypatch.setattr(
        "crossvenue_rag.evaluation.machine.os.getloadavg", lambda: (1.234, 2.0, 3.0)
    )
    state = machine_state()
    assert state.load_average == (1.23, 2.0, 3.0)
    assert [p.pid for p in state.competing_processes] == [11]
    assert [p.pid for p in state.busy_processes] == [13]


@pytest.mark.parametrize("run", committed_with(MACHINE_FILE), ids=lambda p: p.name)
def test_every_committed_machine_json_round_trips_byte_for_byte(run):
    text = (run / "machine.json").read_text(encoding="utf-8")
    assert machine_json(read_machine(run)) == text


def test_a_bench_without_loads_records_an_empty_load_object(tmp_path):
    state = recordable(IDLE)
    record = MachineRecord(before=state, after=state, discarded_pairs=0, load1_during=LoadDuring())
    assert json.loads(machine_json(record))["load1_during"] == {}
    (tmp_path / "machine.json").write_text(machine_json(record), encoding="utf-8")
    assert read_machine(tmp_path) == record


def test_a_process_table_is_classified_without_running_ps():
    from crossvenue_rag.evaluation.machine import classify_processes, parse_process_table

    ps = (
        "  1     0  0.0 /sbin/launchd\n"
        " 10     1  0.0 zsh\n"
        " 20    10 90.0 uv run crossvenue bench-rerank\n"
        " 30     1 12.0 python -m pytest\n"
        " 40     1 80.0 /Applications/Browser\n"
        "not a row\n"
        "  PID  PPID  %CPU COMMAND\n"
        " 50     1   -   defunct\n"
    )
    table = parse_process_table(ps)
    assert [row.pid for row in table] == [1, 10, 20, 30, 40]
    competitors, busy = classify_processes(table, own_pid=20)
    assert [p.pid for p in competitors] == [30]
    assert [p.pid for p in busy] == [40]
